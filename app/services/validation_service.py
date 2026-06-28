from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents.workflow import run_validation_agents
from app.config import settings
from app.schemas.agent_outputs import FinalDecision, ReportWriterOutput, RiskRating
from app.schemas.jobs import JobStatus, ProblemType, ValidationRequest, ValidationResponse
from app.services.job_service import (
    create_job,
    discover_uploaded_files,
    job_dir,
    report_dir,
    upload_dir,
)
from app.services.risk_policy import assess_metrics_quality, detect_overfitting, determine_policy_result
from app.storage.db import get_job, update_job_record
from app.tools.data_tools import generate_sample_credit_data, infer_problem_type, load_dataset, profile_dataset
from app.tools.fairness_tools import analyze_fairness
from app.tools.json_utils import to_jsonable
from app.tools.model_tools import evaluate_model, load_model, train_baseline_model
from app.tools.plot_tools import (
    generate_confusion_matrix_plot,
    generate_feature_importance_plot,
    generate_shap_summary_plot,
)
from app.tools.report_tools import generate_markdown_report, generate_pdf_report


def _read_text(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".py", ".json", ".csv"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix == ".docx":
            from docx import Document

            document = Document(str(path))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception as exc:
        return f"Unable to extract text from {path.name}: {exc}"
    return None


def _review_documentation(text: str | None) -> dict:
    if not text:
        return {"available": False, "summary": "No documentation text was available.", "checks": {}}
    lower = text.lower()
    checks = {
        "model_purpose": any(word in lower for word in ["purpose", "objective", "use case", "business use"]),
        "assumptions": "assumption" in lower,
        "limitations": any(word in lower for word in ["limitation", "constraint"]),
        "data_description": any(word in lower for word in ["data", "dataset", "feature"]),
        "methodology": any(word in lower for word in ["methodology", "algorithm", "model selection", "training"]),
        "monitoring": any(word in lower for word in ["monitoring", "drift", "threshold", "retraining"]),
        "governance": any(word in lower for word in ["governance", "owner", "approval", "oversight"]),
        "approval_evidence": any(word in lower for word in ["approved", "approval", "sign-off", "signoff"]),
    }
    covered = sum(checks.values())
    return {
        "available": True,
        "summary": f"Documentation text was extracted. {covered} of {len(checks)} expected evidence areas were detected by keyword review.",
        "checks": checks,
        "character_count": len(text),
    }


def _review_governance(documentation_text: str | None, training_code_text: str | None) -> dict:
    combined = "\n".join(text for text in [documentation_text, training_code_text] if text).lower()
    return {
        "monitoring_plan_present": any(word in combined for word in ["monitoring", "drift", "threshold", "retraining"]),
        "approval_evidence_present": any(word in combined for word in ["approved", "approval", "sign-off", "signoff"]),
        "owner_present": any(word in combined for word in ["owner", "model owner", "business owner"]),
        "versioning_present": any(word in combined for word in ["version", "model version", "release"]),
        "human_oversight_present": any(word in combined for word in ["human oversight", "manual review", "override"]),
    }


def _select_target_column(columns: list[str], requested: str | None) -> str:
    if requested:
        if requested not in columns:
            raise ValueError(f"Target column '{requested}' is not present in the dataset.")
        return requested
    for candidate in ("defaulted", "target", "label", "outcome"):
        if candidate in columns:
            return candidate
    return columns[-1]


def _files_reviewed(files: dict[str, Any], sample_dataset_used: bool) -> list[str]:
    reviewed = []
    for name, path in files.items():
        if path:
            reviewed.append(f"{name}: {Path(path).name}")
    if sample_dataset_used:
        reviewed.append("dataset: generated sample credit risk dataset")
    return reviewed


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def run_validation(request: ValidationRequest) -> ValidationResponse:
    job_id = request.job_id or create_job()
    if get_job(job_id) is None:
        raise ValueError(f"Job '{job_id}' does not exist.")

    update_job_record(job_id, status=JobStatus.RUNNING)
    timeline_path = job_dir(job_id) / "agent_timeline.json"
    sample_dataset_used = False

    try:
        files = discover_uploaded_files(job_id)
        if files.dataset is None:
            sample_dataset_used = True
            dataset_path = upload_dir(job_id) / "sample_credit_risk.csv"
            generate_sample_credit_data(dataset_path)
            files.dataset = dataset_path

        df = load_dataset(files.dataset)
        target_column = _select_target_column(list(df.columns), request.target_column)
        problem_type = request.problem_type.value
        if problem_type == ProblemType.AUTO.value:
            problem_type = infer_problem_type(df, target_column)

        profile = profile_dataset(df, target_column)
        baseline = train_baseline_model(df, target_column, problem_type)
        active_model = baseline["model"]
        model_source = "baseline_model"
        model_error = None

        if files.model is not None:
            try:
                uploaded_model = load_model(files.model)
                uploaded_metrics = evaluate_model(
                    uploaded_model,
                    baseline["X_test"],
                    baseline["y_test"],
                    problem_type,
                )
                active_model = uploaded_model
                model_source = "uploaded_model"
                test_metrics = uploaded_metrics
                train_metrics = {}
            except Exception as exc:
                model_error = str(exc)
                test_metrics = baseline["test_metrics"]
                train_metrics = baseline["train_metrics"]
        else:
            test_metrics = baseline["test_metrics"]
            train_metrics = baseline["train_metrics"]

        overfitting = detect_overfitting(train_metrics, test_metrics, problem_type) if train_metrics else {
            "severe_overfitting": False,
            "overfitting_evidence": "Train metrics for uploaded model are unavailable.",
        }

        documentation_text = _read_text(files.documentation)
        training_code_text = _read_text(files.training_code)
        metrics_text = _read_text(files.metrics_file)
        documentation_review = _review_documentation(documentation_text)
        governance_review = _review_governance(documentation_text, training_code_text)
        metrics_quality = assess_metrics_quality(
            problem_type=problem_type,
            test_metrics=test_metrics,
            uploaded_metrics_text=metrics_text,
        )

        figures_dir = report_dir(job_id) / "figures"
        confusion_plot = (
            generate_confusion_matrix_plot(test_metrics, figures_dir / "confusion_matrix.png")
            if problem_type == "classification"
            else {"available": False, "reason": "confusion matrix applies only to classification"}
        )
        feature_importance_plot = generate_feature_importance_plot(
            active_model,
            figures_dir / "feature_importance.png",
        )
        shap_plot = generate_shap_summary_plot(
            active_model,
            baseline["X_test"],
            figures_dir / "shap_summary.png",
        )

        fairness = analyze_fairness(
            df,
            target_column,
            active_model,
            baseline["X_test"],
            baseline["y_test"],
            problem_type,
        )

        available_inputs = {
            "dataset": files.dataset is not None,
            "model": files.model is not None and model_error is None,
            "training_code": files.training_code is not None,
            "metrics_file": files.metrics_file is not None,
            "documentation": files.documentation is not None and bool(documentation_text),
        }
        performance = {
            "problem_type": problem_type,
            "model_source": model_source,
            "model_error": model_error,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            **overfitting,
        }
        explainability = {
            "feature_importance": feature_importance_plot,
            "shap": shap_plot,
        }
        files_as_dict = files.model_dump()
        context = {
            "job_id": job_id,
            "target_column": target_column,
            "problem_type": problem_type,
            "sample_dataset_used": sample_dataset_used,
            "available_inputs": available_inputs,
            "files_reviewed": _files_reviewed(files_as_dict, sample_dataset_used),
            "data_profile": profile,
            "performance": performance,
            "documentation_review": documentation_review,
            "governance_review": governance_review,
            "metrics_quality": metrics_quality,
            "explainability": explainability,
            "fairness": fairness,
        }
        context["policy_result"] = determine_policy_result(context)

        deterministic_path = job_dir(job_id) / "deterministic_results.json"
        _write_json(deterministic_path, context)

        agent_outputs = run_validation_agents(context, timeline_path)
        report_output: ReportWriterOutput = agent_outputs["report_writer"]

        policy = context["policy_result"]
        report_output = report_output.model_copy(
            update={
                "risk_rating": RiskRating(policy["risk_rating"]),
                "final_validation_decision": FinalDecision(policy["final_decision"]),
            }
        )
        agent_outputs["report_writer"] = report_output

        _write_json(
            job_dir(job_id) / "agent_outputs.json",
            {name: output.model_dump(mode="json") for name, output in agent_outputs.items()},
        )

        markdown_path = generate_markdown_report(
            report_output,
            report_dir(job_id) / "validation_report.md",
            artifacts={
                "confusion_matrix": confusion_plot,
                "feature_importance": feature_importance_plot,
                "shap_summary": shap_plot,
            },
        )
        generate_pdf_report(markdown_path, report_dir(job_id) / "validation_report.pdf")
        report_markdown = markdown_path.read_text(encoding="utf-8")

        update_job_record(
            job_id,
            status=JobStatus.COMPLETED,
            target_column=target_column,
            problem_type=problem_type,
            risk_rating=policy["risk_rating"],
            final_decision=policy["final_decision"],
            report_path=markdown_path,
        )

        return ValidationResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            risk_rating=policy["risk_rating"],
            final_decision=policy["final_decision"],
            report_path=str(markdown_path),
            report_markdown=report_markdown,
            timeline_path=str(timeline_path),
            findings=[finding.model_dump(mode="json") for finding in report_output.key_findings],
        )
    except Exception as exc:
        update_job_record(job_id, status=JobStatus.FAILED, error=str(exc))
        raise
