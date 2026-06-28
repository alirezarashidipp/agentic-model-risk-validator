from __future__ import annotations

import re
from pathlib import Path

from app.schemas.agent_outputs import ReportWriterOutput


def _finding_lines(report: ReportWriterOutput) -> str:
    if not report.key_findings:
        return "- No findings recorded."
    return "\n".join(
        f"- **{item.severity.value} | {item.category}: {item.title}**. "
        f"Evidence: {item.evidence} Recommendation: {item.recommendation}"
        for item in report.key_findings
    )


def _list_lines(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None recorded."


def generate_markdown_report(
    report: ReportWriterOutput,
    output_path: str | Path,
    *,
    artifacts: dict | None = None,
) -> Path:
    artifacts = artifacts or {}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact_lines = []
    for name, value in artifacts.items():
        if isinstance(value, dict) and value.get("available") and value.get("path"):
            artifact_path = Path(value["path"])
            display_path = (
                f"figures/{artifact_path.name}"
                if artifact_path.parent.name == "figures"
                else artifact_path.name
            )
            artifact_lines.append(f"- {name}: `{display_path}`")
    artifact_section = "\n".join(artifact_lines) if artifact_lines else "- No chart artifacts were generated."

    markdown = f"""# Model Validation Report

## 1. Executive Summary
{report.executive_summary}

## 2. Model Overview
{report.model_overview}

## 3. Files Reviewed
{report.files_reviewed}

## 4. Validation Scope
{report.validation_scope}

## 5. Data Quality Review
{report.data_quality_review}

## 6. Methodology Review
{report.methodology_review}

## 7. Performance Review
{report.performance_review}

## 8. Explainability Review
{report.explainability_review}

## 9. Bias and Fairness Review
{report.bias_and_fairness_review}

## 10. Governance Review
{report.governance_review}

## 11. Key Findings
{_finding_lines(report)}

## 12. Challenge Questions
{_list_lines(report.challenge_questions)}

## 13. Risk Rating
**{report.risk_rating.value}**

## 14. Recommendations
{_list_lines(report.recommendations)}

## 15. Final Validation Decision
**{report.final_validation_decision.value}**

## Deterministic Artifacts
{artifact_section}
"""
    path.write_text(markdown, encoding="utf-8")
    return path


def generate_pdf_report(markdown_path: str | Path, output_path: str | Path) -> dict:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except Exception as exc:
        return {"available": False, "reason": f"reportlab is unavailable: {exc}"}

    markdown = Path(markdown_path).read_text(encoding="utf-8")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            story.append(Paragraph(line.removeprefix("# "), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(line.removeprefix("## "), styles["Heading2"]))
        else:
            clean = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            story.append(Paragraph(clean, styles["BodyText"]))
    SimpleDocTemplate(str(path), pagesize=letter).build(story)
    return {"available": True, "path": str(path)}
