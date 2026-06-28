from __future__ import annotations

from collections.abc import Iterable

from app.schemas.agent_outputs import (
    AgentAssessment,
    DataQualityReviewOutput,
    DocumentationReviewOutput,
    EvidenceGap,
    ExplainabilityReviewOutput,
    FairnessReviewOutput,
    FinalDecision,
    Finding,
    GovernanceReviewOutput,
    MethodologyReviewOutput,
    PerformanceReviewOutput,
    PlannerOutput,
    ReportWriterOutput,
    RiskRating,
    ValidationStep,
)


def finding(
    title: str,
    severity: RiskRating,
    category: str,
    evidence: str,
    recommendation: str,
) -> Finding:
    return Finding(
        title=title,
        severity=severity,
        category=category,
        evidence=evidence,
        recommendation=recommendation,
    )


def max_risk(risks: Iterable[RiskRating]) -> RiskRating:
    order = {RiskRating.LOW: 0, RiskRating.MEDIUM: 1, RiskRating.HIGH: 2}
    return max(risks, key=lambda risk: order[risk], default=RiskRating.LOW)


def planner_fallback(context: dict) -> PlannerOutput:
    available = context.get("available_inputs", {})
    steps = [
        ValidationStep(
            step_name="Data quality profiling",
            possible=bool(available.get("dataset")),
            rationale="Requires a structured dataset.",
            required_inputs=["dataset"],
        ),
        ValidationStep(
            step_name="Baseline model performance",
            possible=bool(available.get("dataset") and context.get("target_column")),
            rationale="Requires dataset and target column.",
            required_inputs=["dataset", "target_column"],
        ),
        ValidationStep(
            step_name="Documentation review",
            possible=bool(available.get("documentation")),
            rationale="Checks supplied model documentation evidence.",
            required_inputs=["documentation"],
        ),
        ValidationStep(
            step_name="Explainability review",
            possible=bool(context.get("explainability", {}).get("feature_importance", {}).get("available")),
            rationale="Uses generated feature-importance or SHAP artifacts.",
            required_inputs=["trained model or baseline model"],
        ),
        ValidationStep(
            step_name="Fairness review",
            possible=bool(context.get("fairness", {}).get("available")),
            rationale="Requires sensitive or proxy group attributes in the evaluation split.",
            required_inputs=["sensitive attributes"],
        ),
    ]
    gaps = []
    if not available.get("documentation"):
        gaps.append(
            EvidenceGap(
                area="Documentation",
                missing_evidence="No model documentation file was supplied.",
                impact="Purpose, assumptions, governance, monitoring, and approval evidence cannot be fully validated.",
            )
        )
    if not available.get("training_code"):
        gaps.append(
            EvidenceGap(
                area="Methodology",
                missing_evidence="No training code was supplied.",
                impact="Feature engineering, split design, and hyperparameter decisions cannot be independently traced.",
            )
        )
    return PlannerOutput(
        summary="Validation plan generated from available files and deterministic analysis outputs.",
        possible_steps=steps,
        missing_evidence=gaps,
        assumptions=["Baseline metrics are calculated by Python tools and used as benchmark evidence."],
    )


def documentation_fallback(context: dict) -> DocumentationReviewOutput:
    review = context.get("documentation_review", {})
    checks = review.get("checks", {})
    findings = []
    missing = [name for name, present in checks.items() if not present]
    if not review.get("available"):
        findings.append(
            finding(
                "Model documentation not supplied",
                RiskRating.MEDIUM,
                "Documentation",
                "No documentation file was uploaded.",
                "Provide model purpose, scope, data description, methodology, limitations, monitoring, governance, and approval evidence.",
            )
        )
    elif missing:
        findings.append(
            finding(
                "Documentation is incomplete",
                RiskRating.MEDIUM,
                "Documentation",
                f"Missing or weak documentation areas: {', '.join(missing)}.",
                "Update the model document to cover each missing validation evidence area.",
            )
        )
    return DocumentationReviewOutput(
        summary=review.get("summary", "Documentation evidence was reviewed."),
        risk_rating=max_risk([item.severity for item in findings]),
        findings=findings,
        challenge_questions=[
            "What business decision does the model support and where is this documented?",
            "What assumptions and limitations were accepted by model governance?",
        ],
        recommendations=["Maintain a complete model document with owner, version, scope, monitoring, and approvals."],
        limitations=[] if review.get("available") else ["Documentation review is limited because no document was supplied."],
    )


def data_quality_fallback(context: dict) -> DataQualityReviewOutput:
    profile = context.get("data_profile", {})
    findings = []
    missing = profile.get("missing_values", {})
    duplicates = profile.get("duplicates", {})
    outliers = profile.get("outliers", {})
    imbalance = profile.get("class_imbalance", {})
    leakage = profile.get("target_leakage", {})

    if leakage.get("suspected_leakage"):
        findings.append(
            finding(
                "Potential target leakage detected",
                RiskRating.HIGH,
                "Data Quality",
                f"Suspicious features: {leakage.get('suspicious_features', [])}.",
                "Remove or justify leakage-suspect features and rerun validation before approval.",
            )
        )
    if missing.get("columns_with_missing", 0) > 0:
        severity = RiskRating.MEDIUM if missing.get("total_missing_cells", 0) > 0 else RiskRating.LOW
        findings.append(
            finding(
                "Missing values require treatment evidence",
                severity,
                "Data Quality",
                f"{missing.get('columns_with_missing')} columns contain missing values.",
                "Document imputation or exclusion logic and monitor missingness drift.",
            )
        )
    if duplicates.get("duplicate_rows", 0) > 0:
        findings.append(
            finding(
                "Duplicate rows detected",
                RiskRating.MEDIUM,
                "Data Quality",
                f"{duplicates.get('duplicate_rows')} duplicate rows were found.",
                "Remove duplicate records or evidence why duplicates are valid observations.",
            )
        )
    if outliers.get("columns_with_outliers", 0) > 0:
        findings.append(
            finding(
                "Outliers detected in numeric features",
                RiskRating.MEDIUM,
                "Data Quality",
                f"{outliers.get('columns_with_outliers')} numeric columns contain IQR outliers.",
                "Review outlier treatment and confirm values are plausible.",
            )
        )
    if imbalance.get("is_imbalanced"):
        findings.append(
            finding(
                "Target class imbalance detected",
                RiskRating.MEDIUM,
                "Data Quality",
                f"Minority class rate is {imbalance.get('minority_class_rate')}.",
                "Use imbalance-aware metrics, sampling, or class weights and monitor minority-class performance.",
            )
        )
    if not findings:
        findings.append(
            finding(
                "No material data quality exception identified",
                RiskRating.LOW,
                "Data Quality",
                "Profiling did not identify leakage, duplicates, or high-severity quality issues.",
                "Continue routine data-quality monitoring.",
            )
        )
    return DataQualityReviewOutput(
        summary=f"Dataset has {profile.get('row_count')} rows and {profile.get('column_count')} columns.",
        risk_rating=max_risk([item.severity for item in findings]),
        findings=findings,
        challenge_questions=[
            "Which upstream controls prevent target-derived fields from entering production scoring?",
            "How are missing values and outliers treated consistently between training and production?",
        ],
        recommendations=[item.recommendation for item in findings],
        limitations=[],
    )


def methodology_fallback(context: dict) -> MethodologyReviewOutput:
    available = context.get("available_inputs", {})
    findings = []
    if not available.get("training_code"):
        findings.append(
            finding(
                "Training code not supplied",
                RiskRating.MEDIUM,
                "Methodology",
                "The validation package does not include training code.",
                "Provide reproducible training code with feature engineering, split logic, hyperparameters, and model artifact creation.",
            )
        )
    if not available.get("model"):
        findings.append(
            finding(
                "Independent validation used a baseline model",
                RiskRating.LOW,
                "Methodology",
                "No trained model artifact was supplied, so deterministic analysis trained a baseline model.",
                "Provide the candidate model artifact for direct validation before production approval.",
            )
        )
    return MethodologyReviewOutput(
        summary="Methodology review considered available model artifact, training code, baseline, and reproducibility evidence.",
        risk_rating=max_risk([item.severity for item in findings]),
        findings=findings,
        challenge_questions=[
            "Why was this algorithm selected over simpler challenger models?",
            "How was the train/test split designed to avoid temporal or entity leakage?",
        ],
        recommendations=[item.recommendation for item in findings]
        or ["Retain reproducible training artifacts and challenger-model evidence."],
        limitations=[] if available.get("training_code") else ["Training-code review is limited by missing evidence."],
    )


def performance_fallback(context: dict) -> PerformanceReviewOutput:
    perf = context.get("performance", {})
    quality = context.get("metrics_quality", {})
    findings = []
    if quality.get("insufficient_metrics"):
        findings.append(
            finding(
                "Metrics evidence is insufficient",
                RiskRating.HIGH,
                "Performance",
                quality.get("reason", "Required performance metrics are missing."),
                "Provide required metrics and keep deterministic metric calculation in the validation package.",
            )
        )
    if perf.get("severe_overfitting"):
        findings.append(
            finding(
                "Severe overfitting indicated",
                RiskRating.HIGH,
                "Performance",
                perf.get("overfitting_evidence", "Train/test performance gap exceeds policy threshold."),
                "Rework model complexity, validation design, or regularization before approval.",
            )
        )
    test_metrics = perf.get("test_metrics", {})
    problem_type = perf.get("problem_type")
    if problem_type == "classification":
        if test_metrics.get("f1_weighted", 1.0) < 0.60:
            findings.append(
                finding(
                    "Weak classification performance",
                    RiskRating.HIGH,
                    "Performance",
                    f"Weighted F1 is {test_metrics.get('f1_weighted')}.",
                    "Improve model performance or justify business use under constrained conditions.",
                )
            )
    elif problem_type == "regression":
        if test_metrics.get("r2", 1.0) < 0.30:
            findings.append(
                finding(
                    "Weak regression fit",
                    RiskRating.HIGH,
                    "Performance",
                    f"R2 is {test_metrics.get('r2')}.",
                    "Improve predictive fit or reject the model for this use case.",
                )
            )
    if not findings:
        findings.append(
            finding(
                "Performance evidence is acceptable for baseline validation",
                RiskRating.LOW,
                "Performance",
                f"Deterministic test metrics were calculated for {problem_type}.",
                "Compare these results with business thresholds and production challenger models.",
            )
        )
    return PerformanceReviewOutput(
        summary="Performance review interprets Python-calculated metrics only.",
        risk_rating=max_risk([item.severity for item in findings]),
        findings=findings,
        challenge_questions=[
            "What minimum performance threshold is required for the business use case?",
            "How stable are these metrics across time, segments, and out-of-sample cohorts?",
        ],
        recommendations=[item.recommendation for item in findings],
        limitations=[],
    )


def explainability_fallback(context: dict) -> ExplainabilityReviewOutput:
    explainability = context.get("explainability", {})
    feature_importance = explainability.get("feature_importance", {})
    shap = explainability.get("shap", {})
    findings = []
    if not feature_importance.get("available") and not shap.get("available"):
        findings.append(
            finding(
                "Explainability evidence is missing",
                RiskRating.MEDIUM,
                "Explainability",
                "Neither feature importance nor SHAP output was generated.",
                "Provide interpretable model drivers or use compatible explainability tooling.",
            )
        )
    elif not shap.get("available"):
        findings.append(
            finding(
                "SHAP analysis was not available",
                RiskRating.MEDIUM,
                "Explainability",
                shap.get("reason", "SHAP summary plot was not generated."),
                "Review feature-importance output and add SHAP or equivalent local explanations for high-impact usage.",
            )
        )
    else:
        findings.append(
            finding(
                "Explainability artifacts generated",
                RiskRating.LOW,
                "Explainability",
                "Feature importance and/or SHAP plots were generated by Python tools.",
                "Review top drivers for business plausibility and prohibited proxies.",
            )
        )
    return ExplainabilityReviewOutput(
        summary="Explainability review is based on generated Python artifacts.",
        risk_rating=max_risk([item.severity for item in findings]),
        findings=findings,
        challenge_questions=[
            "Do the top model drivers align with business expectations?",
            "Could any high-importance feature act as a proxy for a protected class?",
        ],
        recommendations=[item.recommendation for item in findings],
        limitations=[] if feature_importance.get("available") else ["Model explainability is limited by missing importances."],
    )


def fairness_fallback(context: dict) -> FairnessReviewOutput:
    fairness = context.get("fairness", {})
    findings = []
    if not fairness.get("available"):
        findings.append(
            finding(
                "Fairness cannot be fully assessed",
                RiskRating.MEDIUM,
                "Fairness",
                fairness.get("reason", "Sensitive attributes were unavailable."),
                "Collect approved fairness attributes or proxy-review evidence and perform group performance testing.",
            )
        )
    elif fairness.get("material_disparity_detected"):
        findings.append(
            finding(
                "Potential group performance disparity detected",
                RiskRating.MEDIUM,
                "Fairness",
                f"Disparities: {fairness.get('disparities')}.",
                "Investigate group-level errors, thresholds, and mitigation controls before approval.",
            )
        )
    else:
        findings.append(
            finding(
                "No material group disparity detected in available checks",
                RiskRating.LOW,
                "Fairness",
                f"Checked columns: {fairness.get('sensitive_columns_checked')}.",
                "Continue fairness monitoring and expand testing where legally and operationally appropriate.",
            )
        )
    return FairnessReviewOutput(
        summary="Fairness review used available group attributes and Python-calculated group metrics.",
        risk_rating=max_risk([item.severity for item in findings]),
        findings=findings,
        challenge_questions=[
            "Which protected-class or proxy attributes are approved for fairness testing?",
            "What disparity thresholds trigger remediation or governance escalation?",
        ],
        recommendations=[item.recommendation for item in findings],
        limitations=[] if fairness.get("available") else ["Full fairness assessment is not possible without group attributes."],
    )


def governance_fallback(context: dict) -> GovernanceReviewOutput:
    governance = context.get("governance_review", {})
    findings = []
    if not governance.get("monitoring_plan_present"):
        findings.append(
            finding(
                "Monitoring plan is missing or weak",
                RiskRating.MEDIUM,
                "Governance",
                "Monitoring, drift, or retraining triggers were not identified in the supplied evidence.",
                "Define monitoring metrics, thresholds, ownership, escalation, and retraining triggers.",
            )
        )
    if not governance.get("approval_evidence_present"):
        findings.append(
            finding(
                "Approval evidence is missing",
                RiskRating.MEDIUM,
                "Governance",
                "Formal approval evidence was not identified.",
                "Document model owner, version, approvers, approval date, and production-use conditions.",
            )
        )
    return GovernanceReviewOutput(
        summary="Governance review considered owner, versioning, monitoring, retraining, lineage, and oversight evidence.",
        risk_rating=max_risk([item.severity for item in findings]),
        findings=findings,
        challenge_questions=[
            "Who owns model performance and issue remediation after deployment?",
            "What thresholds trigger model retirement, recalibration, or retraining?",
        ],
        recommendations=[item.recommendation for item in findings]
        or ["Maintain governance evidence with owner, version, approvals, and monitoring thresholds."],
        limitations=[],
    )


def report_writer_fallback(context: dict) -> ReportWriterOutput:
    assessments: list[AgentAssessment] = context.get("agent_assessments", [])
    policy = context.get("policy_result", {})
    all_findings = [finding for assessment in assessments for finding in assessment.findings]
    recommendations = []
    challenge_questions = []
    for assessment in assessments:
        recommendations.extend(assessment.recommendations)
        challenge_questions.extend(assessment.challenge_questions)

    risk = RiskRating(policy.get("risk_rating", RiskRating.MEDIUM.value))
    decision = FinalDecision(policy.get("final_decision", FinalDecision.INSUFFICIENT_EVIDENCE.value))
    return ReportWriterOutput(
        executive_summary=policy.get(
            "summary",
            "The validation combined deterministic analysis with agent interpretation to produce a model-risk assessment.",
        ),
        model_overview=f"Target column: {context.get('target_column')}. Problem type: {context.get('problem_type')}.",
        files_reviewed=", ".join(context.get("files_reviewed", [])) or "No uploaded files; sample dataset used.",
        validation_scope="Scope included data quality, methodology, performance, explainability, fairness, governance, and final decisioning.",
        data_quality_review=_summary_for(assessments, "DataQualityAgent"),
        methodology_review=_summary_for(assessments, "MethodologyAgent"),
        performance_review=_summary_for(assessments, "PerformanceAgent"),
        explainability_review=_summary_for(assessments, "ExplainabilityAgent"),
        bias_and_fairness_review=_summary_for(assessments, "FairnessAgent"),
        governance_review=_summary_for(assessments, "GovernanceAgent"),
        key_findings=all_findings,
        challenge_questions=list(dict.fromkeys(challenge_questions)),
        risk_rating=risk,
        recommendations=list(dict.fromkeys(recommendations)),
        final_validation_decision=decision,
    )


def _summary_for(assessments: list[AgentAssessment], agent_name: str) -> str:
    for assessment in assessments:
        if assessment.agent_name == agent_name:
            return assessment.summary
    return "Not assessed."

