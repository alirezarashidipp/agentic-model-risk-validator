from __future__ import annotations

from app.schemas.agent_outputs import FinalDecision, RiskRating


RISK_ORDER = {RiskRating.LOW: 0, RiskRating.MEDIUM: 1, RiskRating.HIGH: 2}


def higher_risk(current: RiskRating, candidate: RiskRating) -> RiskRating:
    return candidate if RISK_ORDER[candidate] > RISK_ORDER[current] else current


def detect_overfitting(train_metrics: dict, test_metrics: dict, problem_type: str) -> dict:
    if problem_type == "classification":
        train_value = train_metrics.get("f1_weighted")
        test_value = test_metrics.get("f1_weighted")
        metric = "weighted F1"
    else:
        train_value = train_metrics.get("r2")
        test_value = test_metrics.get("r2")
        metric = "R2"
    if train_value is None or test_value is None:
        return {"severe_overfitting": False, "overfitting_evidence": "Train/test comparison unavailable."}

    gap = float(train_value) - float(test_value)
    severe = bool(gap >= 0.25 and float(train_value) >= 0.75)
    return {
        "severe_overfitting": severe,
        "overfitting_gap": gap,
        "overfitting_evidence": f"Train {metric}={train_value}; test {metric}={test_value}; gap={gap:.3f}.",
    }


def assess_metrics_quality(
    *,
    problem_type: str,
    test_metrics: dict,
    uploaded_metrics_text: str | None,
) -> dict:
    expected = (
        ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]
        if problem_type == "classification"
        else ["mae", "rmse", "r2", "residual"]
    )

    computed_missing = [
        name
        for name in expected
        if not any(name in key.lower() for key in test_metrics.keys())
    ]
    if computed_missing:
        return {
            "insufficient_metrics": True,
            "reason": f"Deterministic metrics are missing expected values: {computed_missing}.",
            "expected_metrics": expected,
        }

    if uploaded_metrics_text:
        lower = uploaded_metrics_text.lower()
        uploaded_missing = [name for name in expected if name not in lower]
        if len(uploaded_missing) >= max(2, len(expected) // 2):
            return {
                "insufficient_metrics": True,
                "reason": f"Uploaded metrics file omits expected metrics: {uploaded_missing}.",
                "expected_metrics": expected,
            }

    return {
        "insufficient_metrics": False,
        "reason": "Required deterministic metrics were calculated by Python tools.",
        "expected_metrics": expected,
    }


def determine_policy_result(context: dict) -> dict:
    risk = RiskRating.LOW
    reasons: list[str] = []
    insufficient_evidence: list[str] = []

    leakage = context.get("data_profile", {}).get("target_leakage", {})
    if leakage.get("suspected_leakage"):
        risk = higher_risk(risk, RiskRating.HIGH)
        reasons.append("Target leakage rule triggered.")

    performance = context.get("performance", {})
    if performance.get("severe_overfitting"):
        risk = higher_risk(risk, RiskRating.HIGH)
        reasons.append("Severe overfitting rule triggered.")

    metrics_quality = context.get("metrics_quality", {})
    if metrics_quality.get("insufficient_metrics"):
        risk = higher_risk(risk, RiskRating.HIGH)
        reasons.append("Wrong or insufficient metrics rule triggered.")

    explainability = context.get("explainability", {})
    if not explainability.get("feature_importance", {}).get("available") and not explainability.get("shap", {}).get("available"):
        risk = higher_risk(risk, RiskRating.MEDIUM)
        reasons.append("Missing explainability rule triggered.")

    governance = context.get("governance_review", {})
    if not governance.get("monitoring_plan_present"):
        risk = higher_risk(risk, RiskRating.MEDIUM)
        reasons.append("Missing monitoring plan rule triggered.")

    available = context.get("available_inputs", {})
    if not available.get("documentation"):
        risk = higher_risk(risk, RiskRating.MEDIUM)
        reasons.append("Missing documentation rule triggered.")
    if not available.get("model"):
        insufficient_evidence.append("candidate model artifact")
    if not available.get("documentation"):
        insufficient_evidence.append("model documentation")
    if context.get("sample_dataset_used"):
        insufficient_evidence.append("user-supplied dataset")

    if insufficient_evidence:
        decision = FinalDecision.INSUFFICIENT_EVIDENCE
    elif risk == RiskRating.HIGH:
        decision = FinalDecision.REJECTED
    elif risk == RiskRating.MEDIUM:
        decision = FinalDecision.APPROVED_WITH_CONDITIONS
    else:
        decision = FinalDecision.APPROVED

    summary = (
        f"Overall risk is {risk.value}. Final decision is {decision.value}. "
        f"Primary reasons: {', '.join(reasons) if reasons else 'no high-severity deterministic rule triggered'}."
    )
    if insufficient_evidence:
        summary += f" Insufficient evidence items: {', '.join(insufficient_evidence)}."

    return {
        "risk_rating": risk.value,
        "final_decision": decision.value,
        "reasons": reasons,
        "insufficient_evidence": insufficient_evidence,
        "summary": summary,
    }

