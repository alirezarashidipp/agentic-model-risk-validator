from app.services.risk_policy import assess_metrics_quality, detect_overfitting, determine_policy_result
from app.tools.data_tools import generate_sample_credit_data
from app.tools.model_tools import calculate_classification_metrics, train_baseline_model


def test_train_baseline_model_returns_classification_metrics():
    df = generate_sample_credit_data(rows=120)
    result = train_baseline_model(df, "defaulted", "classification")
    metrics = result["test_metrics"]
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert "confusion_matrix" in metrics


def test_calculate_classification_metrics_uses_python_model():
    df = generate_sample_credit_data(rows=120)
    result = train_baseline_model(df, "defaulted", "classification")
    metrics = calculate_classification_metrics(result["model"], result["X_test"], result["y_test"])
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_risk_policy_for_target_leakage_is_high():
    context = {
        "data_profile": {"target_leakage": {"suspected_leakage": True}},
        "performance": {"severe_overfitting": False},
        "metrics_quality": {"insufficient_metrics": False},
        "explainability": {"feature_importance": {"available": True}, "shap": {"available": False}},
        "governance_review": {"monitoring_plan_present": True},
        "available_inputs": {"documentation": True, "model": True},
        "sample_dataset_used": False,
    }
    policy = determine_policy_result(context)
    assert policy["risk_rating"] == "HIGH"
    assert policy["final_decision"] == "rejected"


def test_metrics_quality_requires_core_metrics():
    quality = assess_metrics_quality(
        problem_type="classification",
        test_metrics={"accuracy": 0.8},
        uploaded_metrics_text=None,
    )
    assert quality["insufficient_metrics"] is True


def test_overfitting_detection_flags_large_train_test_gap():
    result = detect_overfitting(
        {"f1_weighted": 0.98},
        {"f1_weighted": 0.60},
        "classification",
    )
    assert result["severe_overfitting"] is True

