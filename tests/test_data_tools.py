import pandas as pd

from app.tools.data_tools import (
    detect_class_imbalance,
    detect_duplicates,
    detect_missing_values,
    detect_target_leakage,
    generate_sample_credit_data,
    profile_dataset,
)


def test_generate_sample_credit_data_has_expected_target():
    df = generate_sample_credit_data(rows=50)
    assert "defaulted" in df.columns
    assert len(df) == 50
    assert set(df["defaulted"].unique()).issubset({0, 1})


def test_profile_dataset_reports_missing_and_duplicates():
    df = pd.DataFrame({"x": [1, 1, None], "target": [0, 0, 1]})
    profiled = profile_dataset(df, "target")
    assert profiled["missing_values"]["columns_with_missing"] == 1
    assert profiled["duplicates"]["duplicate_rows"] == 1


def test_detect_target_leakage_flags_duplicate_target_feature():
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target_copy": [0, 1, 0, 1], "target": [0, 1, 0, 1]})
    leakage = detect_target_leakage(df, "target")
    assert leakage["suspected_leakage"] is True
    assert any(item["column"] == "target_copy" for item in leakage["suspicious_features"])


def test_detect_class_imbalance_flags_minority_rate():
    df = pd.DataFrame({"target": [0] * 95 + [1] * 5})
    imbalance = detect_class_imbalance(df, "target")
    assert imbalance["is_imbalanced"] is True

