from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

from app.tools.json_utils import to_jsonable


DEFAULT_SENSITIVE_COLUMNS = [
    "gender",
    "sex",
    "race",
    "ethnicity",
    "age_group",
    "region",
]


def find_sensitive_columns(df: pd.DataFrame) -> list[str]:
    columns = {column.lower(): column for column in df.columns}
    return [columns[name] for name in DEFAULT_SENSITIVE_COLUMNS if name in columns]


def analyze_fairness(
    df: pd.DataFrame,
    target_column: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
) -> dict:
    sensitive_columns = [column for column in find_sensitive_columns(X_test) if column != target_column]
    if not sensitive_columns:
        return {
            "available": False,
            "reason": "No common sensitive or proxy attributes were detected in the evaluation split.",
            "sensitive_columns_checked": [],
        }

    predictions = pd.Series(model.predict(X_test), index=X_test.index)
    group_results: dict[str, dict] = {}
    for column in sensitive_columns:
        group_results[column] = {}
        for group_value, group_frame in X_test.groupby(column, dropna=False):
            idx = group_frame.index
            if len(idx) < 5:
                continue
            if problem_type == "classification":
                accuracy = accuracy_score(y_test.loc[idx], predictions.loc[idx])
                positive_rate = float(pd.Series(predictions.loc[idx]).astype(str).isin(["1", "True", "true", "yes"]).mean())
                group_results[column][str(group_value)] = {
                    "count": int(len(idx)),
                    "accuracy": float(accuracy),
                    "predicted_positive_rate": positive_rate,
                }
            else:
                mae = mean_absolute_error(y_test.loc[idx], predictions.loc[idx])
                group_results[column][str(group_value)] = {
                    "count": int(len(idx)),
                    "mae": float(mae),
                }

    disparities = []
    for column, groups in group_results.items():
        metric_name = "accuracy" if problem_type == "classification" else "mae"
        values = [metrics[metric_name] for metrics in groups.values() if metric_name in metrics]
        if len(values) > 1:
            disparities.append(
                {
                    "column": column,
                    "metric": metric_name,
                    "max_gap": float(max(values) - min(values)),
                }
            )

    return to_jsonable(
        {
            "available": True,
            "sensitive_columns_checked": sensitive_columns,
            "group_metrics": group_results,
            "disparities": disparities,
            "material_disparity_detected": any(item["max_gap"] >= 0.10 for item in disparities),
        }
    )

