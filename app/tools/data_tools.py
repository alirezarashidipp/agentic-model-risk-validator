from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from app.tools.json_utils import to_jsonable


ProblemTypeLiteral = Literal["classification", "regression"]


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a tabular dataset from a supported local file."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def generate_sample_credit_data(
    output_path: str | Path | None = None,
    *,
    rows: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic sample credit risk classification dataset."""

    rng = np.random.default_rng(random_state)
    age = rng.integers(21, 72, rows)
    income = rng.normal(78000, 26000, rows).clip(18000, 220000)
    debt_to_income = rng.beta(2.2, 5.0, rows).clip(0.01, 0.95)
    credit_score = rng.normal(675, 82, rows).clip(350, 850)
    employment_years = rng.gamma(3.0, 2.2, rows).clip(0, 35)
    missed_payments = rng.poisson(0.7, rows).clip(0, 8)
    loan_amount = rng.normal(23000, 9000, rows).clip(2000, 65000)
    gender = rng.choice(["female", "male", "non_binary"], rows, p=[0.48, 0.49, 0.03])
    region = rng.choice(["north", "south", "east", "west"], rows)

    logit = (
        -3.2
        + 3.6 * debt_to_income
        + 0.36 * missed_payments
        + 0.000018 * loan_amount
        - 0.0065 * (credit_score - 650)
        - 0.000008 * (income - 70000)
        - 0.022 * employment_years
        + np.where(region == "south", 0.2, 0)
    )
    probability = 1 / (1 + np.exp(-logit))
    defaulted = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{idx:05d}" for idx in range(rows)],
            "age": age,
            "income": income.round(2),
            "debt_to_income": debt_to_income.round(4),
            "credit_score": credit_score.round(1),
            "employment_years": employment_years.round(1),
            "missed_payments": missed_payments,
            "loan_amount": loan_amount.round(2),
            "gender": gender,
            "region": region,
            "defaulted": defaulted,
        }
    )

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def infer_problem_type(df: pd.DataFrame, target_column: str) -> ProblemTypeLiteral:
    target = df[target_column].dropna()
    if pd.api.types.is_numeric_dtype(target) and target.nunique() > 15:
        return "regression"
    return "classification"


def detect_missing_values(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    total_rows = max(len(df), 1)
    details = {
        column: {
            "missing_count": int(count),
            "missing_rate": float(count / total_rows),
        }
        for column, count in missing.items()
        if int(count) > 0
    }
    return {
        "total_missing_cells": int(missing.sum()),
        "columns_with_missing": len(details),
        "details": details,
    }


def detect_duplicates(df: pd.DataFrame) -> dict:
    duplicate_count = int(df.duplicated().sum())
    return {
        "duplicate_rows": duplicate_count,
        "duplicate_rate": float(duplicate_count / max(len(df), 1)),
    }


def detect_outliers(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number])
    details: dict[str, dict[str, float | int]] = {}
    total_outlier_cells = 0
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty or series.nunique() <= 2:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        if count > 0:
            details[column] = {
                "outlier_count": count,
                "outlier_rate": float(count / max(len(series), 1)),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
            total_outlier_cells += count
    return {
        "total_outlier_cells": total_outlier_cells,
        "columns_with_outliers": len(details),
        "details": details,
    }


def detect_class_imbalance(df: pd.DataFrame, target_column: str) -> dict:
    if target_column not in df.columns:
        return {"available": False, "reason": "target column is missing"}
    target = df[target_column].dropna()
    distribution = target.value_counts(normalize=True).sort_index()
    counts = target.value_counts().sort_index()
    minority_rate = float(distribution.min()) if not distribution.empty else None
    return {
        "available": True,
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "class_distribution": {str(key): float(value) for key, value in distribution.items()},
        "minority_class_rate": minority_rate,
        "is_imbalanced": bool(minority_rate is not None and minority_rate < 0.2),
    }


def _numeric_target(target: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(target):
        return target.astype(float)
    return pd.Series(pd.factorize(target.astype(str))[0], index=target.index, dtype=float)


def detect_target_leakage(df: pd.DataFrame, target_column: str) -> dict:
    if target_column not in df.columns:
        return {"available": False, "reason": "target column is missing", "suspected_leakage": False}

    target = df[target_column]
    target_numeric = _numeric_target(target)
    suspicious: list[dict] = []
    target_tokens = {token for token in target_column.lower().replace("-", "_").split("_") if token}
    leakage_words = {"target", "label", "outcome", "default", "approved", "approval", "fraud"}

    for column in df.columns:
        if column == target_column:
            continue
        feature = df[column]
        column_lower = column.lower()
        if feature.equals(target):
            suspicious.append(
                {
                    "column": column,
                    "reason": "feature values exactly match the target",
                    "score": 1.0,
                }
            )
            continue
        if target_tokens.intersection(column_lower.replace("-", "_").split("_")) or any(
            word in column_lower for word in leakage_words
        ):
            suspicious.append(
                {
                    "column": column,
                    "reason": "feature name suggests post-outcome or target-derived information",
                    "score": 0.8,
                }
            )
        try:
            feature_numeric = _numeric_target(feature)
            valid = feature_numeric.notna() & target_numeric.notna()
            if valid.sum() > 3 and feature_numeric[valid].nunique() > 1:
                correlation = abs(float(np.corrcoef(feature_numeric[valid], target_numeric[valid])[0, 1]))
                if np.isfinite(correlation) and correlation >= 0.95:
                    suspicious.append(
                        {
                            "column": column,
                            "reason": "near-perfect correlation with target",
                            "score": correlation,
                        }
                    )
        except Exception:
            continue

    deduped = {item["column"]: item for item in suspicious}
    return {
        "available": True,
        "suspected_leakage": bool(deduped),
        "suspicious_features": list(deduped.values()),
    }


def profile_dataset(df: pd.DataFrame, target_column: str | None = None) -> dict:
    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(exclude=[np.number])
    target_summary = {}
    if target_column and target_column in df.columns:
        target_summary = {
            "target_column": target_column,
            "missing_target_count": int(df[target_column].isna().sum()),
            "unique_target_values": int(df[target_column].nunique(dropna=True)),
        }

    profile = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "numeric_columns": list(numeric.columns),
        "categorical_columns": list(categorical.columns),
        "numeric_summary": numeric.describe().round(4).to_dict() if not numeric.empty else {},
        "categorical_summary": {
            column: {
                "unique_values": int(df[column].nunique(dropna=True)),
                "top_values": {
                    str(key): int(value)
                    for key, value in df[column].value_counts(dropna=False).head(5).items()
                },
            }
            for column in categorical.columns
        },
        "target_summary": target_summary,
        "missing_values": detect_missing_values(df),
        "duplicates": detect_duplicates(df),
        "outliers": detect_outliers(df),
    }
    if target_column:
        profile["class_imbalance"] = detect_class_imbalance(df, target_column)
        profile["target_leakage"] = detect_target_leakage(df, target_column)
    return to_jsonable(profile)

