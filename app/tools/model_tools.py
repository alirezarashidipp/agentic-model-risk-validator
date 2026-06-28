from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.tools.data_tools import infer_problem_type
from app.tools.json_utils import to_jsonable


def load_model(file_path: str | Path) -> Any:
    path = Path(file_path)
    if path.suffix.lower() == ".joblib":
        return joblib.load(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _drop_identifier_columns(X: pd.DataFrame) -> pd.DataFrame:
    keep_columns = []
    for column in X.columns:
        unique_rate = X[column].nunique(dropna=True) / max(len(X), 1)
        column_lower = column.lower()
        if column_lower.endswith("id") or column_lower.endswith("_id") or unique_rate >= 0.98:
            continue
        keep_columns.append(column)
    return X[keep_columns].copy()


def calculate_classification_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    labels = sorted(pd.Series(y_test).dropna().unique().tolist())
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "labels": [str(label) for label in labels],
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
    }

    if hasattr(model, "predict_proba") and len(labels) > 1:
        try:
            probabilities = model.predict_proba(X_test)
            if len(labels) == 2:
                metrics["roc_auc"] = roc_auc_score(y_test, probabilities[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                )
        except Exception as exc:
            metrics["roc_auc_error"] = str(exc)
    return to_jsonable(metrics)


def calculate_regression_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    residuals = pd.Series(y_test.to_numpy() - y_pred)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
        "residual_summary": {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "p05": float(residuals.quantile(0.05)),
            "p50": float(residuals.quantile(0.50)),
            "p95": float(residuals.quantile(0.95)),
        },
    }
    return to_jsonable(metrics)


def train_baseline_model(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str = "auto",
) -> dict:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset")

    clean_df = df.dropna(subset=[target_column]).copy()
    if clean_df.empty:
        raise ValueError("No rows remain after dropping missing target values")
    if problem_type == "auto":
        problem_type = infer_problem_type(clean_df, target_column)

    X = _drop_identifier_columns(clean_df.drop(columns=[target_column]))
    y = clean_df[target_column]
    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    if stratify is not None and y.value_counts().min() < 2:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(X_train)
    if problem_type == "classification":
        estimator = RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=3,
        )
    elif problem_type == "regression":
        estimator = RandomForestRegressor(
            n_estimators=120,
            random_state=42,
            min_samples_leaf=3,
        )
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
    model.fit(X_train, y_train)

    if problem_type == "classification":
        train_metrics = calculate_classification_metrics(model, X_train, y_train)
        test_metrics = calculate_classification_metrics(model, X_test, y_test)
    else:
        train_metrics = calculate_regression_metrics(model, X_train, y_train)
        test_metrics = calculate_regression_metrics(model, X_test, y_test)

    return {
        "problem_type": problem_type,
        "target_column": target_column,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
) -> dict:
    if problem_type == "classification":
        return calculate_classification_metrics(model, X_test, y_test)
    return calculate_regression_metrics(model, X_test, y_test)
