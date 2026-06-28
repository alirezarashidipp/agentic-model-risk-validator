from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from app.tools.json_utils import to_jsonable


def generate_confusion_matrix_plot(metrics: dict, output_path: str | Path) -> dict:
    matrix = metrics.get("confusion_matrix")
    labels = metrics.get("labels")
    if not matrix or not labels:
        return {"available": False, "reason": "confusion matrix metrics are unavailable"}

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=np.array(matrix), display_labels=labels).plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"available": True, "path": str(path)}


def _extract_feature_importances(model: Any) -> tuple[list[str], list[float]] | None:
    estimator = model
    names: list[str] | None = None
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)
        preprocessor = model.named_steps.get("preprocess")
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            names = [str(name).replace("numeric__", "").replace("categorical__", "") for name in preprocessor.get_feature_names_out()]

    if hasattr(estimator, "feature_importances_"):
        values = list(map(float, estimator.feature_importances_))
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        values = list(map(float, np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)))
    else:
        return None

    if names is None:
        names = [f"feature_{idx}" for idx in range(len(values))]
    return names, values


def generate_feature_importance_plot(
    model: Any,
    output_path: str | Path,
    *,
    top_n: int = 20,
) -> dict:
    extracted = _extract_feature_importances(model)
    if extracted is None:
        return {"available": False, "reason": "model does not expose feature importances or coefficients"}

    names, values = extracted
    ranking = (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(ranking))))
    ax.barh(ranking["feature"], ranking["importance"], color="#2f6f73")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "available": True,
        "path": str(path),
        "top_features": to_jsonable(ranking.sort_values("importance", ascending=False).to_dict("records")),
    }


def generate_shap_summary_plot(
    model: Any,
    X_sample: pd.DataFrame,
    output_path: str | Path,
    *,
    max_rows: int = 150,
) -> dict:
    try:
        import shap
    except Exception as exc:
        return {"available": False, "reason": f"SHAP is not installed or importable: {exc}"}

    try:
        sample = X_sample.head(max_rows)
        estimator = model
        transformed = sample
        feature_names = list(sample.columns)
        if hasattr(model, "named_steps"):
            preprocessor = model.named_steps.get("preprocess")
            estimator = model.named_steps.get("model", model)
            if preprocessor is not None:
                transformed = preprocessor.transform(sample)
                if hasattr(preprocessor, "get_feature_names_out"):
                    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]

        explainer = shap.Explainer(estimator, transformed)
        shap_values = explainer(transformed)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        return {"available": True, "path": str(path)}
    except Exception as exc:
        plt.close()
        return {"available": False, "reason": f"SHAP plot generation failed: {exc}"}

