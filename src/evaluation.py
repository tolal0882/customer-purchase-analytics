"""
Model Evaluation Module (Section 6):
  - Precision, Recall, F1-score, ROC-AUC, PR-AUC, Accuracy
  - Confusion Matrix & Per-class Classification Report
  - ROC Curve & Precision-Recall Curve coordinates
  - Feature Importance extraction from fitted Scikit-learn Pipelines
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Compute held-out test metrics on binary Revenue target (0 = No Purchase, 1 = Purchase)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    y_pr = np.asarray(y_proba, dtype=float)
    return {
        "accuracy": round(float(accuracy_score(y_t, y_p)), 4),
        "precision": round(float(precision_score(y_t, y_p, zero_division=0)), 4),
        "recall": round(float(recall_score(y_t, y_p, zero_division=0)), 4),
        "f1": round(float(f1_score(y_t, y_p, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_t, y_pr)), 4),
        "pr_auc": round(float(average_precision_score(y_t, y_pr)), 4),
    }


def compute_confusion_and_report(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, list[dict]]:
    """Return Confusion Matrix dict and structured Classification Report rows."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    cm_dict = {
        "labels": ["No Purchase (0)", "Purchase (1)"],
        "matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    rep = classification_report(
        y_t,
        y_p,
        labels=[0, 1],
        target_names=["No Purchase (False)", "Purchase (True)"],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for key in ["No Purchase (False)", "Purchase (True)", "macro avg", "weighted avg"]:
        if key in rep and isinstance(rep[key], dict):
            rows.append({
                "Class / Average": key,
                "Precision": round(float(rep[key]["precision"]), 4),
                "Recall": round(float(rep[key]["recall"]), 4),
                "F1-Score": round(float(rep[key]["f1-score"]), 4),
                "Support": int(rep[key]["support"]),
            })
    return cm_dict, rows


def compute_curves(y_true: np.ndarray, y_proba: np.ndarray, max_points: int = 120) -> dict:
    """Compute downsampled ROC and Precision-Recall curve points for Plotly visualization."""
    y_t = np.asarray(y_true, dtype=int)
    y_pr = np.asarray(y_proba, dtype=float)

    fpr, tpr, _ = roc_curve(y_t, y_pr)
    if len(fpr) > max_points:
        idx = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]

    prec, rec, _ = precision_recall_curve(y_t, y_pr)
    if len(prec) > max_points:
        idx = np.linspace(0, len(prec) - 1, max_points).astype(int)
        prec, rec = prec[idx], rec[idx]

    return {
        "roc": {"fpr": [round(float(x), 4) for x in fpr], "tpr": [round(float(y), 4) for y in tpr]},
        "pr": {"recall": [round(float(x), 4) for x in rec], "precision": [round(float(y), 4) for y in prec]},
    }


def extract_feature_importance(fitted_pipeline, top_n: int = 15) -> list[dict]:
    """Extract feature importance or standardized absolute coefficients from a fitted pipeline."""
    try:
        pre = fitted_pipeline.named_steps["preprocessor"]
        clf = fitted_pipeline.named_steps["classifier"]
        raw_names = list(pre.get_feature_names_out())
        clean_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
        if hasattr(clf, "feature_importances_"):
            vals = np.asarray(clf.feature_importances_, dtype=float)
        elif hasattr(clf, "coef_"):
            vals = np.abs(np.asarray(clf.coef_, dtype=float)).ravel()
        else:
            return []
        s = pd.Series(vals, index=clean_names).sort_values(ascending=False).head(top_n)
        return [{"feature": k, "importance": round(float(v), 4)} for k, v in s.items()]
    except Exception:
        return []
