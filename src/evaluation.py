"""Metrics, curves and feature importance. All values are computed from real predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)


def _downsample(*arrs, n=400):
    m = len(arrs[0])
    if m <= n:
        return [a.tolist() for a in arrs]
    idx = np.unique(np.linspace(0, m - 1, n).astype(int))
    return [a[idx].tolist() for a in arrs]


# =============================================================================
# 1. IMBALANCED CLASSIFICATION METRICS & ROC/PR CURVES
#    - Computes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion
#      Matrix, and full Classification Report from real held-out predictions.
#    - Downsamples curve points to <=400 vertices so JSON metadata stays compact.
# =============================================================================
def compute_metrics(y_true, proba, threshold: float = 0.5, with_curves: bool = True) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    single = len(np.unique(y_true)) < 2
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    out = {
        "n": int(len(y_true)), "positives": int(y_true.sum()), "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float("nan") if single else float(roc_auc_score(y_true, proba)),
        "pr_auc": float("nan") if single else float(average_precision_score(y_true, proba)),
        "majority_baseline_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
        "predicted_positive": int(pred.sum()),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, pred, labels=[0, 1],
                                                       target_names=["No purchase", "Purchase"],
                                                       output_dict=True, zero_division=0),
    }
    if with_curves and not single:
        fpr, tpr, _ = roc_curve(y_true, proba)
        pr, rc, th = precision_recall_curve(y_true, proba)
        out["roc_curve"] = dict(zip(("fpr", "tpr"), _downsample(fpr, tpr)))
        out["pr_curve"] = dict(zip(("precision", "recall"), _downsample(pr, rc)))
        out["score_distribution"] = {
            "positive": np.histogram(proba[y_true == 1], bins=20, range=(0, 1))[0].tolist(),
            "negative": np.histogram(proba[y_true == 0], bins=20, range=(0, 1))[0].tolist()}
    return out


# =============================================================================
# 2. FEATURE IMPORTANCE (NATIVE + MODEL-AGNOSTIC PERMUTATION IMPORTANCE)
#    - Extracts coefficients (Logistic Regression) or impurity importance (trees).
#    - Also computes permutation importance (PR-AUC drop) on held-out test rows.
# =============================================================================
def feature_importance(model, features: list[str], X_test: pd.DataFrame, y_test, seed: int = 42) -> dict:
    """Native importance (coefficients / impurity) + model-agnostic permutation importance (PR-AUC drop)
    computed on a held-out sample."""
    res: dict = {"native": None, "native_kind": None, "permutation": None}
    final = model.estimator_.named_steps["model"]
    if hasattr(final, "coef_"):
        res["native"] = dict(zip(features, final.coef_.ravel().tolist()))
        res["native_kind"] = "Standardised coefficient (sign = direction)"
    elif hasattr(final, "feature_importances_"):
        res["native"] = dict(zip(features, final.feature_importances_.tolist()))
        res["native_kind"] = "Impurity-based importance"
    try:
        Xs = X_test if len(X_test) <= 8000 else X_test.sample(8000, random_state=seed)
        ys = np.asarray(y_test)[X_test.index.get_indexer(Xs.index)] if len(X_test) > 8000 else np.asarray(y_test)
        if len(np.unique(ys)) == 2:
            pi = permutation_importance(model, Xs, ys, scoring="average_precision", n_repeats=3,
                                        random_state=seed, n_jobs=1)
            res["permutation"] = {"mean": dict(zip(features, pi.importances_mean.tolist())),
                                  "std": dict(zip(features, pi.importances_std.tolist()))}
    except Exception as exc:  # importance is optional; never fail training because of it
        res["permutation_error"] = str(exc)
    return res
