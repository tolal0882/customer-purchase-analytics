"""
Step 4 – Evaluation: metrics and figures for the final model on the TEST set.
"""
import matplotlib

matplotlib.use("Agg")  # save figures without opening a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)

from ml import config


# =============================================================================
# 1. HELD-OUT TEST METRICS & CURVES
# =============================================================================
def test_metrics(pipe, X_test, y_test) -> dict:
    """Compute held-out test set metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)."""
    y_pred = pipe.predict(X_test)
    if config.TASK_TYPE == "classification":
        proba = pipe.predict_proba(X_test)[:, 1]
        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        }
    return {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_test, y_pred)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }


def curves_data(pipe, X_test, y_test, max_points: int = 150) -> dict:
    """Return downsampled ROC and Precision-Recall curve points for interactive charts."""
    if config.TASK_TYPE != "classification":
        return {}
    proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)

    def _sub(a):
        if len(a) <= max_points:
            return [round(float(x), 4) for x in a]
        idx = np.unique(np.linspace(0, len(a) - 1, max_points).astype(int))
        return [round(float(a[i]), 4) for i in idx]

    return {
        "roc": {"fpr": _sub(fpr), "tpr": _sub(tpr)},
        "pr": {"recall": _sub(rec), "precision": _sub(prec)},
    }


def confusion(pipe, X_test, y_test) -> dict | None:
    if config.TASK_TYPE != "classification":
        return None
    cm = confusion_matrix(y_test, pipe.predict(X_test), labels=[0, 1])
    labels = [config.NEGATIVE_CLASS, config.POSITIVE_CLASS]
    return {"labels": labels, "matrix": cm.tolist()}


def report_text(pipe, X_test, y_test) -> str:
    if config.TASK_TYPE != "classification":
        return ""
    return classification_report(
        y_test,
        pipe.predict(X_test),
        target_names=[config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
        zero_division=0,
    )


# =============================================================================
# 2. FEATURE IMPORTANCE EXTRACTION
# =============================================================================
def feature_importance(pipe, top: int = 20) -> dict:
    """Importance for tree models, |coefficient| for linear models."""
    model = pipe.named_steps["model"]
    names = pipe.named_steps["preprocess"].named_steps["columns"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        return {}
    s = pd.Series(values, index=names).sort_values(ascending=False).head(top)
    return {k: round(float(v), 4) for k, v in s.items()}


# =============================================================================
# 3. SAVE REPORT FIGURES (reports/figures/*.png)
# =============================================================================
def save_figures(pipe, X_test, y_test, cv_table: pd.DataFrame, importance: dict) -> list[str]:
    """Save charts to reports/figures/ for the final report and presentation slides."""
    out = config.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Model comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    col = f"cv_{config.PRIMARY_METRIC}_mean"
    cv_table.sort_values(col).plot.barh(
        x="model", y=col, xerr=f"cv_{config.PRIMARY_METRIC}_std",
        ax=ax, legend=False, color="#2F855A",
    )
    ax.set_title(f"Model comparison ({config.CV_FOLDS}-fold CV, {config.PRIMARY_METRIC})")
    ax.set_xlabel(config.PRIMARY_METRIC)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)
    saved.append("model_comparison.png")

    # 2. Feature importance
    if importance:
        fig, ax = plt.subplots(figsize=(7, 5))
        pd.Series(importance).sort_values().plot.barh(ax=ax, color="#2F855A")
        ax.set_title("Feature importance (final model)")
        fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=150)
        plt.close(fig)
        saved.append("feature_importance.png")

    if config.TASK_TYPE == "classification":
        # 3. Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_estimator(
            pipe, X_test, y_test,
            display_labels=[config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
            cmap="Greens", ax=ax,
        )
        ax.set_title("Confusion matrix (test set)")
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        saved.append("confusion_matrix.png")

        # 4. ROC curve
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_title("ROC curve (test set)")
        fig.tight_layout()
        fig.savefig(out / "roc_curve.png", dpi=150)
        plt.close(fig)
        saved.append("roc_curve.png")

    return saved
