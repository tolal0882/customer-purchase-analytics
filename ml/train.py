"""
Step 3 – Modelling: compare classification models with cross-validation,
tune the best one, evaluate it on the held-out test set, run K-Means
Customer Segmentation, and save all artifacts.

Run from the project root:
    python -m ml.train

Outputs:
    models/model.joblib               trained classification pipeline (used by API)
    models/metrics.json               CV comparison, test metrics, confusion matrix
    models/model_card.md              human-readable summary of the model
    models/clustering.joblib          trained K-Means customer segmentation bundle
    models/clustering_metrics.json    Elbow/Silhouette scores & segment profiles
    reports/figures/*.png             charts for the final report
"""
import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml import clustering, config, evaluate
from ml.data import make_dataset
from ml.features import build_preprocessor, get_input_columns

RS = config.RANDOM_STATE


# =============================================================================
# 1. CANDIDATE CLASSIFICATION MODELS & HYPERPARAMETER GRIDS
# =============================================================================
def candidate_models() -> dict:
    """Models to compare + hyperparameter grids for RandomizedSearchCV tuning."""
    if config.TASK_TYPE == "classification":
        return {
            "Logistic Regression": (
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS),
                {"model__C": [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]},
            ),
            "Decision Tree": (
                DecisionTreeClassifier(random_state=RS, class_weight="balanced"),
                {"model__max_depth": [4, 5, 6, 8, 10], "model__min_samples_leaf": [10, 20, 50]},
            ),
            "Random Forest": (
                RandomForestClassifier(
                    n_estimators=120, max_depth=12, min_samples_leaf=10,
                    random_state=RS, n_jobs=-1, class_weight="balanced",
                ),
                {
                    "model__n_estimators": [100, 150, 200],
                    "model__max_depth": [8, 10, 14],
                    "model__min_samples_leaf": [5, 10, 20],
                },
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RS),
                {
                    "model__n_estimators": [80, 120, 160],
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [2, 3, 4],
                },
            ),
        }
    return {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0]}),
        "Random Forest": (RandomForestRegressor(random_state=RS, n_jobs=-1), {}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=RS), {}),
    }


def scoring() -> list[str]:
    if config.TASK_TYPE == "classification":
        return ["accuracy", "precision", "recall", "f1", "roc_auc"]
    return ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]


# =============================================================================
# 2. DATASET LOADER & X/y PREPARATION
# =============================================================================
def load_dataset() -> pd.DataFrame:
    if not config.PROCESSED_DATA_PATH.exists():
        print("Processed data not found – running make_dataset first.")
        make_dataset.main()
    return pd.read_csv(config.PROCESSED_DATA_PATH)


def prepare_xy(df: pd.DataFrame):
    input_cols = get_input_columns(df)
    X = df[input_cols]
    y = df[config.TARGET_COLUMN]
    if config.TASK_TYPE == "classification":
        y = (y == config.POSITIVE_CLASS).astype(int)  # 1 = Yes (purchase), 0 = No
    return X, y, input_cols


# =============================================================================
# 3. 5-FOLD STRATIFIED CROSS-VALIDATION MODEL COMPARISON & HYPERPARAMETER TUNING
# =============================================================================
def compare_models(X_train, y_train, verbose=True) -> pd.DataFrame:
    """Cross-validate every candidate model on the TRAINING set only."""
    cv = (
        StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=RS)
        if config.TASK_TYPE == "classification"
        else KFold(config.CV_FOLDS, shuffle=True, random_state=RS)
    )
    rows = []
    for name, (model, _) in candidate_models().items():
        pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", model)])
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring(), n_jobs=-1)
        row = {"model": name}
        for m in scoring():
            row[f"cv_{m}_mean"] = round(float(res[f"test_{m}"].mean()), 4)
            row[f"cv_{m}_std"] = round(float(res[f"test_{m}"].std()), 4)
        rows.append(row)
        if verbose:
            print(
                f"  {name:<22} {config.PRIMARY_METRIC} = "
                f"{row[f'cv_{config.PRIMARY_METRIC}_mean']:.4f} ± {row[f'cv_{config.PRIMARY_METRIC}_std']:.4f}"
            )
    return pd.DataFrame(rows).sort_values(f"cv_{config.PRIMARY_METRIC}_mean", ascending=False)


def tune(name, X_train, y_train):
    """Tune the top-performing model on the training set using RandomizedSearchCV."""
    model, params = candidate_models()[name]
    pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", model)])
    if not params:
        return pipe.fit(X_train, y_train), {}, None
    n_combos = 1
    for v in params.values():
        n_combos *= len(v)
    search = RandomizedSearchCV(
        pipe,
        params,
        n_iter=min(config.TUNING_ITERATIONS, n_combos),
        scoring=config.PRIMARY_METRIC,
        cv=config.CV_FOLDS,
        random_state=RS,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params, round(float(search.best_score_), 4)


# =============================================================================
# 4. MODEL CARD GENERATOR (models/model_card.md)
# =============================================================================
def write_model_card(report: dict):
    lines = [
        "# Model Card – E-commerce Purchase Behavior Prediction",
        "",
        f"- **Trained:** {report['trained_at']}",
        f"- **Task:** {report['task_type']} – predict `{report['target']}` (`Yes` / `No`)",
        f"- **Best Model:** {report['best_model']}",
        f"- **Best Parameters:** {report['best_params']}",
        f"- **Data:** {report['n_train']:,} training rows, {report['n_test']:,} test rows (Retailrocket early-session snapshots)",
        f"- **Selection Metric:** {report['primary_metric']} ({config.CV_FOLDS}-fold Stratified CV)",
        "",
        "## Test-Set Performance",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in report["test_metrics"].items()],
        "",
        "## Model Comparison (Cross-Validation)",
        "",
        "| Model | " + config.PRIMARY_METRIC + " (mean ± std) | ROC-AUC (mean ± std) |",
        "|---|---|---|",
        *[
            f"| {r['model']} | {r[f'cv_{config.PRIMARY_METRIC}_mean']} ± {r[f'cv_{config.PRIMARY_METRIC}_std']} "
            f"| {r.get('cv_roc_auc_mean', '—')} ± {r.get('cv_roc_auc_std', '—')} |"
            for r in report["cv_results"]
        ],
        "",
        "## Methodological Notes & Limitations",
        "",
        "- **Leakage Prevention:** Features are extracted from the first 3 non-transaction events of each session plus strictly earlier sessions. Sessions that already purchased at or before event 3 are excluded.",
        "- **Scope:** Applies to engaged multi-event sessions (≥3 interactions); single-click bounce sessions lack an early observation window.",
        "- **Imbalanced Target:** Only ~7.6% of eligible sessions convert after the snapshot; evaluate with F1, Recall, Precision, PR-AUC, and ROC-AUC rather than accuracy alone.",
        "",
    ]
    config.MODEL_CARD_PATH.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 5. FULL PIPELINE EXECUTION (CLASSIFICATION + K-MEANS SEGMENTATION)
# =============================================================================
def run(verbose=True) -> dict:
    df = load_dataset()
    X, y, input_cols = prepare_xy(df)
    stratify = y if config.TASK_TYPE == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=RS, stratify=stratify
    )
    if verbose:
        print(f"Data: {len(X_train):,} train / {len(X_test):,} test rows, {len(input_cols)} input columns")
        print(f"\n1) Comparing models ({config.CV_FOLDS}-fold CV on training set):")

    cv_table = compare_models(X_train, y_train, verbose)
    best_name = cv_table.iloc[0]["model"]
    if verbose:
        print(f"\n2) Tuning best model: {best_name}")
    pipe, best_params, tuned_score = tune(best_name, X_train, y_train)
    if verbose:
        print(f"   best params: {best_params}  (CV {config.PRIMARY_METRIC} = {tuned_score})")

    if verbose:
        print("\n3) Evaluating on the TEST set (data the model has never seen):")
    metrics = evaluate.test_metrics(pipe, X_test, y_test)
    curves = evaluate.curves_data(pipe, X_test, y_test)
    importance = evaluate.feature_importance(pipe)
    figures = evaluate.save_figures(pipe, X_test, y_test, cv_table, importance)
    if verbose:
        for k, v in metrics.items():
            print(f"   {k:<10} {v}")
        txt = evaluate.report_text(pipe, X_test, y_test)
        if txt:
            print("\n" + txt)

    if verbose:
        print("4) Running K-Means Customer Segmentation:")
    seg_report = clustering.run(n_clusters=config.DEFAULT_K_CLUSTERS, verbose=verbose)

    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "task_type": config.TASK_TYPE,
        "target": config.TARGET_COLUMN,
        "primary_metric": config.PRIMARY_METRIC,
        "best_model": best_name,
        "best_params": best_params,
        "tuned_cv_score": tuned_score,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate": round(float(y.mean()), 4),
        "cv_results": cv_table.to_dict(orient="records"),
        "test_metrics": metrics,
        "confusion_matrix": evaluate.confusion(pipe, X_test, y_test),
        "curves": curves,
        "feature_importance": importance,
        "figures": figures + seg_report.get("figures", []),
        "clustering_summary": {
            "n_clusters": seg_report["n_clusters"],
            "n_customers": seg_report["n_customers"],
            "silhouette_score": seg_report["silhouette_score"],
            "inertia": seg_report["inertia"],
        },
    }

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "input_columns": input_cols}, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_model_card(report)
    if verbose:
        print("Saved ➜ models/model.joblib, models/metrics.json, models/model_card.md, reports/figures/")
    return report


if __name__ == "__main__":
    run()
