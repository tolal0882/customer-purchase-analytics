"""
Step 3 – Supervised & Unsupervised Modelling:
  - Compares 4 classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
  - Supports interactive custom training (algorithm, test split ratio, chronological vs stratified split,
    random seed, CV folds, feature selection, balanced class weights, training-fold resampling,
    decision threshold, and custom hyperparameters)
  - Persists every experiment in models/experiments.json and active pipeline in models/model.joblib
  - Also runs K-Means Customer & Session Segmentation

Run from the project root:
    python -m ml.train
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight

from ml import clustering, config, evaluate
from ml.data import make_dataset
from ml.features import build_preprocessor, get_input_columns

RS = config.RANDOM_STATE


# =============================================================================
# 1. TRAINING-FOLD RESAMPLING & THRESHOLD WRAPPER (ZERO TEST LEAKAGE)
# =============================================================================
class TrainingFoldResampler(ClassifierMixin, BaseEstimator):
    """Wraps a Pipeline so optional under/over-sampling & balanced sample weights
    run strictly inside fit() on training folds—never touching validation/test rows."""
    _estimator_type = "classifier"

    def __init__(
        self,
        pipeline=None,
        sampling: str = "none",
        balanced: bool = True,
        random_state: int = 42,
        threshold: float = 0.5,
    ):
        self.pipeline = pipeline
        self.sampling = sampling
        self.balanced = balanced
        self.random_state = random_state
        self.threshold = threshold

    def _resample(self, X: pd.DataFrame, y: np.ndarray):
        if self.sampling not in ("undersample", "oversample"):
            return X, y
        rng = np.random.RandomState(self.random_state)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        if len(pos) == 0 or len(neg) == 0:
            return X, y
        if self.sampling == "undersample":
            keep_neg = min(len(neg), max(len(pos), len(pos) * 2))
            idx = np.sort(np.concatenate([pos, rng.choice(neg, size=keep_neg, replace=False)]))
        else:
            want_pos = max(len(pos), int(len(neg) * 0.5))
            extra = rng.choice(pos, size=max(0, want_pos - len(pos)), replace=True)
            idx = np.sort(np.concatenate([neg, pos, extra]))
        return X.iloc[idx].reset_index(drop=True), y[idx]

    def fit(self, X, y):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(y).astype(int)
        X_res, y_res = self._resample(X_df, y_arr)
        pipe = clone(self.pipeline)
        fit_kwargs = {}
        model_step = pipe.named_steps["model"]
        if self.balanced and "class_weight" not in model_step.get_params():
            fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_res)
        pipe.fit(X_res, y_res, **fit_kwargs)
        self.pipeline_ = pipe
        self.classes_ = getattr(pipe, "classes_", np.array([0, 1]))
        self.n_fitted_ = len(y_res)
        return self

    @property
    def named_steps(self):
        return self.pipeline_.named_steps

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(pd.DataFrame(X))

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= float(self.threshold)).astype(int)


# =============================================================================
# 2. CANDIDATE CLASSIFICATION MODELS & HYPERPARAMETER BUILDERS
# =============================================================================
def build_estimator(
    name: str,
    hyperparams: dict | None = None,
    balanced: bool = True,
    random_state: int = RS,
):
    """Instantiate one of the 4 classifiers with user-chosen hyperparameters."""
    hp = hyperparams or {}
    cw = "balanced" if balanced else None
    if name == "Logistic Regression":
        return LogisticRegression(
            C=float(hp.get("C", 1.0)),
            max_iter=2000,
            class_weight=cw,
            random_state=random_state,
        )
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=int(hp.get("max_depth", 6)),
            min_samples_leaf=int(hp.get("min_samples_leaf", 20)),
            criterion=str(hp.get("criterion", "gini")),
            class_weight=cw,
            random_state=random_state,
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(hp.get("n_estimators", 120)),
            max_depth=int(hp.get("max_depth", 12)),
            min_samples_leaf=int(hp.get("min_samples_leaf", 10)),
            class_weight=cw,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=int(hp.get("n_estimators", 100)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            max_depth=int(hp.get("max_depth", 3)),
            random_state=random_state,
        )
    raise ValueError(f"Unknown model: {name}")


def candidate_models(balanced: bool = True, random_state: int = RS) -> dict:
    """Models to compare + hyperparameter grids for RandomizedSearchCV tuning."""
    cw = "balanced" if balanced else None
    if config.TASK_TYPE == "classification":
        return {
            "Logistic Regression": (
                LogisticRegression(max_iter=2000, class_weight=cw, random_state=random_state),
                {"model__C": [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]},
            ),
            "Decision Tree": (
                DecisionTreeClassifier(random_state=random_state, class_weight=cw),
                {"model__max_depth": [4, 5, 6, 8, 10], "model__min_samples_leaf": [10, 20, 50]},
            ),
            "Random Forest": (
                RandomForestClassifier(
                    n_estimators=120, max_depth=12, min_samples_leaf=10,
                    random_state=random_state, n_jobs=-1, class_weight=cw,
                ),
                {
                    "model__n_estimators": [100, 150, 200],
                    "model__max_depth": [8, 10, 14],
                    "model__min_samples_leaf": [5, 10, 20],
                },
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_state),
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
        "Random Forest": (RandomForestRegressor(random_state=random_state, n_jobs=-1), {}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=random_state), {}),
    }


def scoring() -> list[str]:
    if config.TASK_TYPE == "classification":
        return ["accuracy", "precision", "recall", "f1", "roc_auc"]
    return ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]


# =============================================================================
# 3. DATASET LOADER & X/y PREPARATION
# =============================================================================
def load_dataset() -> pd.DataFrame:
    if not config.PROCESSED_DATA_PATH.exists():
        print("Processed data not found – running make_dataset first.")
        make_dataset.main()
    return pd.read_csv(config.PROCESSED_DATA_PATH)


def prepare_xy(df: pd.DataFrame, selected_features: list[str] | None = None):
    all_input_cols = get_input_columns(df)
    if selected_features:
        input_cols = [c for c in selected_features if c in all_input_cols]
        if not input_cols:
            input_cols = all_input_cols
    else:
        input_cols = all_input_cols
    X = df[input_cols].copy()
    y = df[config.TARGET_COLUMN]
    if config.TASK_TYPE == "classification":
        y = (y == config.POSITIVE_CLASS).astype(int)
    return X, y, input_cols


# =============================================================================
# 4. CROSS-VALIDATION COMPARISON & TUNING
# =============================================================================
def compare_models(
    X_train,
    y_train,
    cv_folds: int = config.CV_FOLDS,
    balanced: bool = True,
    random_state: int = RS,
    split_mode: str = "stratified",
    verbose: bool = True,
) -> pd.DataFrame:
    """Cross-validate every candidate model on the TRAINING set only."""
    if split_mode == "chronological":
        cv = TimeSeriesSplit(n_splits=int(cv_folds))
    elif config.TASK_TYPE == "classification":
        cv = StratifiedKFold(int(cv_folds), shuffle=True, random_state=int(random_state))
    else:
        cv = KFold(int(cv_folds), shuffle=True, random_state=int(random_state))

    rows = []
    for name, (model, _) in candidate_models(balanced=balanced, random_state=random_state).items():
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


def tune(name, X_train, y_train, cv_folds: int = config.CV_FOLDS, balanced: bool = True, random_state: int = RS):
    """Tune the top-performing model on the training set using RandomizedSearchCV."""
    model, params = candidate_models(balanced=balanced, random_state=random_state)[name]
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
        cv=int(cv_folds),
        random_state=int(random_state),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params, round(float(search.best_score_), 4)


# =============================================================================
# 5. EXPERIMENT HISTORY & MODEL CARD PERSISTENCE
# =============================================================================
def _save_to_experiments_registry(report: dict, bundle: dict) -> dict:
    """Save experiment metadata into models/experiments.json and pipeline to models/."""
    exp_id = report.get("experiment_id") or f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    report["experiment_id"] = exp_id
    exp_pipe_path = config.MODEL_PATH.parent / f"model_{exp_id}.joblib"
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, exp_pipe_path)
    joblib.dump(bundle, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_model_card(report)

    exps = []
    if config.EXPERIMENTS_PATH.exists():
        try:
            exps = json.loads(config.EXPERIMENTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            exps = []
    # Keep newest first, cap at 25 saved experiments
    exps = [e for e in exps if e.get("experiment_id") != exp_id]
    exps.insert(0, report)
    config.EXPERIMENTS_PATH.write_text(json.dumps(exps[:25], indent=2, default=str), encoding="utf-8")
    return report


def write_model_card(report: dict):
    lines = [
        "# Model Card – E-commerce Purchase Behavior Prediction",
        "",
        f"- **Experiment ID:** `{report.get('experiment_id', 'default')}`",
        f"- **Trained:** {report['trained_at']}",
        f"- **Task:** {report['task_type']} – predict `{report['target']}` (`Yes` / `No`)",
        f"- **Best Model:** {report['best_model']}",
        f"- **Best Parameters:** {report['best_params']}",
        f"- **Data:** {report['n_train']:,} training rows, {report['n_test']:,} test rows",
        f"- **Selection Metric:** {report['primary_metric']}",
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
        "- **Imbalanced Target:** Only ~7.6% of eligible sessions convert after the snapshot; evaluate with F1, Recall, Precision, PR-AUC, and ROC-AUC rather than accuracy alone.",
        "",
    ]
    config.MODEL_CARD_PATH.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 6. INTERACTIVE CUSTOM TRAINING & FULL BENCHMARK PIPELINE
# =============================================================================
def _compute_feature_defaults(X_train: pd.DataFrame) -> dict:
    """Compute training-set medians/modes used for local what-if perturbation explanations."""
    defaults = {}
    for col in X_train.columns:
        s = X_train[col].dropna()
        if pd.api.types.is_numeric_dtype(s):
            defaults[col] = float(s.median()) if len(s) else 0.0
        else:
            defaults[col] = str(s.mode().iloc[0]) if len(s) else "No"
    return defaults


def train_custom(
    model_name: str = "all",
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    cv_folds: int = config.CV_FOLDS,
    selected_features: list[str] | None = None,
    balanced: bool = True,
    sampling: str = "none",
    threshold: float = 0.5,
    split_mode: str = "chronological",
    hyperparams: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Interactive training entrypoint supporting custom features, hyperparameters, sampling & split modes."""
    df = load_dataset()
    X, y, input_cols = prepare_xy(df, selected_features=selected_features)

    # Chronological split (preserves time order) or Stratified random split
    if split_mode == "chronological":
        cut = int(len(X) * (1.0 - float(test_size)))
        X_train, X_test = X.iloc[:cut].reset_index(drop=True), X.iloc[cut:].reset_index(drop=True)
        y_train, y_test = y.iloc[:cut].reset_index(drop=True), y.iloc[cut:].reset_index(drop=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
        )

    if model_name == "all" or model_name not in candidate_models():
        cv_table = compare_models(
            X_train, y_train, cv_folds=cv_folds, balanced=balanced,
            random_state=random_state, split_mode=split_mode, verbose=verbose,
        )
        best_name = cv_table.iloc[0]["model"]
        base_pipe, best_params, tuned_score = tune(
            best_name, X_train, y_train, cv_folds=cv_folds, balanced=balanced, random_state=random_state
        )
        if sampling != "none" or float(threshold) != 0.5:
            pipe = TrainingFoldResampler(
                base_pipe, sampling=sampling, balanced=balanced, random_state=random_state, threshold=threshold
            ).fit(X_train, y_train)
        else:
            pipe = base_pipe
    else:
        best_name = model_name
        best_params = hyperparams or {}
        est = build_estimator(best_name, hyperparams=best_params, balanced=balanced, random_state=random_state)
        base_pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", est)])
        cv_obj = (
            TimeSeriesSplit(n_splits=int(cv_folds))
            if split_mode == "chronological"
            else StratifiedKFold(int(cv_folds), shuffle=True, random_state=int(random_state))
        )
        wrapped = TrainingFoldResampler(
            base_pipe, sampling=sampling, balanced=balanced, random_state=random_state, threshold=threshold
        )
        res = cross_validate(wrapped, X_train, y_train, cv=cv_obj, scoring=scoring(), n_jobs=-1)
        row = {"model": best_name}
        for m in scoring():
            row[f"cv_{m}_mean"] = round(float(res[f"test_{m}"].mean()), 4)
            row[f"cv_{m}_std"] = round(float(res[f"test_{m}"].std()), 4)
        cv_table = pd.DataFrame([row])
        tuned_score = row[f"cv_{config.PRIMARY_METRIC}_mean"]
        pipe = wrapped.fit(X_train, y_train)

    # Evaluate on held-out test set respecting decision threshold
    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= float(threshold)).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cls_rep = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=[config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
        output_dict=True, zero_division=0,
    )
    curves = evaluate.curves_data(pipe, X_test, y_test)
    importance = evaluate.feature_importance(pipe)
    figures = evaluate.save_figures(pipe, X_test, y_test, cv_table, importance)
    defaults = _compute_feature_defaults(X_train)

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
        "n_fitted_after_sampling": int(getattr(pipe, "n_fitted_", len(X_train))),
        "positive_rate": round(float(y.mean()), 4),
        "training_config": {
            "model_selection": model_name,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "cv_folds": int(cv_folds),
            "selected_features": input_cols,
            "balanced": bool(balanced),
            "sampling": sampling,
            "threshold": float(threshold),
            "split_mode": split_mode,
            "hyperparams": best_params,
        },
        "cv_results": cv_table.to_dict(orient="records"),
        "test_metrics": metrics,
        "confusion_matrix": {"labels": [config.NEGATIVE_CLASS, config.POSITIVE_CLASS], "matrix": cm.tolist()},
        "classification_report": cls_rep,
        "curves": curves,
        "feature_importance": importance,
        "feature_defaults": defaults,
        "figures": figures,
    }
    bundle = {
        "pipeline": pipe,
        "input_columns": input_cols,
        "threshold": float(threshold),
        "feature_defaults": defaults,
        "model_name": best_name,
    }
    return _save_to_experiments_registry(report, bundle)


def run(verbose=True) -> dict:
    """Standard teacher-template CLI run: compare all 4 classifiers, tune winner, and run K-Means segmentation."""
    report = train_custom(model_name="all", split_mode="stratified", verbose=verbose)
    if verbose:
        print("4) Running K-Means Customer & Session Segmentation:")
    seg_cust = clustering.run(n_clusters=config.DEFAULT_K_CLUSTERS, scope="customer", verbose=verbose)
    clustering.run(n_clusters=config.DEFAULT_K_CLUSTERS, scope="session", verbose=verbose)
    report["clustering_summary"] = {
        "n_clusters": seg_cust["n_clusters"],
        "n_customers": seg_cust["n_customers"],
        "silhouette_score": seg_cust["silhouette_score"],
        "inertia": seg_cust["inertia"],
    }
    config.METRICS_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    run()
