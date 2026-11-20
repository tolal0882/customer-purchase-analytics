"""Leakage-safe classification: pipelines, chronological split, training with real progress.

Design notes
  * Everything that learns from data (imputer, scaler, sampler, model) lives INSIDE one estimator
    (ResamplingClassifier -> Pipeline[ColumnTransformer, model]). It is fitted on the training
    window only, so test data never influences preprocessing.
  * Sampling (random under/over) is applied inside `fit`, i.e. only to the training rows that
    estimator is fitted on. Under cross-validation that means training folds only.
  * Splitting is chronological: samples are ordered by snapshot time; the latest `test_ratio`
    share is the test set. CV uses expanding-window TimeSeriesSplit on the training window."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src import evaluation, feature_engineering as fe

# =============================================================================
# 1. SUPPORTED ALGORITHMS & INTERACTIVE HYPERPARAMETER DEFINITIONS
#    - Defines the 4 classifiers (Logistic Regression, Decision Tree, Random
#      Forest, Gradient Boosting) and their UI slider/input bounds.
# =============================================================================
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
SAMPLING_METHODS = {"none": "None", "random_undersample": "Random under-sampling (majority)",
                    "random_oversample": "Random over-sampling (minority)"}

HYPERPARAM_SPECS: dict[str, list[dict]] = {
    "Logistic Regression": [
        {"name": "C", "label": "Inverse regularisation strength C", "type": "float", "min": 0.001, "max": 100.0, "default": 1.0, "log": True}],
    "Decision Tree": [
        {"name": "max_depth", "label": "Max depth", "type": "int", "min": 1, "max": 30, "default": 6},
        {"name": "min_samples_leaf", "label": "Min samples per leaf", "type": "int", "min": 1, "max": 500, "default": 20},
        {"name": "criterion", "label": "Split criterion", "type": "choice", "options": ["gini", "entropy"], "default": "gini"}],
    "Random Forest": [
        {"name": "n_estimators", "label": "Number of trees", "type": "int", "min": 10, "max": 500, "default": 150},
        {"name": "max_depth", "label": "Max depth", "type": "int", "min": 1, "max": 40, "default": 12},
        {"name": "min_samples_leaf", "label": "Min samples per leaf", "type": "int", "min": 1, "max": 200, "default": 10}],
    "Gradient Boosting": [
        {"name": "n_estimators", "label": "Boosting stages", "type": "int", "min": 10, "max": 500, "default": 150},
        {"name": "learning_rate", "label": "Learning rate", "type": "float", "min": 0.01, "max": 1.0, "default": 0.1, "log": True},
        {"name": "max_depth", "label": "Tree depth", "type": "int", "min": 1, "max": 10, "default": 3}],
}


def default_hyperparams(model_name: str) -> dict:
    return {s["name"]: s["default"] for s in HYPERPARAM_SPECS[model_name]}


def make_estimator(model_name: str, hyperparams: dict, balanced: bool, seed: int):
    hp = {**default_hyperparams(model_name), **(hyperparams or {})}
    cw = "balanced" if balanced else None
    if model_name == "Logistic Regression":
        return LogisticRegression(C=float(hp["C"]), class_weight=cw, max_iter=2000, random_state=seed)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=int(hp["max_depth"]), min_samples_leaf=int(hp["min_samples_leaf"]),
                                      criterion=hp["criterion"], class_weight=cw, random_state=seed)
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=int(hp["n_estimators"]), max_depth=int(hp["max_depth"]),
                                      min_samples_leaf=int(hp["min_samples_leaf"]), class_weight=cw,
                                      n_jobs=-1, random_state=seed)
    if model_name == "Gradient Boosting":
        # GradientBoostingClassifier has no class_weight: balanced weighting is applied via sample_weight in fit.
        return GradientBoostingClassifier(n_estimators=int(hp["n_estimators"]), learning_rate=float(hp["learning_rate"]),
                                          max_depth=int(hp["max_depth"]), random_state=seed)
    raise ValueError(f"Unknown model {model_name}")


# =============================================================================
# 2. LEAKAGE-SAFE ESTIMATOR WRAPPER (ResamplingClassifier) & PIPELINE BUILDER
#    - Wraps ColumnTransformer (median imputation + StandardScaler) + classifier.
#    - Any optional under/over-sampling happens INSIDE fit(), guaranteeing it
#      touches ONLY training folds and NEVER validation or test data.
#    - GradientBoosting lacks class_weight, so balanced sample_weight is computed
#      automatically inside fit() when balanced=True.
# =============================================================================
class ResamplingClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a Pipeline; optionally resamples (train rows only) and applies balanced sample weights."""

    def __init__(self, estimator=None, sampling="none", sampling_ratio=1.0, balanced=False, seed=42, threshold=0.5):
        self.estimator = estimator
        self.sampling = sampling
        self.sampling_ratio = sampling_ratio
        self.balanced = balanced
        self.seed = seed
        self.threshold = threshold

    def _resample(self, X: pd.DataFrame, y: np.ndarray):
        if self.sampling == "none":
            return X, y
        rng = np.random.RandomState(self.seed)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if len(pos) == 0 or len(neg) == 0:
            return X, y
        r = float(self.sampling_ratio)  # target minority:majority ratio
        if self.sampling == "random_undersample":
            keep_neg = min(len(neg), max(len(pos), int(len(pos) / r)))
            idx = np.concatenate([pos, rng.choice(neg, keep_neg, replace=False)])
        elif self.sampling == "random_oversample":
            want_pos = max(len(pos), int(len(neg) * r))
            idx = np.concatenate([neg, pos, rng.choice(pos, want_pos - len(pos), replace=True)])
        else:
            raise ValueError(self.sampling)
        idx = np.sort(idx)
        return X.iloc[idx], y[idx]

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y).astype(int)
        Xr, yr = self._resample(X, y)
        est = clone(self.estimator)
        fit_params = {}
        final = est.steps[-1][1]
        if self.balanced and "class_weight" not in final.get_params():
            fit_params[f"{est.steps[-1][0]}__sample_weight"] = compute_sample_weight("balanced", yr)
        est.fit(Xr, yr, **fit_params)
        self.estimator_ = est
        self.classes_ = est.classes_
        self.n_fit_samples_ = len(yr)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(pd.DataFrame(X))

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)


def build_pipeline(cfg: "TrainConfig") -> ResamplingClassifier:
    fe.assert_no_leakage_features(cfg.features)
    prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                                ("scale", StandardScaler())]), list(cfg.features))],
                             remainder="drop")
    pipe = Pipeline([("prep", prep), ("model", make_estimator(cfg.model_name, cfg.hyperparams, cfg.balanced, cfg.seed))])
    return ResamplingClassifier(pipe, cfg.sampling, cfg.sampling_ratio, cfg.balanced, cfg.seed, cfg.threshold)


@dataclass
class TrainConfig:
    model_name: str = "Logistic Regression"
    features: list = field(default_factory=lambda: list(fe.SNAPSHOT_BASE_FEATURES))
    test_ratio: float = 0.2
    seed: int = 42
    cv_folds: int = 3
    balanced: bool = True
    hyperparams: dict = field(default_factory=dict)
    sampling: str = "none"
    sampling_ratio: float = 0.5
    threshold: float = 0.5


@dataclass
class TrainResult:
    pipeline: ResamplingClassifier
    config: TrainConfig
    metrics: dict
    cv: pd.DataFrame
    split: dict
    importance: dict
    feature_medians: dict
    train_seconds: float


# =============================================================================
# 3. CHRONOLOGICAL TRAIN/TEST SPLIT & EXPANDING-WINDOW CROSS-VALIDATION
#    - Orders all snapshots by snapshot_ts: earliest (1-test_ratio) -> train,
#      latest test_ratio -> held-out test set (prevents future-to-past leakage).
#    - Runs TimeSeriesSplit CV inside the training window, then fits the final
#      pipeline on the full training window and evaluates on the test window.
# =============================================================================
def chronological_split(df: pd.DataFrame, test_ratio: float):
    """Earliest (1-test_ratio) samples -> train, latest -> test (df must be ordered by snapshot_ts)."""
    if not df["snapshot_ts"].is_monotonic_increasing:
        df = df.sort_values("snapshot_ts", kind="stable")
    cut = int(len(df) * (1 - test_ratio))
    return df.iloc[:cut], df.iloc[cut:]


def train_model(data: pd.DataFrame, cfg: TrainConfig,
                progress: Callable[[float, str], None] | None = None) -> TrainResult:
    """Train with chronological CV + final fit + held-out chronological test evaluation.

    `progress(fraction, message)` is called after each REAL unit of work (each CV fold, final fit,
    test evaluation, importance)."""
    t0 = time.time()
    if not cfg.features:
        raise ValueError("Select at least one feature.")
    train, test = chronological_split(data, cfg.test_ratio)
    if train["target"].nunique() < 2 or test["target"].nunique() < 2:
        raise ValueError("Train or test window contains a single class; choose another split ratio.")
    X_tr, y_tr = train[cfg.features], train["target"].to_numpy()
    X_te, y_te = test[cfg.features], test["target"].to_numpy()
    steps_total = cfg.cv_folds + 3
    done = 0

    def tick(msg):
        nonlocal done
        done += 1
        if progress:
            progress(min(done / steps_total, 1.0), msg)

    rows = []
    if cfg.cv_folds >= 2:
        tscv = TimeSeriesSplit(n_splits=cfg.cv_folds)
        for i, (a, b) in enumerate(tscv.split(X_tr), 1):
            m = build_pipeline(cfg).fit(X_tr.iloc[a], y_tr[a])
            p = m.predict_proba(X_tr.iloc[b])[:, 1]
            met = evaluation.compute_metrics(y_tr[b], p, cfg.threshold, with_curves=False)
            rows.append({"fold": i, "train_samples": len(a), "validation_samples": len(b),
                         **{k: met[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}})
            tick(f"Cross-validation fold {i}/{cfg.cv_folds} done")
    cv = pd.DataFrame(rows)

    model = build_pipeline(cfg).fit(X_tr, y_tr)
    tick("Final model fitted on training window")
    proba = model.predict_proba(X_te)[:, 1]
    metrics = evaluation.compute_metrics(y_te, proba, cfg.threshold)
    metrics["train_prevalence"] = float(y_tr.mean())
    tick("Held-out test evaluation done")

    importance = evaluation.feature_importance(model, cfg.features, X_te, y_te, seed=cfg.seed)
    tick("Feature importance computed")
    split = {
        "train_samples": len(train), "test_samples": len(test),
        "train_positives": int(y_tr.sum()), "test_positives": int(y_te.sum()),
        "train_start_ts": int(train["snapshot_ts"].iloc[0]), "train_end_ts": int(train["snapshot_ts"].iloc[-1]),
        "test_start_ts": int(test["snapshot_ts"].iloc[0]), "test_end_ts": int(test["snapshot_ts"].iloc[-1]),
        "samples_fitted_after_sampling": int(model.n_fit_samples_),
    }
    medians = {f: (None if np.isnan(v) else float(v)) for f, v in X_tr.median().items()}
    return TrainResult(model, cfg, metrics, cv, split, importance, medians, time.time() - t0)


def config_to_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
