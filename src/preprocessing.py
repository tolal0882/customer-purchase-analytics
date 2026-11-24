"""
Data Preprocessing Module (Section 5.1 & Page 3):
  - Explicit user-confirmed duplicate handling (never silently deletes rows or modifies raw CSV)
  - Separate Scikit-learn Pipeline & ColumnTransformer builders for:
      1. Supervised Classification (Purchase Intention Prediction)
      2. Unsupervised K-Means Clustering (Session Behavior Segmentation, Revenue strictly excluded)
  - Fits preprocessing strictly on training data when used in classification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src import data_loader, data_validation


def apply_cleaning_options(
    df: pd.DataFrame,
    drop_duplicates: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Return a cleaned copy of the dataset according to explicit user options, plus a before/after report."""
    raw_rows = len(df)
    dup_count = int(df.duplicated().sum())
    rev_before = df[data_validation.TARGET_COL].astype(bool)

    out = df.copy()
    # Ensure consistent types without altering values
    if "Weekend" in out.columns:
        out["Weekend"] = out["Weekend"].astype(bool)
    if data_validation.TARGET_COL in out.columns:
        out[data_validation.TARGET_COL] = out[data_validation.TARGET_COL].astype(bool)

    removed_dups = 0
    if drop_duplicates:
        out = out.drop_duplicates(keep="first").reset_index(drop=True)
        removed_dups = raw_rows - len(out)

    rev_after = out[data_validation.TARGET_COL].astype(bool)
    summary = {
        "drop_duplicates_applied": bool(drop_duplicates),
        "rows_before": raw_rows,
        "rows_after": len(out),
        "duplicates_detected": dup_count,
        "duplicates_removed": removed_dups,
        "duplicates_remaining": int(out.duplicated().sum()),
        "purchase_before": int(rev_before.sum()),
        "no_purchase_before": int(raw_rows - rev_before.sum()),
        "purchase_rate_before_pct": round(float(rev_before.mean() * 100), 2),
        "purchase_after": int(rev_after.sum()),
        "no_purchase_after": int(len(out) - rev_after.sum()),
        "purchase_rate_after_pct": round(float(rev_after.mean() * 100), 2),
    }
    return out, summary


def get_scaler(scaler_name: str = "standard"):
    if scaler_name == "robust":
        return RobustScaler()
    if scaler_name == "minmax":
        return MinMaxScaler()
    return StandardScaler()


def _log1p_array(X):
    return np.log1p(np.clip(np.asarray(X, dtype=float), 0.0, None))


def split_feature_types(selected_features: list[str]) -> tuple[list[str], list[str]]:
    """Separate selected features into numerical/continuous vs categorical/nominal columns."""
    clean_feats = [f for f in selected_features if f != data_validation.TARGET_COL]
    num_cols = [f for f in clean_feats if f in data_validation.CONTINUOUS_FEATURES]
    cat_cols = [f for f in clean_feats if f not in num_cols]
    return num_cols, cat_cols


def build_classification_preprocessor(
    selected_features: list[str],
    scaler_name: str = "standard",
    use_log1p: bool = False,
) -> ColumnTransformer:
    """Build a Scikit-learn ColumnTransformer for supervised purchase prediction.
    Must be fitted strictly on X_train inside a Pipeline."""
    if data_validation.TARGET_COL in selected_features:
        raise ValueError("Target column 'Revenue' must never be included in input features.")

    num_cols, cat_cols = split_feature_types(selected_features)
    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_log1p:
        num_steps.append(("log1p", FunctionTransformer(_log1p_array, validate=False, feature_names_out="one-to-one")))
    num_steps.append(("scaler", get_scaler(scaler_name)))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_clustering_preprocessor(
    selected_features: list[str],
    scaler_name: str = "standard",
    use_log1p: bool = True,
) -> ColumnTransformer:
    """Build a Scikit-learn ColumnTransformer for K-Means session segmentation.
    Strictly forbids 'Revenue' from being used as a clustering input."""
    if data_validation.TARGET_COL in selected_features:
        raise ValueError("CRITICAL RULE VIOLATION: 'Revenue' must never be included in clustering features.")

    num_cols, cat_cols = split_feature_types(selected_features)
    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_log1p:
        num_steps.append(("log1p", FunctionTransformer(_log1p_array, validate=False, feature_names_out="one-to-one")))
    num_steps.append(("scaler", get_scaler(scaler_name)))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_features_for_transform(df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    """Return a clean copy of X[selected_features] with categorical columns cast to string for OneHotEncoder."""
    X = df[selected_features].copy()
    _, cat_cols = split_feature_types(selected_features)
    for c in cat_cols:
        X[c] = X[c].astype(str)
    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def save_processed_csv(df: pd.DataFrame, filename: str = "online_shoppers_processed.csv") -> str:
    """Save a processed dataset to data/processed/ without touching data/raw/."""
    data_loader.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = data_loader.PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    return str(out_path)
