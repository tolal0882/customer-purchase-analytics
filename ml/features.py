"""
Step 2 – Feature engineering and preprocessing (follows teacher's template).

Everything here runs INSIDE the Scikit-learn Pipeline via FunctionTransformer,
so the FastAPI backend / Streamlit form sends the raw session fields and the
pipeline computes all engineered behavioral ratios automatically without leakage.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml import config


# =============================================================================
# 1. DOMAIN FEATURE ENGINEERING (RUNS INSIDE THE MODEL PIPELINE)
#    - cart_to_view_ratio:     add_to_cart_count / max(views_count, 1)
#    - cart_interaction_ratio: add_to_cart_count / SNAPSHOT_EVENTS
#    - item_focus_ratio:       1 - (unique_items_viewed / max(views_count, 1))
#    - avg_seconds_per_action: elapsed_seconds / (SNAPSHOT_EVENTS - 1)
#    - has_early_cart:         1 if add_to_cart_count > 0 else 0
#    - is_returning_visitor:   1 if prior_sessions > 0 else 0
#    - historical_buy_rate:    prior_purchases / max(prior_sessions, 1)
# =============================================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived e-commerce behavioral features from raw early-session inputs."""
    df = df.copy()
    if {"add_to_cart_count", "views_count"} <= set(df.columns):
        views_safe = df["views_count"].clip(lower=1)
        df["cart_to_view_ratio"] = (df["add_to_cart_count"] / views_safe).round(4)
    if "add_to_cart_count" in df.columns:
        df["cart_interaction_ratio"] = (df["add_to_cart_count"] / max(config.SNAPSHOT_EVENTS, 1)).round(4)
        df["has_early_cart"] = (df["add_to_cart_count"] > 0).astype(int)
    if {"unique_items_viewed", "views_count"} <= set(df.columns):
        views_safe = df["views_count"].clip(lower=1)
        df["item_focus_ratio"] = (1.0 - (df["unique_items_viewed"] / views_safe).clip(0, 1)).round(4)
    if "elapsed_seconds" in df.columns:
        df["avg_seconds_per_action"] = (df["elapsed_seconds"] / max(config.SNAPSHOT_EVENTS - 1, 1)).round(2)
    if "prior_sessions" in df.columns:
        df["is_returning_visitor"] = (df["prior_sessions"] > 0).astype(int)
    if {"prior_purchases", "prior_sessions"} <= set(df.columns):
        df["historical_buy_rate"] = np.where(
            df["prior_sessions"] > 0,
            (df["prior_purchases"] / df["prior_sessions"].clip(lower=1)).round(4),
            0.0,
        )
    return df


# =============================================================================
# 2. INPUT COLUMN & DATA TYPE HELPERS
# =============================================================================
def get_input_columns(df: pd.DataFrame) -> list[str]:
    """Return columns the user/API must provide (excluding target and forbidden IDs)."""
    skip = {config.TARGET_COLUMN, *config.DROP_COLUMNS}
    return [c for c in df.columns if c not in skip]


def split_types(df: pd.DataFrame):
    """Split columns into numeric and categorical lists AFTER feature engineering."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


# =============================================================================
# 3. SCIKIT-LEARN PREPROCESSING PIPELINE
# =============================================================================
def build_preprocessor(X_sample: pd.DataFrame) -> Pipeline:
    """add_features ➜ SimpleImputer ➜ StandardScaler / OneHotEncoder."""
    numeric, categorical = split_types(add_features(X_sample))
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    columns = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("add_features", FunctionTransformer(add_features)),
        ("columns", columns),
    ])
