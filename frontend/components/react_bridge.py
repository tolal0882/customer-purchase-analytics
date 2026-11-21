"""
React custom component bridge for Streamlit (`frontend/components/react_dist`).

Serves the compiled Vite + React + TypeScript + Tailwind CSS bundle for:
  - `prediction_form`: interactive early-session behavior form with presets, live derived ratios & validation
  - `cluster_explorer`: interactive K-Means customer/session segment cards with z-score bars
  - `model_compare`: interactive classifier CV & experiment comparison leaderboard
Falls back gracefully to None if the bundle is absent.
"""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

DIST_DIR = Path(__file__).resolve().parent / "react_dist"
_react_comp = (
    components.declare_component("retailrocket_react", path=str(DIST_DIR))
    if (DIST_DIR / "index.html").exists()
    else None
)


def is_available() -> bool:
    return _react_comp is not None


def prediction_form(
    features: list[str],
    values: dict,
    policy: str = "n_events",
    n_events: int = 3,
    reset_token: str = "init",
    key: str = "react_pred_form",
):
    """Render the interactive React + Tailwind Purchase Prediction form with presets & live derived ratios."""
    if _react_comp is None:
        return None
    # Map our pipeline feature names to the React component's expected props
    react_features = list(features)
    if "prior_sessions" in features and "hist_prior_sessions" not in react_features:
        react_features.extend(["hist_prior_sessions", "hist_prior_purchases"])
    react_vals = {
        "total_events": int(values.get("views_count", 2)) + int(values.get("add_to_cart_count", 1)),
        "total_views": int(values.get("views_count", 2)),
        "unique_items_viewed": int(values.get("unique_items_viewed", 2)),
        "add_to_cart_count": int(values.get("add_to_cart_count", 1)),
        "session_duration_sec": float(values.get("elapsed_seconds", 90.0)),
        "first_cart_position": int(values.get("first_cart_position", 2)),
        "ends_with_cart": str(values.get("ends_with_cart", "Yes")) in ("Yes", "True", "1"),
        "start_hour": int(values.get("start_hour", 14)),
        "start_dayofweek": int(values.get("start_dayofweek", 2)),
        "hist_prior_sessions": int(values.get("prior_sessions", 0)),
        "hist_prior_purchases": int(values.get("prior_purchases", 0)),
        "hist_hours_since_prev_session": 0.0,
    }
    return _react_comp(
        view="prediction_form",
        features=react_features,
        policy=policy,
        n_events=int(n_events),
        values=react_vals,
        reset_token=reset_token,
        key=key,
        default=None,
    )


def map_react_prediction_values(raw_react_vals: dict) -> dict:
    """Convert the React form's returned payload into our pipeline's feature dictionary."""
    dow = int(raw_react_vals.get("start_dayofweek", 2))
    return {
        "views_count": int(raw_react_vals.get("total_views", 2)),
        "unique_items_viewed": int(raw_react_vals.get("unique_items_viewed", 2)),
        "add_to_cart_count": int(raw_react_vals.get("add_to_cart_count", 1)),
        "first_cart_position": int(raw_react_vals.get("first_cart_position", 2)),
        "elapsed_seconds": float(raw_react_vals.get("session_duration_sec", 90.0)),
        "ends_with_cart": "Yes" if raw_react_vals.get("ends_with_cart") else "No",
        "start_hour": int(raw_react_vals.get("start_hour", 14)),
        "start_dayofweek": dow,
        "is_weekend": "Yes" if dow >= 5 else "No",
        "prior_sessions": int(raw_react_vals.get("hist_prior_sessions", 0)),
        "prior_purchases": int(raw_react_vals.get("hist_prior_purchases", 0)),
    }


def cluster_explorer(profiles: list[dict], selected_id: int | None = None, key: str = "react_clusters"):
    """Render interactive React segment profile cards."""
    if _react_comp is None:
        return None
    clusters = [
        {
            "id": int(p["cluster_id"]),
            "name": p["segment_name"],
            "description": p["description"],
            "size": int(p["size"]),
            "share_pct": float(p["share_pct"]),
            "purchase_rate_pct": float(p.get("conversion_rate_pct", 0.0)),
            "z": {k: float(v) for k, v in p.get("feature_z_scores", {}).items()},
        }
        for p in profiles
    ]
    return _react_comp(view="cluster_explorer", clusters=clusters, selected=selected_id, key=key, default=None)


def model_compare(cv_results: list[dict], best_model: str, primary_metric: str = "f1", key: str = "react_models"):
    """Render interactive React model comparison leaderboard from 5-fold CV results."""
    if _react_comp is None:
        return None
    metrics_list = ["f1", "roc_auc", "recall", "precision", "accuracy"]
    exps = [
        {
            "id": r["model"],
            "label": r["model"],
            "model": "5-Fold CV",
            "created_at": f"±{r.get(f'cv_{primary_metric}_std', 0):.4f} std",
            "active": r["model"] == best_model,
            "metrics": {m: float(r.get(f"cv_{m}_mean", 0.0)) for m in metrics_list},
        }
        for r in cv_results
    ]
    return _react_comp(
        view="model_compare",
        experiments=exps,
        selected=[r["model"] for r in cv_results],
        metric=primary_metric if primary_metric in metrics_list else "f1",
        metrics=metrics_list,
        key=key,
        default=None,
    )
