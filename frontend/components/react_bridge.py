"""
React custom component bridge for Streamlit (`frontend/components/react_dist`).

Serves the compiled Vite + React + TypeScript + Tailwind CSS bundle for:
  - `cluster_explorer`: interactive K-Means customer segment cards with z-score bars
  - `model_compare`: interactive 4-classifier CV leaderboard
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
            "model": "5-Fold Stratified CV",
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
