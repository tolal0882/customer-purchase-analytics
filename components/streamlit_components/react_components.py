"""Python wrappers for the React/TypeScript/Tailwind components (built with Vite into components/react/dist).

If the build is missing, the wrappers return None and the callers fall back to native Streamlit controls,
so the app keeps working (and says so)."""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

DIST = Path(__file__).resolve().parents[1] / "react" / "dist"
_component = components.declare_component("retail_react", path=str(DIST)) if (DIST / "index.html").exists() else None


def react_available() -> bool:
    return _component is not None


def prediction_form(features: list[str], policy: str, n_events: int, values: dict, reset_token: str, key: str):
    if not _component:
        return None
    return _component(view="prediction_form", features=features, policy=policy, n_events=n_events, values=values,
                      reset_token=reset_token, key=key, default=None)


def cluster_explorer(clusters: list[dict], selected: int | None, key: str):
    if not _component:
        return None
    return _component(view="cluster_explorer", clusters=clusters, selected=selected, key=key, default=None)


def model_compare(experiments: list[dict], selected: list[str], metric: str, metrics: list[str], key: str):
    if not _component:
        return None
    return _component(view="model_compare", experiments=experiments, selected=selected, metric=metric,
                      metrics=metrics, key=key, default=None)
