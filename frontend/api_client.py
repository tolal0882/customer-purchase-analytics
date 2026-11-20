"""
All calls from Streamlit to the backend go through this file.

Follows the teacher's architecture:
  1. By default calls the FastAPI backend over HTTP (http://localhost:8000).
  2. If the FastAPI server is not running (e.g. single-command demo or cloud deploy),
     transparently falls back to calling `ml.service` directly so the app never crashes!
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

# Ensure both frontend/ and project root are importable regardless of working directory
FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent
for p in (str(FRONTEND_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 60


class APIError(Exception):
    pass


# =============================================================================
# 1. HTTP REQUEST HELPER WITH GRACEFUL DIRECT-SERVICE FALLBACK
# =============================================================================
def _request(method: str, path: str, **kwargs):
    url = f"{API_URL}{path}"
    try:
        r = requests.request(method, url, timeout=TIMEOUT, **kwargs)
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise APIError(f"{r.status_code}: {detail}")
        return r.json()
    except requests.ConnectionError:
        # Fallback to direct ml.service calls if FastAPI is not running separately
        from ml import service

        params = kwargs.get("params") or {}
        payload = kwargs.get("json") or {}
        if path == "/api/health":
            return {"status": "ok", "backend": "direct (ml.service)"}
        if path == "/api/schema":
            return service.get_schema()
        if path == "/api/data":
            return service.get_data(limit=params.get("limit", 1000), dataset=params.get("dataset", "modeling"))
        if path == "/api/metrics":
            rep = service.get_metrics()
            if rep is None:
                raise APIError("404: Model is not trained yet.")
            return rep
        if path == "/api/train" and method == "POST":
            return service.train_model()
        if path == "/api/predict" and method == "POST":
            res = service.make_prediction(payload.get("features", {}))
            service.log_prediction(payload.get("features", {}), res)
            return res
        if path == "/api/predictions" and method == "GET":
            return {"items": service.get_predictions(limit=params.get("limit", 100))}
        if path == "/api/predictions" and method == "DELETE":
            return {"deleted": service.clear_predictions()}
        if path == "/api/segments":
            return service.get_segments(n_clusters=params.get("k"))
        if path == "/api/segment/predict" and method == "POST":
            return service.predict_customer_segment(payload.get("features", {}))
        if path == "/api/insights":
            return service.get_insights()
        raise APIError(f"Cannot connect to backend at {API_URL}.")


# =============================================================================
# 2. TYPED API CLIENT FUNCTIONS (CALLED BY ALL 8 STREAMLIT VIEWS)
# =============================================================================
def health() -> dict:
    return _request("GET", "/api/health")


@st.cache_data(ttl=60, show_spinner=False)
def get_schema() -> dict:
    return _request("GET", "/api/schema")


@st.cache_data(ttl=60, show_spinner=False)
def get_data(limit: int = 1000, dataset: str = "modeling") -> dict:
    return _request("GET", "/api/data", params={"limit": limit, "dataset": dataset})


@st.cache_data(ttl=30, show_spinner=False)
def get_metrics() -> dict:
    return _request("GET", "/api/metrics")


def train() -> dict:
    st.cache_data.clear()
    return _request("POST", "/api/train")


def predict(features: dict) -> dict:
    return _request("POST", "/api/predict", json={"features": features})


def get_history(limit: int = 100) -> dict:
    return _request("GET", "/api/predictions", params={"limit": limit})


def clear_history() -> dict:
    return _request("DELETE", "/api/predictions")


@st.cache_data(ttl=60, show_spinner=False)
def get_segments(k: int | None = None) -> dict:
    params = {"k": k} if k is not None else {}
    return _request("GET", "/api/segments", params=params)


def predict_segment(features: dict) -> dict:
    return _request("POST", "/api/segment/predict", json={"features": features})


@st.cache_data(ttl=60, show_spinner=False)
def get_insights() -> dict:
    return _request("GET", "/api/insights")


def show_error_and_stop(err: Exception):
    """Show a friendly error message and stop the page."""
    st.error(str(err))
    st.info("Start the backend (`uvicorn backend_fastapi.main:app --reload --port 8000`) or train the model first.")
    st.stop()
