"""
All calls from Streamlit to the FastAPI backend go through this file.
Automatically falls back to calling `ml.service` directly if FastAPI is not running separately.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent
for p in (str(FRONTEND_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 90


class APIError(Exception):
    pass


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
        from ml import service

        params = kwargs.get("params") or {}
        payload = kwargs.get("json") or {}
        if path == "/api/health":
            return {"status": "ok", "backend": "direct (ml.service)"}
        if path == "/api/schema":
            return service.get_schema()
        if path == "/api/data":
            return service.get_data(limit=params.get("limit", 1000), dataset=params.get("dataset", "modeling"))
        if path == "/api/preprocess" and method == "POST":
            return service.run_preprocessing(timeout_minutes=payload.get("timeout_minutes", 30.0))
        if path == "/api/metrics":
            rep = service.get_metrics()
            if rep is None:
                raise APIError("404: Model is not trained yet.")
            return rep
        if path == "/api/train" and method == "POST":
            return service.train_model(params=payload or None)
        if path == "/api/experiments/activate" and method == "POST":
            return service.activate_experiment(payload.get("experiment_id", ""))
        if path == "/api/predict" and method == "POST":
            res = service.make_prediction(payload.get("features", {}))
            service.log_prediction(payload.get("features", {}), res)
            return res
        if path == "/api/predictions" and method == "GET":
            return {"items": service.get_predictions(limit=params.get("limit", 100))}
        if path == "/api/predictions" and method == "DELETE":
            return {"deleted": service.clear_predictions()}
        if path == "/api/segments":
            return service.get_segments(
                n_clusters=params.get("k"),
                scope=params.get("scope", "customer"),
                scaler=params.get("scaler", "standard"),
                use_log1p=params.get("use_log1p", True),
                random_state=params.get("random_state", 42),
            )
        if path == "/api/segments/fit" and method == "POST":
            return service.get_segments(
                n_clusters=payload.get("n_clusters", 4),
                scope=payload.get("scope", "customer"),
                features=payload.get("features"),
                scaler=payload.get("scaler", "standard"),
                use_log1p=payload.get("use_log1p", True),
                random_state=payload.get("random_state", 42),
            )
        if path == "/api/segment/predict" and method == "POST":
            return service.predict_customer_segment(payload.get("features", {}))
        if path == "/api/insights":
            return service.get_insights()
        raise APIError(f"Cannot connect to backend at {API_URL}.")


def health() -> dict:
    return _request("GET", "/api/health")


@st.cache_data(ttl=30, show_spinner=False)
def get_schema() -> dict:
    return _request("GET", "/api/schema")


@st.cache_data(ttl=30, show_spinner=False)
def get_data(limit: int = 1000, dataset: str = "modeling") -> dict:
    return _request("GET", "/api/data", params={"limit": limit, "dataset": dataset})


def preprocess(timeout_minutes: float = 30.0) -> dict:
    st.cache_data.clear()
    return _request("POST", "/api/preprocess", json={"timeout_minutes": float(timeout_minutes)})


@st.cache_data(ttl=15, show_spinner=False)
def get_metrics() -> dict:
    return _request("GET", "/api/metrics")


def train(params: dict | None = None) -> dict:
    st.cache_data.clear()
    return _request("POST", "/api/train", json=params or {})


def activate_experiment(experiment_id: str) -> dict:
    st.cache_data.clear()
    return _request("POST", "/api/experiments/activate", json={"experiment_id": experiment_id})


def predict(features: dict) -> dict:
    return _request("POST", "/api/predict", json={"features": features})


def get_history(limit: int = 100) -> dict:
    return _request("GET", "/api/predictions", params={"limit": limit})


def clear_history() -> dict:
    return _request("DELETE", "/api/predictions")


@st.cache_data(ttl=30, show_spinner=False)
def get_segments(k: int | None = None, scope: str = "customer") -> dict:
    params = {"scope": scope}
    if k is not None:
        params["k"] = k
    return _request("GET", "/api/segments", params=params)


def fit_segments(
    n_clusters: int = 4,
    scope: str = "customer",
    features: list[str] | None = None,
    scaler: str = "standard",
    use_log1p: bool = True,
    random_state: int = 42,
) -> dict:
    st.cache_data.clear()
    return _request(
        "POST",
        "/api/segments/fit",
        json={
            "n_clusters": int(n_clusters),
            "scope": scope,
            "features": features,
            "scaler": scaler,
            "use_log1p": bool(use_log1p),
            "random_state": int(random_state),
        },
    )


def predict_segment(features: dict) -> dict:
    return _request("POST", "/api/segment/predict", json={"features": features})


@st.cache_data(ttl=30, show_spinner=False)
def get_insights() -> dict:
    return _request("GET", "/api/insights")


def show_error_and_stop(err: Exception):
    st.error(str(err))
    st.info("Start the backend (`uvicorn backend_fastapi.main:app --reload --port 8000`) or train the model first.")
    st.stop()
