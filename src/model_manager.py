"""
Model Persistence Manager using Joblib (Section 6 & Section 11):
  - Saves complete fitted classification pipelines, preprocessing objects, model name,
    selected features, training configuration, dataset information, evaluation metrics,
    and training timestamp to `models/classification/` (with automatic `/tmp` fallback
    if the container filesystem is read-only or watched).
  - Keeps in-memory session cache so Streamlit Cloud never crashes on disk writes.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import joblib

from src import data_loader

_MEM_CACHE: dict = {
    "active_bundle": None,
    "bundles_by_id": {},
    "registry": [],
    "clustering_bundle": None,
}


def _writable_models_dir() -> Path:
    """Return project `models/` if writable, otherwise a writable temp directory."""
    proj_models = data_loader.ROOT_DIR / "models"
    try:
        proj_models.mkdir(parents=True, exist_ok=True)
        test_file = proj_models / ".write_check"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return proj_models
    except Exception:
        fallback = Path(tempfile.gettempdir()) / "uci_online_shoppers_models"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


MODELS_DIR = _writable_models_dir()
CLASSIFICATION_DIR = MODELS_DIR / "classification"
CLUSTERING_DIR = MODELS_DIR / "clustering"
ACTIVE_CLASSIFIER_PATH = CLASSIFICATION_DIR / "active_classifier.joblib"
ACTIVE_METADATA_PATH = CLASSIFICATION_DIR / "active_metadata.json"
REGISTRY_PATH = CLASSIFICATION_DIR / "models_registry.json"


def ensure_model_dirs() -> None:
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)


def save_classification_bundle(bundle: dict) -> dict:
    """Persist a fitted classification pipeline bundle with Joblib and update the registry."""
    ensure_model_dirs()
    ts = bundle.get("trained_at") or datetime.now().isoformat(timespec="seconds")
    model_id = bundle.get("model_id") or f"{bundle['model_name'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bundle["model_id"] = model_id
    bundle["trained_at"] = ts

    # Always store in memory first
    _MEM_CACHE["active_bundle"] = bundle
    _MEM_CACHE["bundles_by_id"][model_id] = bundle

    meta = {k: v for k, v in bundle.items() if k != "pipeline"}
    model_file = CLASSIFICATION_DIR / f"{model_id}.joblib"
    meta["joblib_file"] = model_file.name

    try:
        joblib.dump(bundle, model_file)
        joblib.dump(bundle, ACTIVE_CLASSIFIER_PATH)
        ACTIVE_METADATA_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

    registry = list_saved_classifiers()
    registry = [m for m in registry if m.get("model_id") != model_id]
    registry.insert(0, meta)
    registry = registry[:25]
    _MEM_CACHE["registry"] = registry
    try:
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    return meta


def list_saved_classifiers() -> list[dict]:
    ensure_model_dirs()
    if _MEM_CACHE["registry"]:
        return list(_MEM_CACHE["registry"])
    if REGISTRY_PATH.exists():
        try:
            items = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            valid = [
                m for m in items
                if m.get("model_id") in _MEM_CACHE["bundles_by_id"]
                or (CLASSIFICATION_DIR / m.get("joblib_file", "")).exists()
                or ACTIVE_CLASSIFIER_PATH.exists()
            ]
            if valid:
                return valid
        except Exception:
            pass
    if ACTIVE_METADATA_PATH.exists() and ACTIVE_CLASSIFIER_PATH.exists():
        try:
            return [json.loads(ACTIVE_METADATA_PATH.read_text(encoding="utf-8"))]
        except Exception:
            return []
    return []


def load_classifier_bundle(model_id: str | None = None) -> dict | None:
    """Load a specific saved classification bundle by model_id, or the active classifier bundle."""
    ensure_model_dirs()
    if model_id and model_id in _MEM_CACHE["bundles_by_id"]:
        return _MEM_CACHE["bundles_by_id"][model_id]
    if model_id:
        candidate = CLASSIFICATION_DIR / f"{model_id}.joblib"
        if candidate.exists():
            try:
                b = joblib.load(candidate)
                _MEM_CACHE["bundles_by_id"][model_id] = b
                return b
            except Exception:
                pass
    if _MEM_CACHE["active_bundle"] is not None:
        return _MEM_CACHE["active_bundle"]
    if ACTIVE_CLASSIFIER_PATH.exists():
        try:
            b = joblib.load(ACTIVE_CLASSIFIER_PATH)
            _MEM_CACHE["active_bundle"] = b
            return b
        except Exception:
            return None
    return None


def save_clustering_bundle(bundle: dict) -> str:
    """Persist a fitted K-Means clustering bundle to models/clustering/."""
    ensure_model_dirs()
    _MEM_CACHE["clustering_bundle"] = bundle
    out_file = CLUSTERING_DIR / "active_kmeans.joblib"
    try:
        joblib.dump(bundle, out_file)
        meta = {k: v for k, v in bundle.items() if k not in ("preprocessor", "kmeans", "pca", "assignments_df")}
        (CLUSTERING_DIR / "active_kmeans_metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        pass
    return str(out_file)
