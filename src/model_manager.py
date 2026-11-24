"""
Model Persistence Manager using Joblib (Section 6 & Section 11):
  - Saves complete fitted classification pipelines, preprocessing objects, model name,
    selected features, training configuration, dataset information, evaluation metrics,
    and training timestamp to `models/classification/`.
  - Saves fitted K-Means clustering bundles to `models/clustering/`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib

from src import data_loader

MODELS_DIR = data_loader.ROOT_DIR / "models"
CLASSIFICATION_DIR = MODELS_DIR / "classification"
CLUSTERING_DIR = MODELS_DIR / "clustering"
ACTIVE_CLASSIFIER_PATH = CLASSIFICATION_DIR / "active_classifier.joblib"
ACTIVE_METADATA_PATH = CLASSIFICATION_DIR / "active_metadata.json"
REGISTRY_PATH = CLASSIFICATION_DIR / "models_registry.json"


def ensure_model_dirs() -> None:
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)


def save_classification_bundle(bundle: dict) -> dict:
    """Persist a fitted classification pipeline bundle with Joblib and update the JSON registry."""
    ensure_model_dirs()
    ts = bundle.get("trained_at") or datetime.now().isoformat(timespec="seconds")
    model_id = bundle.get("model_id") or f"{bundle['model_name'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bundle["model_id"] = model_id
    bundle["trained_at"] = ts

    model_file = CLASSIFICATION_DIR / f"{model_id}.joblib"
    joblib.dump(bundle, model_file)
    joblib.dump(bundle, ACTIVE_CLASSIFIER_PATH)

    meta = {k: v for k, v in bundle.items() if k != "pipeline"}
    meta["joblib_file"] = model_file.name
    ACTIVE_METADATA_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    registry = list_saved_classifiers()
    registry = [m for m in registry if m.get("model_id") != model_id]
    registry.insert(0, meta)
    registry = registry[:25]
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    return meta


def list_saved_classifiers() -> list[dict]:
    ensure_model_dirs()
    if not REGISTRY_PATH.exists():
        if ACTIVE_METADATA_PATH.exists():
            try:
                return [json.loads(ACTIVE_METADATA_PATH.read_text(encoding="utf-8"))]
            except Exception:
                return []
        return []
    try:
        items = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        # Filter to entries whose joblib file exists or active_classifier exists
        return [
            m for m in items
            if (CLASSIFICATION_DIR / m.get("joblib_file", "")).exists() or ACTIVE_CLASSIFIER_PATH.exists()
        ]
    except Exception:
        return []


def load_classifier_bundle(model_id: str | None = None) -> dict | None:
    """Load a specific saved classification bundle by model_id, or the active classifier bundle."""
    ensure_model_dirs()
    if model_id:
        candidate = CLASSIFICATION_DIR / f"{model_id}.joblib"
        if candidate.exists():
            try:
                return joblib.load(candidate)
            except Exception:
                pass
    if ACTIVE_CLASSIFIER_PATH.exists():
        try:
            return joblib.load(ACTIVE_CLASSIFIER_PATH)
        except Exception:
            return None
    return None


def save_clustering_bundle(bundle: dict) -> str:
    """Persist a fitted K-Means clustering bundle to models/clustering/."""
    ensure_model_dirs()
    out_file = CLUSTERING_DIR / "active_kmeans.joblib"
    joblib.dump(bundle, out_file)
    meta = {k: v for k, v in bundle.items() if k not in ("preprocessor", "kmeans", "pca", "assignments_df")}
    (CLUSTERING_DIR / "active_kmeans_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return str(out_file)
