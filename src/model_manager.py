"""Persistence of trained classifiers (joblib) and experiment metadata (JSON), plus clustering bundles."""
from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from config import settings
from src.utilities import atomic_write_json, read_json


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# =============================================================================
# 1. CLASSIFICATION EXPERIMENT PERSISTENCE (joblib pipeline + JSON metadata)
#    - Saves the entire fitted pipeline (imputer + scaler + sampler + model)
#      together with features, cutoff policy, dataset version, and metrics.
# =============================================================================
def save_experiment(result, snapshot_params: dict, snapshot_report: dict, dataset_version: str,
                    session_timeout_minutes: float, note: str = "") -> dict:
    """Persist the fully fitted pipeline + everything needed to reproduce/interpret it."""
    now = datetime.now()
    exp_id = f"{now:%Y%m%d_%H%M%S}_{_slug(result.config.model_name)}"
    model_dir = settings.CLASSIFICATION_MODEL_DIR / exp_id
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, model_dir / "pipeline.joblib", compress=3)
    from src.classification import config_to_dict
    meta = {
        "experiment_id": exp_id, "created_at": now.isoformat(timespec="seconds"),
        "model_name": result.config.model_name, "config": config_to_dict(result.config),
        "features": list(result.config.features), "threshold": result.config.threshold,
        "snapshot_params": snapshot_params, "snapshot_report": snapshot_report,
        "dataset_version": dataset_version, "session_timeout_minutes": session_timeout_minutes,
        "split": result.split, "metrics": result.metrics, "cv": result.cv.to_dict("records"),
        "importance": result.importance, "feature_medians": result.feature_medians,
        "train_seconds": round(result.train_seconds, 2), "note": note,
        "pipeline_file": f"{exp_id}/pipeline.joblib",  # relative to models/classification/
        "library_versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                             "python": sys.version.split()[0]},
    }
    atomic_write_json(settings.EXPERIMENTS_DIR / f"{exp_id}.json", meta)
    return meta


def list_experiments() -> list[dict]:
    settings.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(settings.EXPERIMENTS_DIR.glob("*.json")):
        try:
            out.append(read_json(p))
        except Exception:
            continue  # skip corrupt files rather than crash the UI
    return sorted(out, key=lambda m: m["created_at"], reverse=True)


def load_experiment(exp_id: str) -> dict:
    return read_json(settings.EXPERIMENTS_DIR / f"{exp_id}.json")


def load_pipeline(exp_id: str):
    meta = load_experiment(exp_id)
    # Import here so unpickling can find ResamplingClassifier.
    import src.classification  # noqa: F401
    return joblib.load(settings.CLASSIFICATION_MODEL_DIR / meta["pipeline_file"])


def delete_experiment(exp_id: str) -> None:
    (settings.EXPERIMENTS_DIR / f"{exp_id}.json").unlink(missing_ok=True)
    shutil.rmtree(settings.CLASSIFICATION_MODEL_DIR / exp_id, ignore_errors=True)


def experiments_table(exps: list[dict]) -> pd.DataFrame:
    rows = []
    for m in exps:
        mt = m["metrics"]
        rows.append({"experiment_id": m["experiment_id"], "created_at": m["created_at"], "model": m["model_name"],
                     "accuracy": mt["accuracy"], "precision": mt["precision"], "recall": mt["recall"],
                     "f1": mt["f1"], "roc_auc": mt["roc_auc"], "pr_auc": mt["pr_auc"],
                     "train_n": m["split"]["train_samples"], "test_n": m["split"]["test_samples"],
                     "n_features": len(m["features"]), "balanced": m["config"].get("balanced"),
                     "sampling": m["config"].get("sampling"), "threshold": m["threshold"],
                     "dataset_version": m["dataset_version"]})
    return pd.DataFrame(rows)


# =============================================================================
# 2. ACTIVE MODEL POINTER (models/classification/active.json)
#    - Tracks which saved classification experiment is selected for inference
#      on the Purchase Prediction page and shown in the sidebar status badge.
# =============================================================================
_ACTIVE = settings.CLASSIFICATION_MODEL_DIR / "active.json"


def get_active_experiment_id() -> str | None:
    exps = list_experiments()
    ids = {e["experiment_id"] for e in exps}
    if _ACTIVE.exists():
        try:
            aid = read_json(_ACTIVE).get("experiment_id")
            if aid in ids:
                return aid
        except Exception:
            pass
    return exps[0]["experiment_id"] if exps else None


def set_active_experiment(exp_id: str) -> None:
    atomic_write_json(_ACTIVE, {"experiment_id": exp_id})


# =============================================================================
# 3. CLUSTERING MODEL BUNDLES (models/customer_clustering & session_clustering)
#    - Persists fitted K-Means + preprocessing transformer + cluster profiles.
# =============================================================================
def save_cluster_bundle(kind: str, bundle: dict, meta: dict) -> Path:
    d = settings.CUSTOMER_CLUSTER_DIR if kind == "customer" else settings.SESSION_CLUSTER_DIR
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = d / f"{kind}_kmeans_{stamp}.joblib"
    joblib.dump({"bundle": bundle, "meta": meta}, p, compress=3)
    atomic_write_json(p.with_suffix(".json"), meta)
    return p


def load_latest_cluster_bundle(kind: str):
    d = settings.CUSTOMER_CLUSTER_DIR if kind == "customer" else settings.SESSION_CLUSTER_DIR
    files = sorted(d.glob("*.joblib"))
    return joblib.load(files[-1]) if files else None


def list_cluster_bundles(kind: str) -> list[dict]:
    """List saved clustering metadata JSON files (newest first)."""
    d = settings.CUSTOMER_CLUSTER_DIR if kind == "customer" else settings.SESSION_CLUSTER_DIR
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json"), reverse=True):
        try:
            m = read_json(p)
            m["file"] = p.with_suffix(".joblib").name
            out.append(m)
        except Exception:
            continue
    return out
