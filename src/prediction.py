"""Prediction with a SAVED pipeline + local prediction history (CSV)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from config import settings

HISTORY_COLUMNS = ["prediction_id", "timestamp", "experiment_id", "model", "model_version", "predicted_outcome",
                   "purchase_probability", "threshold", "input_summary"]


# =============================================================================
# 1. INPUT VALIDATION & DERIVED PRE-SNAPSHOT FEATURES
#    - Checks logical consistency (e.g., views + carts <= events so far,
#      unique items <= total views, valid first_cart_position).
#    - Computes derived ratios (avg interval, cart ratio, revisit ratio) using
#      the exact same formulas as src/feature_engineering.py.
# =============================================================================
def derive_features(raw: dict, meta: dict) -> tuple[dict, list[str], list[str]]:
    """Turn user-entered session facts into the saved model's feature vector.

    Returns (features, warnings, errors). Validates internal consistency (e.g. views+carts <= events)
    instead of silently accepting impossible sessions."""
    errs, warns = [], []
    n = int(raw.get("total_events", 0))
    views = int(raw.get("total_views", 0))
    carts = int(raw.get("add_to_cart_count", 0))
    uniq = int(raw.get("unique_items_viewed", 0))
    dur = float(raw.get("session_duration_sec", 0.0))
    fcp = int(raw.get("first_cart_position", -1))
    if n < 1:
        errs.append("Number of events so far must be at least 1.")
    if views + carts > n:
        errs.append(f"Views ({views}) + add-to-cart ({carts}) cannot exceed events so far ({n}). "
                    "Transactions are not allowed before the prediction snapshot.")
    if uniq > views:
        errs.append("Unique products viewed cannot exceed total views.")
    if views > 0 and uniq < 1:
        errs.append("At least one unique product must be viewed when views > 0.")
    if n == 1 and dur > 0:
        errs.append("A session with a single event has zero duration.")
    if carts == 0 and fcp != -1:
        errs.append("First cart position must be -1 when there are no add-to-cart events.")
    if carts > 0 and not (0 <= fcp < n):
        errs.append("First add-to-cart position must be between 0 and events-1.")
    if raw.get("ends_with_cart") and carts == 0:
        errs.append("'Last event was add-to-cart' requires at least one add-to-cart.")
    sp = meta.get("snapshot_params", {})
    if sp.get("policy") == "n_events" and n != sp.get("n_events"):
        warns.append(f"The model was trained on snapshots taken after exactly {sp.get('n_events')} events; "
                     f"entering {n} is outside its training distribution.")
    feats = {
        "total_events": n, "total_views": views, "unique_items_viewed": uniq, "add_to_cart_count": carts,
        "session_duration_sec": dur,
        "avg_time_between_events_sec": dur / (n - 1) if n > 1 else 0.0,
        "cart_interaction_ratio": carts / n if n else 0.0,
        "item_revisit_ratio": (1.0 - uniq / views) if views > 0 else 0.0,
        "first_cart_position": fcp, "ends_with_cart": int(bool(raw.get("ends_with_cart", False))),
        "start_hour": int(raw.get("start_hour", 12)), "start_dayofweek": int(raw.get("start_dayofweek", 0)),
        "unique_categories_viewed": int(raw.get("unique_categories_viewed", 0)),
        "hist_prior_sessions": int(raw.get("hist_prior_sessions", 0)),
        "hist_prior_purchases": int(raw.get("hist_prior_purchases", 0)),
        "hist_hours_since_prev_session": raw.get("hist_hours_since_prev_session", np.nan),
    }
    return feats, warns, errs


# =============================================================================
# 2. SAVED-PIPELINE INFERENCE & LOCAL PERTURBATION EXPLANATION
#    - Runs predict_proba() through the exact saved Scikit-learn pipeline.
#    - Explains the prediction by measuring how probability shifts when each
#      feature is replaced by its training-window median ("what-if" effect).
# =============================================================================
def predict_one(pipeline, meta: dict, feats: dict) -> dict:
    """Use the exact saved pipeline. Local explanation = probability change when each input is replaced
    by its training-window median (model-agnostic, 'what-if' attribution; not causal)."""
    cols = meta["features"]
    X = pd.DataFrame([{c: feats.get(c, np.nan) for c in cols}])[cols]
    p = float(pipeline.predict_proba(X)[0, 1])
    thr = float(meta.get("threshold", 0.5))
    contrib = []
    for c in cols:
        med = meta.get("feature_medians", {}).get(c)
        if med is None:
            continue
        X2 = X.copy()
        X2[c] = med
        contrib.append({"feature": c, "value": float(X.iloc[0][c]) if pd.notna(X.iloc[0][c]) else None,
                        "training_median": med, "probability_effect": p - float(pipeline.predict_proba(X2)[0, 1])})
    contrib.sort(key=lambda r: abs(r["probability_effect"]), reverse=True)
    return {"probability": p, "threshold": thr, "predicted_class": int(p >= thr),
            "label": "Purchase" if p >= thr else "No purchase", "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": meta["model_name"], "experiment_id": meta["experiment_id"], "explanation": contrib,
            "inputs": {c: (None if (isinstance(v, float) and np.isnan(v)) else v) for c, v in X.iloc[0].to_dict().items()}}


# =============================================================================
# 3. LOCAL PREDICTION HISTORY LOG (data/history/prediction_history.csv)
#    - Kept strictly separate from real dataset transactions so model estimates
#      never contaminate analytics or training data.
# =============================================================================
def append_history(result: dict, meta: dict) -> str:
    settings.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    pid = uuid.uuid4().hex[:12]
    row = {"prediction_id": pid, "timestamp": result["timestamp"], "experiment_id": result["experiment_id"],
           "model": result["model"], "model_version": f"{meta['experiment_id']}|data:{meta['dataset_version']}",
           "predicted_outcome": result["label"], "purchase_probability": round(result["probability"], 6),
           "threshold": result["threshold"], "input_summary": json.dumps(result["inputs"], default=str)}
    f = settings.PREDICTION_HISTORY_FILE
    pd.DataFrame([row], columns=HISTORY_COLUMNS).to_csv(f, mode="a", header=not f.exists(), index=False)
    return pid


def read_history() -> pd.DataFrame:
    f = settings.PREDICTION_HISTORY_FILE
    if not f.exists() or f.stat().st_size == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(f, dtype={"prediction_id": "string"})
    return df.sort_values("timestamp", ascending=False, kind="stable").reset_index(drop=True)


def clear_history() -> None:
    settings.PREDICTION_HISTORY_FILE.unlink(missing_ok=True)
