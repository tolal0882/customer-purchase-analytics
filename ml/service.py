"""
Service layer: the functions the backend API calls.

Both backends (FastAPI and Django) call these functions so the API behaves
identically. Every function returns plain Python dicts/lists ready for JSON.
Also manages persistent SQLite prediction history in data/predictions.db.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import joblib
import pandas as pd

from ml import clustering, config, train
from ml.data import make_dataset

_cache: dict = {"bundle": None}

# Human-readable labels and help text for the schema-driven prediction form
FIELD_LABELS = {
    "views_count": ("Product Views (in first 3 actions)", "Number of product view events observed in the early session window (0–3)."),
    "unique_items_viewed": ("Unique Products Viewed", "Number of distinct products viewed so far (1–3)."),
    "add_to_cart_count": ("Add-to-Cart Actions (in first 3 actions)", "Number of add-to-cart events observed in the early session window (0–3)."),
    "elapsed_seconds": ("Session Elapsed Time (seconds)", "Seconds elapsed between the 1st and 3rd interaction in the session."),
    "ends_with_cart": ("Latest Action Was Add-to-Cart?", "Whether the 3rd interaction was an add-to-cart event."),
    "start_hour": ("Session Start Hour (0–23 UTC)", "Hour of day when the browsing session started."),
    "is_weekend": ("Weekend Session?", "Whether the session took place on Saturday or Sunday."),
    "prior_sessions": ("Customer's Prior Sessions", "Number of earlier browsing sessions recorded for this customer."),
    "prior_purchases": ("Customer's Prior Purchases", "Number of transactions completed by this customer in strictly earlier sessions."),
}


# =============================================================================
# 1. MODEL & DATA CACHING HELPERS
# =============================================================================
def _bundle():
    if _cache["bundle"] is None and config.MODEL_PATH.exists():
        _cache["bundle"] = joblib.load(config.MODEL_PATH)
    return _cache["bundle"]


def _native(v):
    return v.item() if hasattr(v, "item") else v


def _clean_data() -> pd.DataFrame:
    return train.load_dataset()


# =============================================================================
# 2. SCHEMA, DATASET & METRICS ENDPOINT SERVICES
# =============================================================================
def get_schema() -> dict:
    """Describe the model inputs so the frontend can build the form dynamically."""
    df = _clean_data()
    bundle = _bundle()
    cols = bundle["input_columns"] if bundle else train.prepare_xy(df)[2]
    fields = []
    for col in cols:
        s = df[col].dropna()
        label, help_txt = FIELD_LABELS.get(col, (col.replace("_", " ").capitalize(), ""))
        if pd.api.types.is_numeric_dtype(s):
            is_int = bool((s % 1 == 0).all())
            fields.append({
                "name": col,
                "label": label,
                "help": help_txt,
                "type": "integer" if is_int else "number",
                "default": int(s.median()) if is_int else round(float(s.median()), 2),
                "min": _native(s.min()),
                "max": _native(s.max()),
            })
        else:
            fields.append({
                "name": col,
                "label": label,
                "help": help_txt,
                "type": "category",
                "options": sorted(map(str, s.unique())),
                "default": str(s.mode().iloc[0]),
            })
    return {
        "task_type": config.TASK_TYPE,
        "target": config.TARGET_COLUMN,
        "positive_class": config.POSITIVE_CLASS,
        "negative_class": config.NEGATIVE_CLASS,
        "features": fields,
        "model_trained": bundle is not None,
    }


def get_data(limit: int = 100, dataset: str = "modeling") -> dict:
    """Return rows from clean.csv ('modeling'), customer_profiles.csv ('customers'), or category_tree.csv ('categories')."""
    if dataset == "customers":
        if not config.CUSTOMER_PROFILES_PATH.exists():
            make_dataset.main()
        df = pd.read_csv(config.CUSTOMER_PROFILES_PATH)
    elif dataset == "categories" and config.CATEGORY_TREE_PATH.exists():
        df = pd.read_csv(config.CATEGORY_TREE_PATH)
    else:
        df = _clean_data()
    rows = json.loads(df.head(limit).to_json(orient="records", date_format="iso"))
    return {"dataset": dataset, "total_rows": len(df), "columns": list(df.columns), "rows": rows}


def get_metrics() -> dict | None:
    if not config.METRICS_PATH.exists():
        return None
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))


def train_model() -> dict:
    """Re-run both supervised classification training and K-Means customer segmentation."""
    train.run(verbose=False)
    _cache["bundle"] = None
    return get_metrics()


# =============================================================================
# 3. PURCHASE PREDICTION & EXPLANATION SERVICE
# =============================================================================
def make_prediction(features: dict) -> dict:
    """Run inference through the saved classification pipeline and explain the prediction."""
    bundle = _bundle()
    if bundle is None:
        raise RuntimeError("Model is not trained yet. Call POST /api/train first.")
    cols = bundle["input_columns"]
    missing = [c for c in cols if c not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = pd.DataFrame([{c: features[c] for c in cols}])
    pipe = bundle["pipeline"]
    if config.TASK_TYPE == "classification":
        p = float(pipe.predict_proba(X)[0, 1])
        label = config.POSITIVE_CLASS if p >= 0.5 else config.NEGATIVE_CLASS

        # Generate a concise business recommendation based on purchase probability
        if p >= 0.70:
            tier = "High Purchase Intent"
            advice = "High conversion likelihood. Avoid unnecessary discount pop-ups; keep checkout frictionless and suggest complementary cross-sell items."
        elif p >= 0.45:
            tier = "Moderate / Deciding Shopper"
            advice = "Customer is actively evaluating items. Display stock availability, customer reviews, or a subtle free-shipping nudge to close the sale."
        else:
            tier = "Low Immediate Purchase Intent"
            advice = "Browsing-oriented session. Highlight popular items, save viewed products to recently-viewed lists, or offer an incentive if cart is abandoned."

        return {
            "prediction": label,
            "probability": round(p, 4),
            "probabilities": {
                config.NEGATIVE_CLASS: round(1 - p, 4),
                config.POSITIVE_CLASS: round(p, 4),
            },
            "intent_tier": tier,
            "business_recommendation": advice,
        }
    return {"prediction": round(float(pipe.predict(X)[0]), 4)}


# =============================================================================
# 4. CUSTOMER SEGMENTATION & BUSINESS INSIGHTS SERVICES
# =============================================================================
def get_segments(n_clusters: int | None = None) -> dict:
    """Return K-Means customer segmentation metrics, profiles, and 2-D PCA points."""
    if n_clusters is not None and (
        not config.CLUSTERING_METRICS_PATH.exists()
        or json.loads(config.CLUSTERING_METRICS_PATH.read_text(encoding="utf-8")).get("n_clusters") != int(n_clusters)
    ):
        return clustering.run(n_clusters=int(n_clusters), verbose=False)
    if not config.CLUSTERING_METRICS_PATH.exists():
        return clustering.run(verbose=False)
    return json.loads(config.CLUSTERING_METRICS_PATH.read_text(encoding="utf-8"))


def predict_customer_segment(customer_features: dict) -> dict:
    """Assign a customer behavior profile to its nearest K-Means cluster."""
    return clustering.predict_segment(customer_features)


def get_insights() -> dict:
    """Return e-commerce KPIs, conversion funnel, temporal patterns, and model/segment insights."""
    if not config.INSIGHTS_SUMMARY_PATH.exists():
        make_dataset.main()
    summary = json.loads(config.INSIGHTS_SUMMARY_PATH.read_text(encoding="utf-8"))
    metrics = get_metrics() or {}
    segments = get_segments() or {}
    return {
        **summary,
        "best_model": metrics.get("best_model"),
        "test_metrics": metrics.get("test_metrics", {}),
        "feature_importance": metrics.get("feature_importance", {}),
        "segments_overview": segments.get("profiles", []),
        "clustering_silhouette": segments.get("silhouette_score"),
    }


# =============================================================================
# 5. PERSISTENT SQLITE PREDICTION HISTORY (data/predictions.db or /tmp fallback)
# =============================================================================
def _db_path():
    p = config.PREDICTIONS_DB_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(p) as conn:
            conn.execute("SELECT 1")
        return p
    except Exception:
        from pathlib import Path
        return Path("/tmp/predictions.db")


def _init_db() -> None:
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL,
                intent_tier TEXT,
                input_json TEXT NOT NULL,
                probabilities_json TEXT
            )
            """
        )


def log_prediction(features: dict, result: dict) -> dict:
    """Save a prediction record to SQLite and return the saved item."""
    _init_db()
    created_at = datetime.now().isoformat(timespec="seconds")
    probs = result.get("probabilities") or {}
    prob_pos = result.get("probability", probs.get(config.POSITIVE_CLASS))
    with sqlite3.connect(_db_path()) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (created_at, prediction, probability, intent_tier, input_json, probabilities_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                str(result["prediction"]),
                float(prob_pos) if prob_pos is not None else None,
                result.get("intent_tier", ""),
                json.dumps(features),
                json.dumps(probs),
            ),
        )
        row_id = cur.lastrowid
    return {
        "id": row_id,
        "created_at": created_at,
        "input": features,
        "prediction": str(result["prediction"]),
        "probability": prob_pos,
        "intent_tier": result.get("intent_tier", ""),
        "probabilities": probs,
    }


def get_predictions(limit: int = 50) -> list[dict]:
    """Read the most recent prediction records from SQLite."""
    _init_db()
    with sqlite3.connect(_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [
        {
            "id": r["id"],
            "created_at": r["created_at"],
            "prediction": r["prediction"],
            "probability": r["probability"],
            "intent_tier": r["intent_tier"],
            "input": json.loads(r["input_json"]),
            "probabilities": json.loads(r["probabilities_json"]) if r["probabilities_json"] else {},
        }
        for r in rows
    ]


def clear_predictions() -> int:
    """Delete all saved prediction history records from SQLite."""
    _init_db()
    with sqlite3.connect(_db_path()) as conn:
        cur = conn.execute("DELETE FROM predictions")
        return cur.rowcount
