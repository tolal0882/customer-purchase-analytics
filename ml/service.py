"""
Service layer: the functions the backend API calls.

Supports:
  - Schema inspection & dataset browsing (clean.csv, customer_profiles.csv, session_profiles.csv, category_tree.csv)
  - Interactive data preprocessing & session timeout rebuilding
  - Interactive classification training (4 algorithms, custom hyperparameters, CV, feature selection,
    chronological/stratified split, class weighting, training-fold sampling, threshold)
  - Experiment comparison & switching active model
  - Purchase prediction with local feature-perturbation explanation
  - Interactive K-Means Customer & Session Segmentation (configurable features, K, scaling, log1p, seed)
  - Persistent SQLite prediction history (data/predictions.db)
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from ml import clustering, config, train
from ml.data import make_dataset

_cache: dict = {"bundle": None}

FIELD_LABELS = {
    "views_count": ("Product Views (in first 3 actions)", "Number of product view events observed in the early session window (0–3)."),
    "unique_items_viewed": ("Unique Products Viewed", "Number of distinct products viewed so far (1–3)."),
    "add_to_cart_count": ("Add-to-Cart Actions (in first 3 actions)", "Number of add-to-cart events observed in the early session window (0–3)."),
    "first_cart_position": ("First Add-to-Cart Position (-1 if none)", "0-based index of the first add-to-cart event (0, 1, 2, or -1 if no cart yet)."),
    "elapsed_seconds": ("Session Elapsed Time (seconds)", "Seconds elapsed between the 1st and 3rd interaction in the session."),
    "ends_with_cart": ("Latest Action Was Add-to-Cart?", "Whether the 3rd interaction was an add-to-cart event."),
    "start_hour": ("Session Start Hour (0–23 UTC)", "Hour of day when the browsing session started."),
    "start_dayofweek": ("Day of Week (0=Mon .. 6=Sun)", "Day of week when the session started."),
    "is_weekend": ("Weekend Session?", "Whether the session took place on Saturday or Sunday."),
    "prior_sessions": ("Customer's Prior Sessions", "Number of earlier browsing sessions recorded for this customer."),
    "prior_purchases": ("Customer's Prior Purchases", "Number of transactions completed by this customer in strictly earlier sessions."),
}


# =============================================================================
# 1. MODEL & DATA CACHING HELPERS
# =============================================================================
def _bundle():
    if _cache["bundle"] is None:
        if not config.MODEL_PATH.exists():
            train.run(verbose=False)
        if config.MODEL_PATH.exists():
            try:
                _cache["bundle"] = joblib.load(config.MODEL_PATH)
            except Exception:
                # Re-train automatically if pickled under a different Python/scikit-learn version
                train.run(verbose=False)
                _cache["bundle"] = joblib.load(config.MODEL_PATH)
    return _cache["bundle"]


def _native(v):
    return v.item() if hasattr(v, "item") else v


def _clean_data() -> pd.DataFrame:
    return train.load_dataset()


# =============================================================================
# 2. SCHEMA, DATASET & PREPROCESSING SERVICES
# =============================================================================
def get_schema() -> dict:
    """Describe the model inputs so the frontend can build the form dynamically."""
    df = _clean_data()
    bundle = _bundle()
    all_cols = train.prepare_xy(df)[2]
    active_cols = bundle["input_columns"] if bundle else all_cols
    fields = []
    for col in all_cols:
        s = df[col].dropna()
        label, help_txt = FIELD_LABELS.get(col, (col.replace("_", " ").capitalize(), ""))
        if pd.api.types.is_numeric_dtype(s):
            is_int = bool((s % 1 == 0).all())
            fields.append({
                "name": col,
                "label": label,
                "help": help_txt,
                "active_in_model": col in active_cols,
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
                "active_in_model": col in active_cols,
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
        "active_input_columns": active_cols,
        "model_trained": bundle is not None,
        "active_model_name": bundle.get("model_name", "Logistic Regression") if bundle else None,
    }


def get_data(limit: int = 100, dataset: str = "modeling") -> dict:
    """Return rows from clean.csv ('modeling'), customer_profiles.csv ('customers'),
    session_profiles.csv ('sessions'), or category_tree.csv ('categories')."""
    if dataset == "customers":
        if not config.CUSTOMER_PROFILES_PATH.exists():
            make_dataset.main()
        df = pd.read_csv(config.CUSTOMER_PROFILES_PATH)
    elif dataset == "sessions":
        if not config.SESSION_PROFILES_PATH.exists():
            make_dataset.main()
        df = pd.read_csv(config.SESSION_PROFILES_PATH)
    elif dataset == "categories" and config.CATEGORY_TREE_PATH.exists():
        df = pd.read_csv(config.CATEGORY_TREE_PATH)
    else:
        df = _clean_data()

    missing_summary = {c: int(v) for c, v in df.isna().sum().items()}
    dtypes_summary = {c: str(t) for c, t in df.dtypes.items()}
    rows = json.loads(df.head(limit).to_json(orient="records", date_format="iso"))
    return {
        "dataset": dataset,
        "total_rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "dtypes": dtypes_summary,
        "missing": missing_summary,
        "rows": rows,
    }


def run_preprocessing(timeout_minutes: float = config.SESSION_TIMEOUT_MINUTES) -> dict:
    """Re-run data cleaning and session construction with a custom session timeout.
    On cloud hosts where the 90 MB raw events.csv is not uploaded, updates the active
    session timeout setting from pre-computed sensitivity tables and refreshes insights_summary.json."""
    to_val = float(timeout_minutes)
    if config.RAW_DATA_PATH.exists():
        make_dataset.main(timeout_minutes=to_val)
        return get_insights()

    # Cloud fallback when only data/processed/* is deployed in Git
    if config.INSIGHTS_SUMMARY_PATH.exists():
        summary = json.loads(config.INSIGHTS_SUMMARY_PATH.read_text(encoding="utf-8"))
        summary["session_timeout_minutes"] = to_val
        sens = summary.get("timeout_sensitivity", [])
        if sens:
            nearest = min(sens, key=lambda r: abs(float(r.get("timeout_minutes", 30)) - to_val))
            summary["total_sessions"] = int(nearest.get("total_sessions", summary.get("total_sessions", 1761675)))
            summary["avg_events_per_session"] = float(nearest.get("mean_events_per_session", summary.get("avg_events_per_session", 1.56)))
            summary["single_event_sessions_pct"] = float(nearest.get("single_event_sessions_pct", summary.get("single_event_sessions_pct", 78.6)))
        config.INSIGHTS_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return get_insights()


# =============================================================================
# 3. INTERACTIVE TRAINING & EXPERIMENT REGISTRY SERVICES
# =============================================================================
def get_metrics() -> dict | None:
    if not config.METRICS_PATH.exists():
        train.run(verbose=False)
    if not config.METRICS_PATH.exists():
        return None
    rep = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    rep["experiments"] = list_experiments()
    return rep


def list_experiments() -> list[dict]:
    if not config.EXPERIMENTS_PATH.exists():
        if config.METRICS_PATH.exists():
            return [json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))]
        return []
    try:
        return json.loads(config.EXPERIMENTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def activate_experiment(experiment_id: str) -> dict:
    """Switch the active classification model to a previously saved experiment."""
    exps = list_experiments()
    match = next((e for e in exps if e.get("experiment_id") == experiment_id), None)
    if not match:
        raise ValueError(f"Experiment '{experiment_id}' not found.")
    exp_pipe = config.MODEL_PATH.parent / f"model_{experiment_id}.joblib"
    if exp_pipe.exists():
        bundle = joblib.load(exp_pipe)
        joblib.dump(bundle, config.MODEL_PATH)
        _cache["bundle"] = bundle
    config.METRICS_PATH.write_text(json.dumps(match, indent=2, default=str), encoding="utf-8")
    train.write_model_card(match)
    return get_metrics()


def train_model(params: dict | None = None) -> dict:
    """Train either the full 4-model benchmark or a custom interactive experiment."""
    p = params or {}
    if not p:
        train.run(verbose=False)
    else:
        train.train_custom(
            model_name=p.get("model_name", "all"),
            test_size=float(p.get("test_size", config.TEST_SIZE)),
            random_state=int(p.get("random_state", config.RANDOM_STATE)),
            cv_folds=int(p.get("cv_folds", config.CV_FOLDS)),
            selected_features=p.get("selected_features"),
            balanced=bool(p.get("balanced", True)),
            sampling=str(p.get("sampling", "none")),
            threshold=float(p.get("threshold", 0.5)),
            split_mode=str(p.get("split_mode", "chronological")),
            hyperparams=p.get("hyperparams"),
            verbose=False,
        )
    _cache["bundle"] = None
    return get_metrics()


# =============================================================================
# 4. PURCHASE PREDICTION & LOCAL FEATURE-PERTURBATION EXPLANATION
# =============================================================================
def make_prediction(features: dict) -> dict:
    """Run inference through the active saved pipeline and compute local feature attribution."""
    bundle = _bundle()
    if bundle is None:
        raise RuntimeError("Model is not trained yet. Call POST /api/train first.")
    cols = bundle["input_columns"]

    # Require every active input column selected for the current model
    missing = [c for c in cols if c not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    defaults = bundle.get("feature_defaults") or {}
    row_dict = {c: features[c] for c in cols}
    X = pd.DataFrame([row_dict])[cols]
    pipe = bundle["pipeline"]
    thr = float(bundle.get("threshold", 0.5))
    model_name = bundle.get("model_name", "Trained Classifier")

    if config.TASK_TYPE == "classification":
        p = float(pipe.predict_proba(X)[0, 1])
        label = config.POSITIVE_CLASS if p >= thr else config.NEGATIVE_CLASS

        # Local feature perturbation explanation (probability shift vs training median/mode)
        explanations = []
        for c in cols:
            if c in defaults:
                X_pert = X.copy()
                X_pert[c] = defaults[c]
                p_base = float(pipe.predict_proba(X_pert)[0, 1])
                explanations.append({
                    "feature": c,
                    "input_value": row_dict[c],
                    "baseline_value": defaults[c],
                    "probability_effect": round(p - p_base, 4),
                })
        explanations.sort(key=lambda r: abs(r["probability_effect"]), reverse=True)

        if p >= 0.70:
            tier = "High Purchase Intent"
            advice = "High conversion likelihood. Keep checkout frictionless and suggest complementary cross-sell items."
        elif p >= thr:
            tier = "Moderate / Deciding Shopper"
            advice = "Customer is actively evaluating items. Display stock availability, customer reviews, or a free-shipping nudge."
        else:
            tier = "Low Immediate Purchase Intent"
            advice = "Browsing-oriented session. Highlight popular items, save viewed products, or offer an exit-intent cart reminder."

        return {
            "prediction": label,
            "probability": round(p, 4),
            "threshold": thr,
            "model_name": model_name,
            "probabilities": {
                config.NEGATIVE_CLASS: round(1 - p, 4),
                config.POSITIVE_CLASS: round(p, 4),
            },
            "intent_tier": tier,
            "business_recommendation": advice,
            "explanation": explanations,
        }
    return {"prediction": round(float(pipe.predict(X)[0]), 4)}


# =============================================================================
# 5. CUSTOMER & SESSION SEGMENTATION + INSIGHTS SERVICES
# =============================================================================
def get_segments(
    n_clusters: int | None = None,
    scope: str = "customer",
    features: list[str] | None = None,
    scaler: str = "standard",
    use_log1p: bool = True,
    random_state: int = config.RANDOM_STATE,
) -> dict:
    """Return or fit K-Means segmentation for 'customer' or 'session' scope."""
    metrics_path = config.SESSION_CLUSTERING_METRICS_PATH if scope == "session" else config.CLUSTERING_METRICS_PATH
    need_refit = not metrics_path.exists()
    if not need_refit:
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            (n_clusters is not None and cached.get("n_clusters") != int(n_clusters))
            or (features and cached.get("features") != features)
            or (scaler and cached.get("scaler", "standard") != scaler)
            or (cached.get("use_log1p", True) != bool(use_log1p))
            or (cached.get("random_state", config.RANDOM_STATE) != int(random_state))
        ):
            need_refit = True
        else:
            return cached
    return clustering.run(
        n_clusters=int(n_clusters or config.DEFAULT_K_CLUSTERS),
        scope=scope,
        features=features,
        scaler_name=scaler,
        use_log1p=use_log1p,
        random_state=int(random_state),
        verbose=False,
    )


def predict_customer_segment(customer_features: dict) -> dict:
    return clustering.predict_segment(customer_features)


def get_insights() -> dict:
    if not config.INSIGHTS_SUMMARY_PATH.exists():
        make_dataset.main()
    summary = json.loads(config.INSIGHTS_SUMMARY_PATH.read_text(encoding="utf-8"))
    metrics = get_metrics() or {}
    segments = get_segments(scope="customer") or {}
    return {
        **summary,
        "best_model": metrics.get("best_model"),
        "test_metrics": metrics.get("test_metrics", {}),
        "feature_importance": metrics.get("feature_importance", {}),
        "segments_overview": segments.get("profiles", []),
        "clustering_silhouette": segments.get("silhouette_score"),
    }


# =============================================================================
# 6. PERSISTENT SQLITE PREDICTION HISTORY (data/predictions.db or /tmp fallback)
# =============================================================================
def _db_path() -> Path:
    p = config.PREDICTIONS_DB_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(p) as conn:
            conn.execute("SELECT 1")
        return p
    except Exception:
        return Path("/tmp/predictions.db")


def _init_db() -> None:
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_name TEXT,
                prediction TEXT NOT NULL,
                probability REAL,
                intent_tier TEXT,
                input_json TEXT NOT NULL,
                probabilities_json TEXT
            )
            """
        )


def log_prediction(features: dict, result: dict) -> dict:
    _init_db()
    created_at = datetime.now().isoformat(timespec="seconds")
    probs = result.get("probabilities") or {}
    prob_pos = result.get("probability", probs.get(config.POSITIVE_CLASS))
    model_name = result.get("model_name", "Classifier")
    with sqlite3.connect(_db_path()) as conn:
        # Ensure model_name column exists if upgrading from an older table schema
        cols = [r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()]
        if "model_name" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN model_name TEXT")
        cur = conn.execute(
            """
            INSERT INTO predictions (created_at, model_name, prediction, probability, intent_tier, input_json, probabilities_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                model_name,
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
        "model_name": model_name,
        "input": features,
        "prediction": str(result["prediction"]),
        "probability": prob_pos,
        "intent_tier": result.get("intent_tier", ""),
        "probabilities": probs,
    }


def get_predictions(limit: int = 50) -> list[dict]:
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
            "model_name": r["model_name"] if "model_name" in r.keys() and r["model_name"] else "Classifier",
            "prediction": r["prediction"],
            "probability": r["probability"],
            "intent_tier": r["intent_tier"],
            "input": json.loads(r["input_json"]),
            "probabilities": json.loads(r["probabilities_json"]) if r["probabilities_json"] else {},
        }
        for r in rows
    ]


def clear_predictions() -> int:
    _init_db()
    with sqlite3.connect(_db_path()) as conn:
        cur = conn.execute("DELETE FROM predictions")
        return cur.rowcount
