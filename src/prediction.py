"""
Prediction & Local Prediction History Module (Section 6 & Page 4):
  - Validates all required session input features
  - Runs inference through the loaded compatible Joblib classification pipeline
  - Computes local feature-perturbation effects relative to training median/mode
  - Stores prediction history locally in `data/processed/prediction_history.json`
    and supports CSV export, clearly distinguishing model predictions from actual purchases.
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from src import data_loader, data_validation, model_manager, preprocessing

HISTORY_PATH = data_loader.PROCESSED_DIR / "prediction_history.json"


def validate_prediction_inputs(input_values: dict, required_features: list[str]) -> list[str]:
    """Validate user inputs before running prediction."""
    errors = []
    missing = [f for f in required_features if f not in input_values]
    if missing:
        errors.append(f"Missing required model input features: {missing}")
        return errors

    for col in [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
    ]:
        if col in input_values:
            val = float(input_values[col])
            if val < 0:
                errors.append(f"'{col}' cannot be negative (got {val}).")
    for rate_col in ("BounceRates", "ExitRates"):
        if rate_col in input_values:
            val = float(input_values[rate_col])
            if not (0.0 <= val <= 1.0):
                errors.append(f"'{rate_col}' must be between 0.0 and 1.0 (got {val}).")
    if "SpecialDay" in input_values:
        val = float(input_values["SpecialDay"])
        if not (0.0 <= val <= 1.0):
            errors.append(f"'SpecialDay' must be between 0.0 and 1.0 (got {val}).")
    return errors


def predict_session(input_values: dict, model_id: str | None = None) -> dict:
    """Run purchase intention prediction on a single session using a saved compatible pipeline."""
    bundle = model_manager.load_classifier_bundle(model_id)
    if bundle is None:
        raise RuntimeError("No trained classification model found. Please train a model first.")

    req_features = bundle["selected_features"]
    errors = validate_prediction_inputs(input_values, req_features)
    if errors:
        raise ValueError("; ".join(errors))

    raw_df = pd.DataFrame([{f: input_values[f] for f in req_features}])
    X_in = preprocessing.prepare_features_for_transform(raw_df, req_features)

    pipe = bundle["pipeline"]
    pred_class = int(pipe.predict(X_in)[0])
    prob_purchase = float(pipe.predict_proba(X_in)[0, 1]) if hasattr(pipe, "predict_proba") else float(pred_class)
    timestamp = datetime.now().isoformat(timespec="seconds")

    # Local feature-perturbation explanation against training defaults
    defaults = bundle.get("feature_defaults", {})
    effects = []
    if hasattr(pipe, "predict_proba") and defaults:
        for f in req_features:
            if f in defaults:
                pert_df = raw_df.copy()
                pert_df[f] = defaults[f]
                X_pert = preprocessing.prepare_features_for_transform(pert_df, req_features)
                p_base = float(pipe.predict_proba(X_pert)[0, 1])
                effects.append({
                    "feature": f,
                    "input_value": input_values[f],
                    "baseline_value": defaults[f],
                    "probability_shift": round(prob_purchase - p_base, 4),
                })
        effects.sort(key=lambda r: abs(r["probability_shift"]), reverse=True)

    result = {
        "timestamp": timestamp,
        "model_id": bundle.get("model_id", "active"),
        "model_name": bundle.get("model_name", "Classifier"),
        "predicted_class": pred_class,
        "predicted_label": "Purchase Intended (Predicted Revenue = True)" if pred_class == 1 else "No Purchase Predicted (Predicted Revenue = False)",
        "purchase_probability": round(prob_purchase, 4),
        "no_purchase_probability": round(1.0 - prob_purchase, 4),
        "includes_pagevalues": "PageValues" in req_features,
        "inputs": {f: input_values[f] for f in req_features},
        "feature_effects": effects[:10],
    }
    append_prediction_history(result)
    return result


def append_prediction_history(record: dict) -> None:
    """Persist prediction record locally in data/processed/prediction_history.json."""
    data_loader.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    history = load_prediction_history()
    entry = {
        "prediction_id": len(history) + 1,
        "timestamp": record["timestamp"],
        "model_name": record["model_name"],
        "model_id": record["model_id"],
        "predicted_class": record["predicted_class"],
        "predicted_label": "Purchase (Estimated)" if record["predicted_class"] == 1 else "No Purchase (Estimated)",
        "purchase_probability_pct": round(record["purchase_probability"] * 100, 2),
        "record_type": "Model Prediction (Not Actual Transaction)",
        **record["inputs"],
    }
    history.insert(0, entry)
    HISTORY_PATH.write_text(json.dumps(history[:250], indent=2, default=str), encoding="utf-8")


def load_prediction_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def clear_prediction_history() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
