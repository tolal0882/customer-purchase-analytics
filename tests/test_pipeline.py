"""
Automated tests for the Retailrocket Purchase Prediction & Customer Segmentation pipeline.
Run from the project root:

    pytest
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend_fastapi.main import app
from ml import clustering, config, service, train
from ml.data.make_dataset import clean, load_raw
from ml.features import add_features

client = TestClient(app)


# =============================================================================
# 1. FIXTURES
# =============================================================================
@pytest.fixture(scope="session")
def raw_sample():
    """Load a 25,000-row sample of raw events.csv for fast unit testing of clean()."""
    return load_raw(nrows=25_000)


@pytest.fixture(scope="session")
def cleaned():
    """Load the processed clean.csv dataset produced by make_dataset."""
    return train.load_dataset()


@pytest.fixture(scope="session")
def trained_model():
    if not config.MODEL_PATH.exists() or not config.CLUSTERING_MODEL_PATH.exists():
        train.run(verbose=False)
    return config.MODEL_PATH


# =============================================================================
# 2. DATA CLEANING & LEAKAGE PREVENTION TESTS
# =============================================================================
def test_no_duplicates(cleaned):
    assert cleaned.duplicated().sum() == 0


def test_target_has_only_valid_labels(cleaned):
    assert set(cleaned[config.TARGET_COLUMN].unique()) <= {config.POSITIVE_CLASS, config.NEGATIVE_CLASS}


def test_valid_early_session_counts(cleaned):
    assert (cleaned["views_count"] >= 0).all()
    assert (cleaned["add_to_cart_count"] >= 0).all()
    assert (cleaned["unique_items_viewed"] <= cleaned["views_count"] + cleaned["add_to_cart_count"]).all()
    assert cleaned["start_hour"].between(0, 23).all()


def test_clean_does_not_change_raw(raw_sample):
    before = raw_sample.copy()
    clean(raw_sample, verbose=False)
    pd.testing.assert_frame_equal(raw_sample, before)


def test_forbidden_id_columns_excluded_from_inputs(cleaned):
    _, _, input_cols = train.prepare_xy(cleaned)
    for col in config.DROP_COLUMNS:
        assert col not in input_cols


# =============================================================================
# 3. IN-PIPELINE FEATURE ENGINEERING TESTS
# =============================================================================
def test_add_features_creates_columns(cleaned):
    out = add_features(cleaned)
    expected = {"cart_to_view_ratio", "item_focus_ratio", "avg_seconds_per_action", "is_returning_visitor"}
    assert expected <= set(out.columns)
    assert len(out) == len(cleaned)


# =============================================================================
# 4. SUPERVISED PREDICTION & MODEL QUALITY TESTS
# =============================================================================
def _example_input():
    return {f["name"]: f["default"] for f in service.get_schema()["features"]}


def test_prediction_returns_valid_label_and_probabilities(trained_model):
    result = service.make_prediction(_example_input())
    assert result["prediction"] in {config.POSITIVE_CLASS, config.NEGATIVE_CLASS}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert "intent_tier" in result and "business_recommendation" in result
    assert "explanation" in result and len(result["explanation"]) > 0


def test_missing_feature_raises_error(trained_model):
    data = _example_input()
    data.pop(next(iter(data)))
    with pytest.raises(ValueError):
        service.make_prediction(data)


def test_model_beats_minimum_quality(trained_model):
    metrics = service.get_metrics()["test_metrics"]
    assert metrics["roc_auc"] > 0.70


def test_interactive_custom_training_and_experiment_history(trained_model):
    """Verify interactive model training with custom hyperparameters and sampling."""
    rep = service.train_model({
        "model_name": "Decision Tree",
        "split_mode": "chronological",
        "test_size": 0.2,
        "cv_folds": 3,
        "balanced": True,
        "sampling": "undersample",
        "threshold": 0.45,
        "hyperparams": {"max_depth": 5, "min_samples_split": 20},
    })
    assert rep["best_model"] == "Decision Tree"
    assert rep["training_config"]["threshold"] == 0.45
    assert "curves" in rep and "roc" in rep["curves"] and "pr" in rep["curves"]
    exps = service.list_experiments()
    assert len(exps) >= 1
    # Restore full 4-model benchmark as the default active model
    full_rep = service.train_model({})
    assert len(full_rep["cv_results"]) == 4


# =============================================================================
# 5. UNSUPERVISED K-MEANS CUSTOMER & SESSION SEGMENTATION TESTS
# =============================================================================
def test_customer_and_session_segmentation_outputs(trained_model):
    for scope, feat_list in (
        ("customer", config.CUSTOMER_CLUSTERING_FEATURES),
        ("session", config.SESSION_CLUSTERING_FEATURES),
    ):
        seg = service.get_segments(scope=scope)
        assert seg["scope"] == scope
        assert seg["n_clusters"] in config.K_RANGE
        assert seg["silhouette_score"] > 0.20
        assert len(seg["profiles"]) == seg["n_clusters"]
        assert len(seg["k_evaluation"]) == len(config.K_RANGE)

    sample_cust = {f: 2.0 for f in config.CUSTOMER_CLUSTERING_FEATURES}
    assigned = clustering.predict_segment(sample_cust)
    assert 0 <= assigned["cluster_id"] < config.DEFAULT_K_CLUSTERS
    assert assigned["segment_name"]


# =============================================================================
# 6. FASTAPI ENDPOINT & SQLITE HISTORY TESTS
# =============================================================================
def test_fastapi_endpoints_and_sqlite_history(trained_model):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/schema").status_code == 200
    assert client.get("/api/data?limit=10").json()["total_rows"] > 0
    assert client.get("/api/metrics").status_code == 200
    assert client.get("/api/segments").status_code == 200
    assert client.get("/api/insights").status_code == 200

    pred_res = client.post("/api/predict", json={"features": _example_input()})
    assert pred_res.status_code == 200
    hist = client.get("/api/predictions").json()["items"]
    assert len(hist) >= 1


def test_all_8_streamlit_pages_render(trained_model):
    """Verify the Streamlit frontend entrypoint and all 8 views render without errors."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT_DIR / "frontend" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Home page raised exception: {at.exception}"

    for view_name in ("home", "data_explorer", "prediction", "segmentation", "model_performance", "history", "insights", "about"):
        view_path = config.ROOT_DIR / "frontend" / "views" / f"{view_name}.py"
        vt = AppTest.from_file(str(view_path), default_timeout=60)
        vt.run()
        assert not vt.exception, f"View {view_name}.py raised exception: {vt.exception}"
