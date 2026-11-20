"""ML tests: pipeline correctness, no-leakage-in-preprocessing, persistence, clustering, prediction."""
import numpy as np
import pandas as pd
import pytest

from src import classification as clf, clustering, feature_engineering as fe, model_manager as mm, prediction
from config import settings


def synthetic_snap(n=3000, seed=0):
    """Pure unit-test fixture (never shown in the app): checks pipeline mechanics only."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({f: rng.rand(n) for f in fe.SNAPSHOT_BASE_FEATURES})
    df["target"] = (df["add_to_cart_count"] + 0.3 * rng.randn(n) > 0.8).astype(int)
    df["snapshot_ts"] = np.arange(n)
    df["visitorid"] = np.arange(n)
    return df


@pytest.mark.parametrize("name", clf.MODEL_NAMES)
def test_train_each_model(name):
    df = synthetic_snap()
    calls = []
    cfg = clf.TrainConfig(model_name=name, cv_folds=2, hyperparams={"n_estimators": 20} if "Forest" in name or "Boosting" in name else {})
    res = clf.train_model(df, cfg, progress=lambda f, m: calls.append(f))
    assert calls and calls[-1] == 1.0 and calls == sorted(calls)
    m = res.metrics
    assert 0 <= m["accuracy"] <= 1 and 0.5 < m["roc_auc"] <= 1 and 0 <= m["pr_auc"] <= 1
    assert np.array(m["confusion_matrix"]).sum() == res.split["test_samples"]
    # chronological split: every train sample precedes every test sample
    assert res.split["train_end_ts"] < res.split["test_start_ts"]


def test_sampling_only_touches_training_rows():
    df = synthetic_snap()
    cfg = clf.TrainConfig(model_name="Logistic Regression", sampling="random_undersample", sampling_ratio=1.0, cv_folds=2)
    res = clf.train_model(df, cfg)
    assert res.split["samples_fitted_after_sampling"] < res.split["train_samples"]
    assert res.split["test_samples"] == int(len(df) * 0.2)  # test set untouched


def test_scaler_fitted_on_train_only():
    df = synthetic_snap()
    res = clf.train_model(df, clf.TrainConfig(cv_folds=0))
    scaler = res.pipeline.estimator_.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    train, _ = clf.chronological_split(df, 0.2)
    np.testing.assert_allclose(scaler.mean_, train[res.config.features].mean().to_numpy(), rtol=1e-6)


def test_leaky_feature_rejected():
    with pytest.raises(ValueError):
        clf.build_pipeline(clf.TrainConfig(features=["total_views", "transaction_count"]))


def test_save_load_roundtrip_and_predict(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EXPERIMENTS_DIR", tmp_path / "exp")
    monkeypatch.setattr(settings, "CLASSIFICATION_MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "HISTORY_DIR", tmp_path / "hist")
    monkeypatch.setattr(settings, "PREDICTION_HISTORY_FILE", tmp_path / "hist" / "h.csv")
    monkeypatch.setattr(mm, "_ACTIVE", tmp_path / "models" / "active.json")
    df = synthetic_snap()
    res = clf.train_model(df, clf.TrainConfig(cv_folds=0))
    meta = mm.save_experiment(res, {"policy": "time"}, {}, "v1", 30.0)
    loaded = mm.load_pipeline(meta["experiment_id"])
    X = df[res.config.features].tail(50)
    np.testing.assert_allclose(res.pipeline.predict_proba(X), loaded.predict_proba(X))
    assert mm.get_active_experiment_id() == meta["experiment_id"]
    feats, w, e = prediction.derive_features({"total_events": 5, "total_views": 4, "unique_items_viewed": 3,
                                              "add_to_cart_count": 1, "session_duration_sec": 100,
                                              "first_cart_position": 2}, meta)
    assert not e
    out = prediction.predict_one(loaded, meta, feats)
    assert 0 <= out["probability"] <= 1
    pid = prediction.append_history(out, meta)
    h = prediction.read_history()
    assert len(h) == 1 and h.loc[0, "prediction_id"] == pid
    prediction.clear_history()
    assert prediction.read_history().empty


def test_prediction_validation_errors():
    meta = {"snapshot_params": {"policy": "n_events", "n_events": 3}}
    _, w, e = prediction.derive_features({"total_events": 3, "total_views": 3, "add_to_cart_count": 2,
                                          "unique_items_viewed": 5, "first_cart_position": -1}, meta)
    assert len(e) >= 2


def test_kmeans_profile_names_from_data():
    rng = np.random.RandomState(0)
    a = pd.DataFrame({"x": rng.normal(0, 1, 500) + 10, "y": rng.normal(0, 1, 500)})
    b = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500) + 10})
    df = pd.concat([a, b], ignore_index=True)
    df["has_purchase"] = 0
    res = clustering.fit_kmeans(df, ["x", "y"], 2, seed=1, scaler="standard")
    assert sorted(res.profile["size"]) == [500, 500]
    assert all("x" in n or "y" in n for n in res.names.values())
    assert res.silhouette > 0.5 and len(res.labels) == 1000
    rng_df = clustering.k_range_analysis(df, ["x", "y"], [], "standard", 2, 4, 1)
    assert list(rng_df.k) == [2, 3, 4] and rng_df.inertia.is_monotonic_decreasing
    with pytest.raises(ValueError):
        clustering.fit_kmeans(df.assign(visitorid=1), ["visitorid"], 2)
