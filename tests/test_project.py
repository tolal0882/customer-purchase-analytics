"""
Automated Test Suite for:
"Online Shopping Purchase Intention Prediction and Session Behavior Segmentation Using Machine Learning"
Dataset: UCI Online Shoppers Purchasing Intention Dataset (data/raw/online_shoppers_intention.csv)

Verifies all 25 Acceptance Criteria from Section 14:
  - Raw dataset loading, dimensions (12,330 x 18), schema, missing values (0), duplicates (125), Revenue distribution
  - Explicit cleaning options without modifying raw file
  - Separate Scikit-learn ColumnTransformer pipelines for classification and clustering
  - Logistic Regression, Decision Tree, and Random Forest training, CV, evaluation metrics, Joblib persistence
  - PageValues ablation comparison (17 features vs 16 features excluding PageValues)
  - Single-session purchase prediction and local prediction history
  - K-Means session segmentation, Revenue exclusion enforcement, Elbow, Silhouette, 2-D PCA, data-driven profiles
  - Headless rendering of streamlit_app.py and all 6 application pages via Streamlit AppTest
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import (
    classification,
    clustering,
    data_loader,
    data_validation,
    model_manager,
    prediction,
    preprocessing,
)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return data_loader.load_raw_csv()


# =============================================================================
# 1. DATASET LOADING & VALIDATION TESTS
# =============================================================================
def test_uci_dataset_loads_and_matches_schema(raw_df):
    assert raw_df.shape == (12330, 18)
    val = data_validation.validate_dataset(raw_df)
    assert val["is_valid_schema"] is True
    assert val["n_input_features"] == 17
    assert val["n_missing_total"] == 0
    assert val["n_duplicates"] == 125
    td = val["target_distribution"]
    assert td["purchase_count"] == 1908
    assert td["no_purchase_count"] == 10422
    assert td["purchase_rate_pct"] == pytest.approx(15.47, abs=0.05)


def test_explicit_cleaning_preserves_raw_dataset(raw_df):
    before_copy = raw_df.copy()
    df_keep, sum_keep = preprocessing.apply_cleaning_options(raw_df, drop_duplicates=False)
    assert len(df_keep) == 12330
    assert sum_keep["duplicates_removed"] == 0

    df_drop, sum_drop = preprocessing.apply_cleaning_options(raw_df, drop_duplicates=True)
    assert len(df_drop) == 12205
    assert sum_drop["duplicates_removed"] == 125
    assert sum_drop["duplicates_remaining"] == 0
    pd.testing.assert_frame_equal(raw_df, before_copy)


def test_separate_classification_and_clustering_preprocessors(raw_df):
    # Revenue must be rejected if passed to either preprocessor
    with pytest.raises(ValueError):
        preprocessing.build_classification_preprocessor(data_validation.EXPECTED_COLUMNS)
    with pytest.raises(ValueError):
        preprocessing.build_clustering_preprocessor(data_validation.DEFAULT_CLUSTERING_FEATURES + ["Revenue"])

    X_cls = preprocessing.prepare_features_for_transform(raw_df.head(200), data_validation.INPUT_FEATURES)
    cls_pre = preprocessing.build_classification_preprocessor(data_validation.INPUT_FEATURES, scaler_name="standard")
    Xt_cls = cls_pre.fit_transform(X_cls)
    assert Xt_cls.shape[0] == 200 and Xt_cls.shape[1] >= 17

    X_clu = preprocessing.prepare_features_for_transform(raw_df.head(200), data_validation.DEFAULT_CLUSTERING_FEATURES)
    clu_pre = preprocessing.build_clustering_preprocessor(data_validation.DEFAULT_CLUSTERING_FEATURES, scaler_name="standard")
    Xt_clu = clu_pre.fit_transform(X_clu)
    assert Xt_clu.shape == (200, len(data_validation.DEFAULT_CLUSTERING_FEATURES))


# =============================================================================
# 2. SUPERVISED CLASSIFICATION, PAGEVALUES STUDY & PREDICTION TESTS
# =============================================================================
@pytest.fixture(scope="session")
def trained_classification_results(raw_df):
    return classification.train_and_evaluate_models(
        raw_df,
        selected_algorithms=classification.ALGORITHMS,
        selected_features=data_validation.INPUT_FEATURES,
        test_size=0.20,
        random_state=42,
        cv_folds=5,
        use_class_weight=True,
        persist_best=True,
    )


def test_all_three_classifiers_train_and_persist(trained_classification_results):
    res = trained_classification_results
    assert set(res["models"].keys()) == set(classification.ALGORITHMS)
    assert len(res["comparison_table"]) == 3

    for algo, detail in res["models"].items():
        tm = detail["test_metrics"]
        assert 0.50 < tm["roc_auc"] <= 1.0
        assert 0.30 < tm["f1"] <= 1.0
        assert 0.30 < tm["recall"] <= 1.0
        assert 0.20 < tm["precision"] <= 1.0
        assert "confusion_matrix" in detail and "classification_report" in detail

    bundle = model_manager.load_classifier_bundle()
    assert bundle is not None
    assert "pipeline" in bundle and "selected_features" in bundle


def test_pagevalues_investigation_compares_17_vs_16_features(raw_df):
    study = classification.run_pagevalues_investigation(
        raw_df,
        algorithm_name="Logistic Regression",
        test_size=0.20,
        random_state=42,
    )
    assert len(study["comparison_rows"]) == 6
    f1_row = next(r for r in study["comparison_rows"] if r["Metric"] == "F1-Score")
    assert f1_row["With PageValues (17 Features)"] > f1_row["Without PageValues (16 Features)"]


def test_session_prediction_and_local_history(trained_classification_results):
    bundle = model_manager.load_classifier_bundle()
    sample_inputs = dict(bundle["feature_defaults"])
    sample_inputs["ProductRelated"] = 35
    sample_inputs["ProductRelated_Duration"] = 1400.0
    sample_inputs["PageValues"] = 25.0

    pred = prediction.predict_session(sample_inputs)
    assert pred["predicted_class"] in (0, 1)
    assert 0.0 <= pred["purchase_probability"] <= 1.0
    assert pred["purchase_probability"] + pred["no_purchase_probability"] == pytest.approx(1.0, abs=1e-3)

    hist = prediction.load_prediction_history()
    assert len(hist) >= 1
    assert hist[0]["record_type"] == "Model Prediction (Not Actual Transaction)"


# =============================================================================
# 3. UNSUPERVISED K-MEANS SESSION SEGMENTATION TESTS
# =============================================================================
def test_session_segmentation_excludes_revenue_and_builds_profiles(raw_df):
    with pytest.raises(ValueError):
        clustering.run_session_segmentation(
            raw_df,
            selected_features=data_validation.DEFAULT_CLUSTERING_FEATURES + ["Revenue"],
            n_clusters=4,
        )

    seg = clustering.run_session_segmentation(
        raw_df,
        selected_features=data_validation.DEFAULT_CLUSTERING_FEATURES,
        n_clusters=4,
        k_min=2,
        k_max=6,
        random_state=42,
    )
    assert seg["revenue_excluded_from_inputs"] is True
    assert seg["n_clusters"] == 4
    assert seg["n_sessions"] == 12330
    assert seg["silhouette_score"] > 0.15
    assert len(seg["k_evaluation"]) == 5  # K = 2..6
    assert len(seg["profiles"]) == 4
    assert len(seg["pca_explained_variance_pct"]) == 2
    assert "Session_Segment" in seg["assignments_df"].columns


# =============================================================================
# 4. ALL 6 STREAMLIT APPLICATION PAGES RENDER TEST
# =============================================================================
def test_all_6_streamlit_pages_render_without_error(trained_classification_results):
    from streamlit.testing.v1 import AppTest

    root_app = AppTest.from_file(str(data_loader.ROOT_DIR / "streamlit_app.py"), default_timeout=90)
    root_app.run()
    assert not root_app.exception, f"streamlit_app.py raised exception: {root_app.exception}"

    page_files = [
        "dashboard.py",
        "dataset_eda.py",
        "preprocessing.py",
        "purchase_prediction.py",
        "session_segmentation.py",
        "about.py",
    ]
    for pf in page_files:
        at = AppTest.from_file(str(data_loader.ROOT_DIR / "app_pages" / pf), default_timeout=90)
        at.run()
        assert not at.exception, f"app_pages/{pf} raised exception: {at.exception}"
