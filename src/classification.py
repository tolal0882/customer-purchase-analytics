"""
Supervised Machine Learning Module — Task 1: Purchase Intention Prediction (Section 6):
  - Target: Revenue (1 = Purchase, 0 = No Purchase)
  - Algorithms:
      1. Logistic Regression
      2. Decision Tree Classifier
      3. Random Forest Classifier
  - Stratified train/test split (default 80/20, random_state=42)
  - Stratified 5-fold cross-validation on training data
  - Configurable hyperparameters & optional class weighting
  - PageValues Investigation: compares all 17 features vs 16 features (excluding PageValues)
    on the exact same stratified train/test split.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import data_validation, evaluation, model_manager, preprocessing

# =============================================================================
# 1. SUPPORTED CLASSIFICATION ALGORITHMS & SCORING METRICS (SECTION 6)
# =============================================================================
ALGORITHMS = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
]

SCORING_METRICS = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


# =============================================================================
# 2. CLASSIFIER FACTORY FUNCTION
#    Instantiates Logistic Regression, Decision Tree, or Random Forest with
#    user-selected hyperparameters and optional balanced class weights.
# =============================================================================
def build_classifier(
    algorithm_name: str,
    hyperparams: dict | None = None,
    use_class_weight: bool = True,
    random_state: int = 42,
):
    """Instantiate one of the 3 required classification algorithms with configurable hyperparameters."""
    hp = hyperparams or {}
    cw = "balanced" if use_class_weight else None

    if algorithm_name == "Logistic Regression":
        return LogisticRegression(
            C=float(hp.get("C", 1.0)),
            max_iter=int(hp.get("max_iter", 1000)),
            class_weight=cw,
            random_state=int(random_state),
        )
    if algorithm_name == "Decision Tree Classifier":
        return DecisionTreeClassifier(
            max_depth=int(hp.get("max_depth", 6)),
            min_samples_split=int(hp.get("min_samples_split", 20)),
            min_samples_leaf=int(hp.get("min_samples_leaf", 10)),
            criterion=str(hp.get("criterion", "gini")),
            class_weight=cw,
            random_state=int(random_state),
        )
    if algorithm_name == "Random Forest Classifier":
        return RandomForestClassifier(
            n_estimators=int(hp.get("n_estimators", 150)),
            max_depth=int(hp.get("max_depth", 10)),
            min_samples_split=int(hp.get("min_samples_split", 10)),
            min_samples_leaf=int(hp.get("min_samples_leaf", 5)),
            class_weight=cw,
            n_jobs=1,
            random_state=int(random_state),
        )
    raise ValueError(f"Unsupported algorithm: {algorithm_name}")


# =============================================================================
# 3. UNIFIED TRAINING, STRATIFIED K-FOLD CV & TEST EVALUATION WORKFLOW
#    - Splits data using a single Stratified train/test split (default 80/20)
#      so all selected models are compared on the exact same evaluation split.
#    - Runs Stratified 5-fold cross-validation strictly on X_train.
#    - Fits the full Pipeline (ColumnTransformer + Classifier) on X_train,
#      evaluates on X_test, and saves the fitted bundle via Joblib.
# =============================================================================
def train_and_evaluate_models(
    df: pd.DataFrame,
    selected_algorithms: list[str] | None = None,
    selected_features: list[str] | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
    cv_folds: int = 5,
    use_class_weight: bool = True,
    scaler_name: str = "standard",
    use_log1p: bool = False,
    drop_duplicates: bool = False,
    hyperparams_by_algo: dict[str, dict] | None = None,
    primary_metric: str = "f1",
    persist_best: bool = True,
) -> dict:
    """Train 1..3 selected classifiers on the exact same stratified split, run Stratified K-fold CV
    on the training set, evaluate on the held-out test set, and save each fitted pipeline with Joblib."""
    algos = selected_algorithms or list(ALGORITHMS)
    features = [
        f for f in (selected_features or data_validation.INPUT_FEATURES)
        if f in df.columns and f != data_validation.TARGET_COL
    ]
    if not features:
        raise ValueError("Please select at least one input feature for training.")

    # 3.1 Apply user-selected cleaning option and separate X (features) and y (Revenue)
    clean_df, clean_info = preprocessing.apply_cleaning_options(df, drop_duplicates=drop_duplicates)
    X_all = preprocessing.prepare_features_for_transform(clean_df, features)
    y_all = clean_df[data_validation.TARGET_COL].astype(bool).astype(int).to_numpy()

    # 3.2 Stratified Train/Test Split (preserves the ~15.5% Revenue=True ratio in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y_all,
    )

    skf = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=int(random_state))
    hp_map = hyperparams_by_algo or {}
    trained_at = datetime.now().isoformat(timespec="seconds")

    comparison_rows = []
    model_details = {}
    saved_bundles = {}

    # 3.3 Compute training-set medians/modes for prediction form defaults & local explanations
    feature_defaults = {}
    for c in features:
        if c in data_validation.CONTINUOUS_FEATURES:
            feature_defaults[c] = round(float(clean_df[c].median()), 4)
        elif c == "Weekend":
            feature_defaults[c] = bool(clean_df[c].mode().iloc[0])
        elif c in ("Month", "VisitorType"):
            feature_defaults[c] = str(clean_df[c].mode().iloc[0])
        else:
            feature_defaults[c] = int(clean_df[c].mode().iloc[0])

    # 3.4 Train, cross-validate, and evaluate each selected algorithm
    for algo in algos:
        algo_hp = hp_map.get(algo, {})
        pre = preprocessing.build_classification_preprocessor(
            features, scaler_name=scaler_name, use_log1p=use_log1p
        )
        clf = build_classifier(
            algo,
            hyperparams=algo_hp,
            use_class_weight=use_class_weight,
            random_state=int(random_state),
        )
        pipe = Pipeline([
            ("preprocessor", pre),
            ("classifier", clf),
        ])

        cv_res = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=skf,
            scoring=SCORING_METRICS,
            n_jobs=1,
        )

        # Fit strictly on X_train; preprocess X_test through the fitted pipeline
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        test_m = evaluation.compute_classification_metrics(y_test, y_pred, y_proba)
        cm_dict, cls_rep = evaluation.compute_confusion_and_report(y_test, y_pred)
        curves = evaluation.compute_curves(y_test, y_proba)
        imp = evaluation.extract_feature_importance(pipe)

        cv_summary = {
            m: round(float(cv_res[f"test_{m}"].mean()), 4) for m in SCORING_METRICS
        }
        cv_std = {
            m: round(float(cv_res[f"test_{m}"].std()), 4) for m in SCORING_METRICS
        }

        row = {
            "Algorithm": algo,
            "CV F1 (mean±std)": f"{cv_summary['f1']:.4f} ± {cv_std['f1']:.4f}",
            "CV ROC-AUC": cv_summary["roc_auc"],
            "CV PR-AUC": cv_summary["pr_auc"],
            "Test Accuracy": test_m["accuracy"],
            "Test Precision": test_m["precision"],
            "Test Recall": test_m["recall"],
            "Test F1": test_m["f1"],
            "Test ROC-AUC": test_m["roc_auc"],
            "Test PR-AUC": test_m["pr_auc"],
        }
        comparison_rows.append(row)

        model_id = f"{algo.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        bundle = {
            "model_id": model_id,
            "model_name": algo,
            "trained_at": trained_at,
            "pipeline": pipe,
            "selected_features": features,
            "feature_defaults": feature_defaults,
            "training_config": {
                "test_size": float(test_size),
                "random_state": int(random_state),
                "cv_folds": int(cv_folds),
                "use_class_weight": bool(use_class_weight),
                "scaler_name": scaler_name,
                "use_log1p": bool(use_log1p),
                "drop_duplicates": bool(drop_duplicates),
                "hyperparameters": algo_hp,
                "includes_pagevalues": "PageValues" in features,
            },
            "dataset_info": {
                "n_total": len(clean_df),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "positive_rate_pct": round(float(y_all.mean() * 100), 2),
                "cleaning_summary": clean_info,
            },
            "cv_metrics": cv_summary,
            "cv_std": cv_std,
            "test_metrics": test_m,
            "confusion_matrix": cm_dict,
            "classification_report": cls_rep,
            "curves": curves,
            "feature_importance": imp,
        }
        model_details[algo] = bundle
        saved_bundles[algo] = bundle
        if persist_best:
            model_manager.save_classification_bundle(bundle)

    # 3.5 Rank trained algorithms by the selected primary evaluation metric
    sort_col = {
        "f1": "Test F1",
        "roc_auc": "Test ROC-AUC",
        "pr_auc": "Test PR-AUC",
        "recall": "Test Recall",
        "precision": "Test Precision",
    }.get(primary_metric, "Test F1")

    comparison_df = pd.DataFrame(comparison_rows).sort_values(sort_col, ascending=False).reset_index(drop=True)
    best_algo = str(comparison_df.iloc[0]["Algorithm"])

    # Ensure the highest-ranked model is saved as the default active_classifier.joblib
    if persist_best and best_algo in saved_bundles:
        model_manager.save_classification_bundle(saved_bundles[best_algo])

    return {
        "trained_at": trained_at,
        "best_algorithm": best_algo,
        "primary_metric": primary_metric,
        "selected_features": features,
        "dataset_info": saved_bundles[best_algo]["dataset_info"],
        "comparison_table": comparison_df.to_dict(orient="records"),
        "models": {k: {mk: mv for mk, mv in v.items() if mk != "pipeline"} for k, v in model_details.items()},
    }


# =============================================================================
# 4. PAGEVALUES INVESTIGATION (SECTION 6)
#    Trains a model using all 17 input features vs. a comparison model excluding
#    PageValues (16 features) on the exact same stratified split to quantify
#    how strongly PageValues drives purchase prediction scores.
# =============================================================================
def run_pagevalues_investigation(
    df: pd.DataFrame,
    algorithm_name: str = "Random Forest Classifier",
    test_size: float = 0.20,
    random_state: int = 42,
    use_class_weight: bool = True,
    drop_duplicates: bool = False,
) -> dict:
    """Section 6 PageValues Investigation:
    Trains a model using all 17 input features vs a comparison model excluding PageValues (16 features)
    on the exact same stratified split and compares evaluation metrics."""
    all_17 = list(data_validation.INPUT_FEATURES)
    without_pv_16 = [f for f in all_17 if f != "PageValues"]

    res_with = train_and_evaluate_models(
        df,
        selected_algorithms=[algorithm_name],
        selected_features=all_17,
        test_size=test_size,
        random_state=random_state,
        use_class_weight=use_class_weight,
        drop_duplicates=drop_duplicates,
        persist_best=False,
    )
    res_without = train_and_evaluate_models(
        df,
        selected_algorithms=[algorithm_name],
        selected_features=without_pv_16,
        test_size=test_size,
        random_state=random_state,
        use_class_weight=use_class_weight,
        drop_duplicates=drop_duplicates,
        persist_best=False,
    )

    m_with = res_with["models"][algorithm_name]["test_metrics"]
    m_without = res_without["models"][algorithm_name]["test_metrics"]
    cv_with = res_with["models"][algorithm_name]["cv_metrics"]
    cv_without = res_without["models"][algorithm_name]["cv_metrics"]

    rows = []
    for metric_key, label in [
        ("f1", "F1-Score"),
        ("recall", "Recall (Sensitivity)"),
        ("precision", "Precision"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("accuracy", "Accuracy"),
    ]:
        v_w = m_with[metric_key]
        v_wo = m_without[metric_key]
        rows.append({
            "Metric": label,
            "With PageValues (17 Features)": v_w,
            "Without PageValues (16 Features)": v_wo,
            "Difference (With - Without)": round(v_w - v_wo, 4),
            "CV (17 Features)": cv_with[metric_key],
            "CV (16 Features)": cv_without[metric_key],
        })

    return {
        "algorithm": algorithm_name,
        "comparison_rows": rows,
        "with_pagevalues": res_with["models"][algorithm_name],
        "without_pagevalues": res_without["models"][algorithm_name],
    }
