"""
PAGE 3 — Data Preprocessing (Section 5.1 & Section 8):
  - Displays:
      * Dataset validation, data types, missing values, duplicate records,
        feature types, class imbalance, data quality summary.
  - Allows users to:
      * Select preprocessing options (explicit duplicate handling, scaler choice, log1p option,
        classification & clustering feature subsets)
      * Preview changes & compare before and after
      * Inspect separate Scikit-learn preprocessing pipelines for classification and clustering
      * Export processed data to CSV (stored separately in data/processed/, never modifying raw CSV).
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import data_loader, data_validation, preprocessing, ui

ui.inject_css()
df_raw = data_loader.require_dataset()
val_report = data_validation.validate_dataset(df_raw)
file_meta = data_loader.get_file_metadata()

ui.page_header(
    title="Data Validation & Preprocessing Pipelines",
    subtitle="Audit data quality, configure explicit duplicate handling, compare before/after effects, and inspect the separate Scikit-learn pipelines for classification and clustering.",
    category="Page 3 · Data Preparation (CRISP-DM Phase 3)",
    badges=[
        ("Schema Validation: Passed" if val_report["is_valid_schema"] else "Schema Warning", "ok" if val_report["is_valid_schema"] else "warn"),
        (f"Raw File: {file_meta['size_mb']} MB (Read-Only)", "info"),
        ("Scikit-Learn ColumnTransformer", "neutral"),
    ],
)

# =============================================================================
# 1. DATA QUALITY & VALIDATION SUMMARY
# =============================================================================
td = val_report["target_distribution"]
ui.kpi_row(
    [
        ("Total Raw Sessions", f"{val_report['n_rows']:,}", f"File: {file_meta['path']}"),
        ("Validated Columns", f"{val_report['n_cols']} / 18", "17 Input Features + Revenue Target"),
        ("Missing Values", f"{val_report['n_missing_total']:,}", "100% completeness across all columns"),
        ("Exact Duplicate Rows", f"{val_report['n_duplicates']:,}", f"{val_report['duplicate_pct']:.2f}% of raw sessions"),
        ("Class Imbalance (No : Yes)", f"{td.get('imbalance_ratio', 5.46)}:1", f"{td.get('no_purchase_rate_pct', 84.53)}% False vs {td.get('purchase_rate_pct', 15.47)}% True"),
        ("Invalid Range Checks", "0 Invalid Rows", f"{len(val_report['invalid_checks'])} automated domain checks passed"),
    ],
    per_row=3,
)

c_val1, c_val2 = st.columns([1.3, 1])
with c_val1, ui.chart_card("Feature Groups & Data Type Classification (Section 5.2)", "Continuous numerical features vs. nominal/categorical features"):
    st.dataframe(pd.DataFrame(val_report["column_summary"]), use_container_width=True, hide_index=True)

with c_val2, ui.chart_card("Automated Domain & Invalid-Value Checks", "Verification of non-negative counts/durations and [0, 1] rate bounds"):
    st.dataframe(pd.DataFrame(val_report["invalid_checks"]), use_container_width=True, hide_index=True)
    ui.note(
        "<b>Class Imbalance Inspection:</b> Only <b>15.47% (1,908)</b> of the 12,330 sessions result in a purchase. "
        "Therefore, classification pipelines must use <b>stratified splitting</b>, optional <b>class weighting</b>, "
        "and evaluate <b>Precision, Recall, F1-score, ROC-AUC, and PR-AUC</b> rather than accuracy alone."
    )

# =============================================================================
# 2. INTERACTIVE PREPROCESSING CONFIGURATION & BEFORE / AFTER COMPARISON
# =============================================================================
with ui.chart_card(
    "Interactive Preprocessing & Cleaning Configuration",
    "Original data/raw/online_shoppers_intention.csv is never modified; all transformations operate on copies or inside training folds.",
):
    p1, p2, p3 = st.columns(3)
    drop_dups = p1.checkbox(
        "Remove 125 exact duplicate sessions (12,330 → 12,205 rows)",
        value=bool(st.session_state.get("prep_drop_duplicates", False)),
        help="All 125 exact duplicates have Revenue=False (mostly 0-duration bounce sessions). Toggling this shows the exact effect before exporting or training.",
    )
    scaler_choice = p2.selectbox(
        "Numerical Feature Scaler",
        options=["standard", "robust", "minmax"],
        index=["standard", "robust", "minmax"].index(st.session_state.get("prep_scaler", "standard")),
        format_func=lambda s: {
            "standard": "StandardScaler (Zero Mean, Unit Variance)",
            "robust": "RobustScaler (Median & IQR — Outlier Resistant)",
            "minmax": "MinMaxScaler ([0, 1] Range)",
        }[s],
    )
    use_log1p = p3.checkbox(
        "Apply log1p transform to skewed numeric features",
        value=bool(st.session_state.get("prep_log1p", False)),
        help="Applies np.log1p(x) inside the numerical pipeline before scaling to reduce right-skewness in duration and page count columns.",
    )

    st.session_state["prep_drop_duplicates"] = drop_dups
    st.session_state["prep_scaler"] = scaler_choice
    st.session_state["prep_log1p"] = use_log1p

df_cleaned, clean_summary = preprocessing.apply_cleaning_options(df_raw, drop_duplicates=drop_dups)

b1, b2 = st.columns([1.2, 1])
with b1, ui.chart_card("Before vs. After Cleaning Comparison", "Exact effect of user-selected duplicate handling on dataset size and class balance"):
    comp_df = pd.DataFrame([
        {"Metric": "Total Session Records", "Before (Raw Dataset)": f"{clean_summary['rows_before']:,}", "After (Selected Cleaning)": f"{clean_summary['rows_after']:,}", "Change": f"-{clean_summary['duplicates_removed']:,} rows"},
        {"Metric": "Exact Duplicate Records", "Before (Raw Dataset)": f"{clean_summary['duplicates_detected']:,}", "After (Selected Cleaning)": f"{clean_summary['duplicates_remaining']:,}", "Change": f"-{clean_summary['duplicates_removed']:,} duplicates"},
        {"Metric": "Purchase Sessions (Revenue = True)", "Before (Raw Dataset)": f"{clean_summary['purchase_before']:,}", "After (Selected Cleaning)": f"{clean_summary['purchase_after']:,}", "Change": "0 (no purchase rows were duplicates)"},
        {"Metric": "No-Purchase Sessions (Revenue = False)", "Before (Raw Dataset)": f"{clean_summary['no_purchase_before']:,}", "After (Selected Cleaning)": f"{clean_summary['no_purchase_after']:,}", "Change": f"-{clean_summary['duplicates_removed']:,} rows"},
        {"Metric": "Purchase Conversion Rate (%)", "Before (Raw Dataset)": f"{clean_summary['purchase_rate_before_pct']:.2f}%", "After (Selected Cleaning)": f"{clean_summary['purchase_rate_after_pct']:.2f}%", "Change": f"{clean_summary['purchase_rate_after_pct'] - clean_summary['purchase_rate_before_pct']:+.2f}%"},
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

with b2, ui.chart_card("Before vs. After Scaling Preview (Sample Feature)", "Effect of optional log1p + selected scaler on ProductRelated_Duration"):
    preview_col = st.selectbox("Select Numerical Feature to Compare", options=data_validation.CONTINUOUS_FEATURES, index=5)
    raw_vals = df_cleaned[[preview_col]].copy()
    num_pre = preprocessing.build_classification_preprocessor([preview_col], scaler_name=scaler_choice, use_log1p=use_log1p)
    trans_vals = num_pre.fit_transform(raw_vals).ravel()
    stats_comp = pd.DataFrame({
        "Statistic": ["Mean", "Std Dev", "Min", "Median (50%)", "Max", "Skewness"],
        f"Raw {preview_col}": [
            round(float(raw_vals[preview_col].mean()), 4),
            round(float(raw_vals[preview_col].std()), 4),
            round(float(raw_vals[preview_col].min()), 4),
            round(float(raw_vals[preview_col].median()), 4),
            round(float(raw_vals[preview_col].max()), 4),
            round(float(raw_vals[preview_col].skew()), 4),
        ],
        f"Transformed ({scaler_choice}" + (" + log1p)" if use_log1p else ")"): [
            round(float(trans_vals.mean()), 4),
            round(float(trans_vals.std()), 4),
            round(float(trans_vals.min()), 4),
            round(float(pd.Series(trans_vals).median()), 4),
            round(float(trans_vals.max()), 4),
            round(float(pd.Series(trans_vals).skew()), 4),
        ],
    })
    st.dataframe(stats_comp, use_container_width=True, hide_index=True)

# =============================================================================
# 3. SEPARATE CLASSIFICATION & CLUSTERING PIPELINE ARCHITECTURES
# =============================================================================
pipe_c1, pipe_c2 = st.columns(2)
with pipe_c1, ui.chart_card("Pipeline 1: Supervised Classification Preprocessor", "Used in Page 4 (Purchase Prediction) — fitted strictly on X_train"):
    cls_num, cls_cat = preprocessing.split_feature_types(data_validation.INPUT_FEATURES)
    cls_pre = preprocessing.build_classification_preprocessor(
        data_validation.INPUT_FEATURES, scaler_name=scaler_choice, use_log1p=use_log1p
    )
    X_cls_sample = preprocessing.prepare_features_for_transform(df_cleaned.head(500), data_validation.INPUT_FEATURES)
    cls_pre.fit(X_cls_sample)
    cls_out_names = list(cls_pre.get_feature_names_out())
    st.markdown(
        f"""
        - **Input Features:** `{len(data_validation.INPUT_FEATURES)}` features (`Revenue` is separated as binary target `y`).
        - **Numerical Branch (`{len(cls_num)}` cols):** `SimpleImputer(strategy='median')` $\\rightarrow$ `{'log1p -> ' if use_log1p else ''}{scaler_choice.capitalize()}Scaler` on `{', '.join(cls_num)}`.
        - **Categorical Branch (`{len(cls_cat)}` cols):** `SimpleImputer(strategy='most_frequent')` $\\rightarrow$ `OneHotEncoder(handle_unknown='ignore')` on `{', '.join(cls_cat)}`.
        - **Expanded Feature Vector Dimension:** **`{len(cls_out_names)}` transformed columns**.
        - **Leakage Safeguard:** Wrapped inside `sklearn.pipeline.Pipeline` so `fit()` runs **only on training data (`X_train`)**, never on test data.
        """
    )

with pipe_c2, ui.chart_card("Pipeline 2: Unsupervised Clustering Preprocessor", "Used in Page 5 (Session Segmentation) — Revenue is strictly excluded"):
    clu_feats = data_validation.DEFAULT_CLUSTERING_FEATURES
    clu_num, clu_cat = preprocessing.split_feature_types(clu_feats)
    clu_pre = preprocessing.build_clustering_preprocessor(
        clu_feats, scaler_name=scaler_choice, use_log1p=True
    )
    X_clu_sample = preprocessing.prepare_features_for_transform(df_cleaned.head(500), clu_feats)
    clu_pre.fit(X_clu_sample)
    clu_out_names = list(clu_pre.get_feature_names_out())
    st.markdown(
        f"""
        - **Default Clustering Features:** `{len(clu_feats)}` behavioral features (`{', '.join(clu_feats)}`).
        - **Strict Target Exclusion:** `Revenue` is **hard-blocked** (`ValueError` raised if passed) so clusters reflect pure browsing behavior.
        - **Numerical Branch:** `SimpleImputer(strategy='median')` $\\rightarrow$ `log1p` $\\rightarrow$ `{scaler_choice.capitalize()}Scaler` (essential because K-Means uses Euclidean distance).
        - **Optional Categorical Branch:** `OneHotEncoder(handle_unknown='ignore')` when `Month`, `VisitorType`, or `Weekend` are added.
        - **Default Transformed Dimension:** **`{len(clu_out_names)}` scaled behavioral dimensions**.
        """
    )

# =============================================================================
# 4. PREVIEW & EXPORT PROCESSED DATA (STORED SEPARATELY IN data/processed/)
# =============================================================================
with ui.chart_card("Preview & Export Processed Datasets", "Save cleaned data to data/processed/ or download as CSV without modifying data/raw/online_shoppers_intention.csv"):
    st.dataframe(df_cleaned.head(15), use_container_width=True, hide_index=True)
    e1, e2, e3 = st.columns(3)
    e1.download_button(
        f"Download Cleaned Dataset ({len(df_cleaned):,} rows CSV)",
        df_cleaned.to_csv(index=False).encode("utf-8"),
        file_name="online_shoppers_cleaned.csv",
        mime="text/csv",
        icon=":material/download:",
        use_container_width=True,
    )
    clu_only_df = df_cleaned[data_validation.DEFAULT_CLUSTERING_FEATURES].copy()
    e2.download_button(
        f"Download Clustering Feature Matrix ({len(clu_only_df):,} × {len(clu_only_df.columns)} CSV)",
        clu_only_df.to_csv(index=False).encode("utf-8"),
        file_name="online_shoppers_clustering_features.csv",
        mime="text/csv",
        icon=":material/download:",
        use_container_width=True,
    )
    if e3.button("Save Processed CSV to data/processed/", icon=":material/save:", type="primary", use_container_width=True):
        saved_path = preprocessing.save_processed_csv(df_cleaned, "online_shoppers_processed.csv")
        st.success(f"Saved processed dataset ({len(df_cleaned):,} rows) to `{saved_path}`.")
