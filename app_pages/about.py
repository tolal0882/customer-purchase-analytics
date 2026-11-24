"""
PAGE 6 — About Project (Section 8):
  Includes:
    - Project introduction & problem statement
    - Research objectives & research questions
    - Dataset description, official UCI source, CC BY 4.0 license & citation
    - CRISP-DM methodology
    - Classification methodology, Clustering methodology & Evaluation metrics
    - Actual experimental results dynamically loaded from trained models
    - Methodological limitations (session-level scope, not long-term customer identities
      or validated real-time early-session prediction)
    - References
"""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from src import data_loader, data_validation, model_manager, ui

ui.inject_css()
df_raw = data_loader.require_dataset()
val_report = data_validation.validate_dataset(df_raw)

ui.page_header(
    title="About Project & Academic Methodology",
    subtitle="Online Shopping Purchase Intention Prediction and Session Behavior Segmentation Using Machine Learning — An Interactive Data Science Application Using Streamlit.",
    category="Page 6 · Academic Documentation & CRISP-DM Methodology",
    badges=[
        ("UCI Dataset ID #468 (12,330 Sessions)", "ok"),
        ("CRISP-DM Lifecycle", "info"),
        ("License: CC BY 4.0", "neutral"),
    ],
)

# =============================================================================
# 1. INTRODUCTION, PROBLEM STATEMENT, RESEARCH OBJECTIVES & QUESTIONS
# =============================================================================
c1, c2 = st.columns(2)
with c1, ui.chart_card("1. Project Introduction & Problem Statement"):
    st.markdown(
        """
        **Project Introduction:**
        E-commerce platforms record rich session-level telemetry describing how visitors navigate administrative, informational, and product-related pages. However, the vast majority of browsing sessions end without a transaction. In the **UCI Online Shoppers Purchasing Intention Dataset**, **84.53% (`10,422`)** of the **`12,330`** recorded sessions do not convert, while only **15.47% (`1,908`)** finalize with a purchase (`Revenue = True`).

        **Problem Statement:**
        Analyzing raw web analytics logs manually cannot reliably separate high-intent purchasing sessions from casual browsing or high-bounce traffic. This project applies a rigorous **CRISP-DM** workflow combining **Supervised Learning (Binary Classification)** to predict session purchase intention and **Unsupervised Learning (K-Means Clustering)** to segment shopping sessions into interpretable behavioral groups.
        """
    )

with c2, ui.chart_card("2. Research Objectives & Research Questions"):
    st.markdown(
        """
        **Research Objectives:**
        1. **Supervised Purchase Intention Prediction:** Train, cross-validate, and compare **Logistic Regression**, **Decision Tree**, and **Random Forest** pipelines to predict `Revenue` under class imbalance (`15.47%` positive class).
        2. **`PageValues` Ablation Analysis:** Quantify how much predictive performance depends on Google Analytics `PageValues` by comparing a 17-feature pipeline against a 16-feature pipeline excluding `PageValues`.
        3. **Unsupervised Session Behavior Segmentation:** Group shopping sessions using **K-Means Clustering** (strictly excluding `Revenue` from inputs), evaluate $K$ via the **Elbow Method** and **Silhouette Score**, and compare post-clustering purchase rates.

        **Research Questions:**
        - *RQ1:* How accurately can supervised classifiers distinguish purchasing sessions (`Revenue = True`) from non-purchasing sessions in terms of Precision, Recall, F1-score, ROC-AUC, and PR-AUC?
        - *RQ2:* To what extent does excluding `PageValues` change classification performance across metrics?
        - *RQ3:* What distinct session behavior segments emerge from K-Means clustering on page visit, duration, and bounce/exit features, and how do purchase rates vary across those segments *after* clustering?
        """
    )

# =============================================================================
# 2. DATASET DESCRIPTION, SOURCE & OFFICIAL CITATION
# =============================================================================
with ui.chart_card("3. Dataset Description, Official Source & Citation", "Verified from data/raw/online_shoppers_intention.csv"):
    d1, d2 = st.columns([1.25, 1])
    with d1:
        st.markdown(
            f"""
            - **Dataset Name:** UCI Online Shoppers Purchasing Intention Dataset
            - **Official Repository URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
            - **Verified Dimensions:** **`{val_report['n_rows']:,}` sessions** $\\times$ **`{val_report['n_cols']}` columns** (`17` input features + `Revenue` target)
            - **Target Distribution (`Revenue`):**
              - `False` (No Purchase): **`10,422` sessions (`84.53%`)**
              - `True` (Purchase): **`1,908` sessions (`15.47%`)**
            - **Data Completeness:** **`0` missing values** across all 18 columns; **`125` exact duplicate rows** (`1.01%`) surfaced for user-confirmed handling.
            - **Scope Note:** The dataset was formed so that each session belongs to a different user over a 1-year period to avoid any tendency to a specific campaign, special day, user profile, or period.
            """
        )
    with d2:
        ui.note(
            "<b>Official License & Academic Citation:</b><br>"
            "Licensed under <b>Creative Commons Attribution 4.0 International (CC BY 4.0)</b>.<br><br>"
            "<i>Sakar, C. &amp; Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q.</i><br><br>"
            "<i>Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2019). Real-time prediction of online shoppers' "
            "purchasing intention using multilayer perceptron and LSTM recurrent neural networks. "
            "Neural Computing and Applications, 31, 6893–6908.</i>"
        )

# =============================================================================
# 3. CRISP-DM METHODOLOGY & ML PIPELINE ARCHITECTURE
# =============================================================================
with ui.chart_card("4. CRISP-DM Data Science Methodology", "Six-phase lifecycle implemented across the application"):
    crisp_rows = [
        {"CRISP-DM Phase": "1. Business Understanding", "Implementation in App": "Page 1 (Dashboard) & Page 6 (About Project)", "Methodological Focus": "Define session conversion KPIs, class imbalance challenges, and segmentation objectives."},
        {"CRISP-DM Phase": "2. Data Understanding", "Implementation in App": "Page 2 (Dataset Explorer & 4-Section EDA)", "Methodological Focus": "Audit schema, dtypes, 125 exact duplicates, feature distributions, and correlation structure."},
        {"CRISP-DM Phase": "3. Data Preparation", "Implementation in App": "Page 3 (Data Preprocessing) & src/preprocessing.py", "Methodological Focus": "User-confirmed duplicate handling + separate ColumnTransformer pipelines fitted strictly on X_train."},
        {"CRISP-DM Phase": "4. Modeling", "Implementation in App": "Page 4 (Purchase Prediction) & Page 5 (Session Segmentation)", "Methodological Focus": "Train Logistic Regression, Decision Tree, Random Forest + K-Means session clustering (Revenue excluded)."},
        {"CRISP-DM Phase": "5. Evaluation", "Implementation in App": "Page 4 (Metrics, ROC/PR, PageValues Study) & Page 5 (Elbow/Silhouette)", "Methodological Focus": "Evaluate Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, PageValues ablation, and Silhouette."},
        {"CRISP-DM Phase": "6. Deployment", "Implementation in App": "streamlit_app.py (6 Pages) + Joblib Model Persistence", "Methodological Focus": "Interactive session prediction with local history export and reproducible segment assignment export."},
    ]
    st.dataframe(pd.DataFrame(crisp_rows), use_container_width=True, hide_index=True)

m1, m2 = st.columns(2)
with m1, ui.chart_card("5. Supervised Classification & Evaluation Methodology"):
    st.markdown(
        """
        - **Splitting & Cross-Validation:** Default **80/20 stratified train/test split** (`random_state=42`) paired with **Stratified 5-Fold Cross-Validation** on the training split.
        - **Leakage Prevention:** `SimpleImputer`, `StandardScaler`/`RobustScaler`/`MinMaxScaler`, and `OneHotEncoder(handle_unknown='ignore')` are encapsulated inside a Scikit-learn `Pipeline` and fitted **only on training folds**.
        - **Imbalance & Evaluation Metrics:** Because `84.53%` of sessions are non-purchases, accuracy alone is insufficient. We evaluate **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, **PR-AUC**, **Confusion Matrix**, and the full **Classification Report**.
        """
    )

with m2, ui.chart_card("6. Unsupervised Session Segmentation Methodology"):
    st.markdown(
        """
        - **Behavioral Feature Set:** Starts with 9 core session activity, duration, bounce/exit, and special-day features, with optional inclusion of `PageValues`, `Month`, `VisitorType`, and `Weekend`.
        - **Strict Target Exclusion:** `Revenue` is never passed to K-Means. All numeric features are scaled (with optional `log1p` transform) so Euclidean distance is not dominated by raw duration seconds.
        - **Cluster Selection & Interpretation:** Evaluated across $K = 2 \\dots 8$ via **Inertia (Elbow Method)** and **Silhouette Score**, projected onto **2-D PCA**, and profiled from actual cluster feature means before comparing post-clustering purchase rates.
        """
    )

# =============================================================================
# 4. ACTUAL EXPERIMENTAL RESULTS (LOADED FROM SAVED MODELS)
# =============================================================================
with ui.chart_card("7. Actual Experimental Results (Live from Saved Models)", "Automatically populated from models/classification/ and models/clustering/"):
    saved_cls = model_manager.list_saved_classifiers()
    if saved_cls:
        st.markdown("**Saved Supervised Classification Results (Held-Out Test Set & 5-Fold CV):**")
        exp_rows = []
        for m in saved_cls[:6]:
            tm = m.get("test_metrics", {})
            cvm = m.get("cv_metrics", {})
            tc = m.get("training_config", {})
            exp_rows.append({
                "Model ID": m.get("model_id"),
                "Algorithm": m.get("model_name"),
                "Features": len(m.get("selected_features", [])),
                "Includes PageValues": tc.get("includes_pagevalues", True),
                "CV F1": cvm.get("f1"),
                "Test Precision": tm.get("precision"),
                "Test Recall": tm.get("recall"),
                "Test F1": tm.get("f1"),
                "Test ROC-AUC": tm.get("roc_auc"),
                "Test PR-AUC": tm.get("pr_auc"),
                "Test Accuracy": tm.get("accuracy"),
            })
        st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Train models on **Page 4 (Purchase Prediction)** to populate live classification results here.")

    km_meta_path = model_manager.CLUSTERING_DIR / "active_kmeans_metadata.json"
    if km_meta_path.exists():
        try:
            km_meta = json.loads(km_meta_path.read_text(encoding="utf-8"))
            st.markdown(
                f"**Latest K-Means Session Segmentation Run:** `K = {km_meta.get('n_clusters')}` clusters across "
                f"`{km_meta.get('n_sessions'):,}` sessions · **Silhouette Score:** `{km_meta.get('silhouette_score'):.4f}` · "
                f"**Inertia:** `{km_meta.get('inertia'):,.1f}` · **Davies–Bouldin Index:** `{km_meta.get('davies_bouldin_index'):.3f}`."
            )
        except Exception:
            pass

# =============================================================================
# 5. LIMITATIONS & REFERENCES
# =============================================================================
l1, l2 = st.columns(2)
with l1, ui.chart_card("8. Methodological Limitations"):
    st.markdown(
        """
        1. **Session-Level Scope (Not Long-Term Customer Identities):** Each record represents a single shopping session rather than a longitudinal customer history. Segments represent **session behavior segments**, not lifetime customer value or loyalty tiers.
        2. **Completed-Session Summaries vs. Real-Time Prediction:** The UCI dataset aggregates page counts, durations, exit rates, and `PageValues` over the full session. Predictions estimate purchase outcomes from session summaries and have **not** been validated for real-time early-session interception.
        3. **Predictions Are Statistical Estimates:** A predicted class of `1` (`Purchase`) is a probabilistic model output, never a confirmed transaction.
        """
    )

with l2, ui.chart_card("9. Academic References"):
    st.markdown(
        """
        1. **Sakar, C. & Kastro, Y. (2018).** *Online Shoppers Purchasing Intention Dataset.* UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q.
        2. **Sakar, C.O., Polat, S.O., Katircioglu, M., & Kastro, Y. (2019).** *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications, 31(10), 6893–6908.
        3. **Chapman, P., et al. (2000).** *CRISP-DM 1.0: Step-by-step data mining guide.* SPSS Inc.
        4. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
        """
    )
