"""
Page 8 – About Project: project background, objectives, Retailrocket dataset,
CRISP-DM workflow, leakage prevention methodology, and team members.
"""
import pandas as pd
import streamlit as st

import config

st.title("ℹ️ About Project")
st.subheader(config.PROJECT_TITLE)
st.caption(config.PROJECT_SUBTITLE)
st.write(config.PROJECT_DESCRIPTION)

# =============================================================================
# 1. MAIN OBJECTIVES & DUAL ML METHODOLOGY
# =============================================================================
c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.subheader("A. Purchase Behavior Prediction (Supervised Learning)")
    st.markdown(
        """
        - **Objective:** Predict whether an active browsing session will result in a transaction (`will_purchase` = `Yes` / `No`).
        - **Leakage-Safe Snapshot Design:** Observes the customer's first **3 non-transaction interactions** (`views_count`, `unique_items_viewed`, `add_to_cart_count`, `elapsed_seconds`, `ends_with_cart`, `start_hour`, `is_weekend`) plus strictly earlier session history (`prior_sessions`, `prior_purchases`).
        - **Engineered Pipeline Features (`ml/features.py`):** `cart_to_view_ratio`, `item_focus_ratio`, `avg_seconds_per_action`, and `is_returning_visitor` computed inside a Scikit-learn `Pipeline` (`FunctionTransformer` + `ColumnTransformer`).
        - **Models Compared (`ml/train.py`):** Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting via 5-fold Stratified Cross-Validation + `RandomizedSearchCV`.
        """
    )

with c2, st.container(border=True):
    st.subheader("B. Customer Segmentation (Unsupervised Learning)")
    st.markdown(
        """
        - **Objective:** Group engaged e-commerce visitors into distinct behavioral segments (`ml/clustering.py`).
        - **Behavioral Features:** `total_sessions`, `total_views`, `unique_items_viewed`, `total_add_to_cart`, `avg_session_duration_sec`, `avg_items_per_session`, `cart_abandonment_rate`, and `recency_days` (`visitorid` is excluded).
        - **Preprocessing & Evaluation:** Right-skewed features are transformed with `np.log1p` and standardized with `StandardScaler`. Evaluated across `K = 2..8` using the **Elbow Method (Inertia)** and **Silhouette Score**.
        - **Visualization & Profiling:** 2-D **Principal Component Analysis (PCA)** projection and data-driven segment names with post-clustering conversion rate comparisons.
        """
    )

# =============================================================================
# 2. CRISP-DM WORKFLOW & JUPYTER NOTEBOOKS
# =============================================================================
with st.container(border=True):
    st.subheader("CRISP-DM Data Science Workflow")
    crisp_df = pd.DataFrame([
        {"Stage": "1. Business Understanding", "Artifact / Module": "Project Scope & frontend/views/about.py", "Summary": "Define early-session conversion prediction and behavioral customer segmentation goals."},
        {"Stage": "2. Data Understanding (EDA)", "Artifact / Module": "notebooks/01_eda.ipynb", "Summary": "Inspect 2.75M Retailrocket events, event funnel, duplicate rows, and target imbalance."},
        {"Stage": "3. Data Preparation & Cleaning", "Artifact / Module": "notebooks/02_data_cleaning.ipynb ➜ ml/data/make_dataset.py", "Summary": "Remove 460 duplicates, construct 30-min inactivity sessions, and build clean.csv & customer_profiles.csv."},
        {"Stage": "4. Feature Engineering", "Artifact / Module": "ml/features.py", "Summary": "In-pipeline FunctionTransformer + ColumnTransformer (SimpleImputer, StandardScaler, OneHotEncoder)."},
        {"Stage": "5. Machine Learning Modeling", "Artifact / Module": "notebooks/03_modeling.ipynb ➜ ml/train.py & ml/clustering.py", "Summary": "5-fold CV across 4 classifiers + RandomizedSearchCV tuning + K-Means clustering (K=2..8)."},
        {"Stage": "6. Model Evaluation", "Artifact / Module": "notebooks/04_evaluation.ipynb ➜ ml/evaluate.py", "Summary": "Held-out test F1, Recall, Precision, ROC-AUC, PR-AUC, Confusion Matrix, Elbow & Silhouette."},
        {"Stage": "7. Deployment & Web Application", "Artifact / Module": "backend_fastapi/main.py + frontend/app.py (8 pages)", "Summary": "FastAPI REST backend with SQLite prediction history + 8-page Streamlit & React frontend."},
    ])
    st.dataframe(crisp_df, use_container_width=True, hide_index=True)

# =============================================================================
# 3. DATASET & TEAM INFORMATION
# =============================================================================
d1, d2 = st.columns(2)
with d1, st.container(border=True):
    st.subheader("Dataset")
    st.write(config.DATA_SOURCE)
    st.markdown("- **Kaggle Link:** https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset")

with d2, st.container(border=True):
    st.subheader("Team & Course Information")
    st.write(f"**Course / Class:** {config.CLASS_NAME}")
    for member in config.TEAM_MEMBERS:
        st.markdown(f"- {member}")
