"""
System Overview — what the system does, how the ML pipelines work, how the
layers fit together, and the API surface. (Team/work allocation lives in docs/, not in the app.)
"""
import pandas as pd
import streamlit as st

import api_client
import config
try:
    from components import ui
except ImportError:
    from frontend.components import ui

ui.inject_css()

try:
    ins = api_client.get_insights()
except Exception:
    ins = {}

ui.page_header(
    "System Overview",
    "How the platform turns raw clickstream events into purchase predictions and customer segments.",
    badges=[
        ("Streamlit + React", "ok"),
        ("FastAPI + SQLite", "info"),
        ("Scikit-learn", "neutral"),
    ],
)

ui.kpi_row([
    ("Raw clickstream events", f"{ins.get('raw_rows', 2756101):,}", "Retailrocket events.csv"),
    ("Engineered sessions", f"{ins.get('total_sessions', 1761675):,}", f"{ins.get('session_timeout_minutes', 30):g}-min inactivity timeout"),
    ("Prediction snapshots", "30,000", "First 3 non-transaction events"),
    ("Clustered profiles", "25,000", "Customers and sessions"),
])

with ui.chart_card("What the system does", "Two machine-learning components on one data foundation"):
    st.markdown(
        """
        1. **Purchase prediction (supervised).** From the first three non-transaction actions of a shopping session
           (plus earlier history of the visitor) it estimates the probability that the session ends in a purchase.
        2. **Customer & session segmentation (unsupervised).** K-Means groups visitors and sessions by behavior
           and describes each group with a data-derived profile and a recommended action.
        """
    )

c1, c2 = st.columns(2)
with c1, ui.chart_card("Purchase prediction pipeline", "ml/features.py · ml/train.py · ml/evaluate.py"):
    st.markdown(
        """
        - **Target:** `will_purchase` — a transaction occurs strictly after the third event of the session.
        - **Leakage prevention:** identifiers and post-snapshot information never reach the model; imputing, scaling and encoding are fitted on training folds only.
        - **Engineered features:** cart-to-view ratio, item focus ratio, average seconds per action, returning-visitor flag, historical buy rate.
        - **Models:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting; chronological or stratified split; class weighting and in-fold resampling; adjustable decision threshold.
        - **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC / PR curves, feature importance.
        """
    )

with c2, ui.chart_card("Segmentation pipeline", "ml/clustering.py"):
    st.markdown(
        r"""
        - **Scopes:** customer-level (one row per visitor) and session-level (one row per session).
        - **Preprocessing:** `log1p` for skewed counts and durations, then Standard / Robust / MinMax scaling.
        - **Model selection:** K from 2 to 8 compared with the Elbow method, Silhouette score and Davies–Bouldin index; 2-D PCA projection.
        - **Profiles:** segment names and z-score bars are derived from cluster centroids; purchase rate is compared *after* clustering.
        """
    )

with ui.chart_card("Architecture", "UI, API, ML service and storage are separated"):
    arch_df = pd.DataFrame([
        {"Layer": "Data preparation", "Module": "ml/data/make_dataset.py", "Responsibility": "Validate raw events, remove duplicates, build sessions, snapshots and profiles."},
        {"Layer": "Features", "Module": "ml/features.py", "Responsibility": "In-pipeline feature engineering and preprocessing."},
        {"Layer": "Modelling", "Module": "ml/train.py, ml/clustering.py", "Responsibility": "Train and evaluate classifiers; fit K-Means; persist models and experiments."},
        {"Layer": "Service", "Module": "ml/service.py", "Responsibility": "Inference, explanations, experiment registry, SQLite prediction history."},
        {"Layer": "REST API", "Module": "backend_fastapi/main.py", "Responsibility": "HTTP endpoints for data, training, prediction, segmentation and insights."},
        {"Layer": "Web app", "Module": "frontend/", "Responsibility": "Streamlit pages with embedded React + Tailwind components."},
    ])
    st.dataframe(arch_df, use_container_width=True, hide_index=True)

e1, e2 = st.columns(2)
with e1, ui.chart_card("REST API endpoints", "Served by FastAPI; the web app falls back to direct calls when the API is not running"):
    api_df = pd.DataFrame([
        {"Method": "GET", "Path": "/api/health", "Purpose": "Service status"},
        {"Method": "GET", "Path": "/api/schema", "Purpose": "Model input schema"},
        {"Method": "GET", "Path": "/api/data", "Purpose": "Browse datasets"},
        {"Method": "POST", "Path": "/api/preprocess", "Purpose": "Rebuild sessions with a timeout"},
        {"Method": "POST", "Path": "/api/train", "Purpose": "Train and register a model"},
        {"Method": "POST", "Path": "/api/predict", "Purpose": "Purchase probability"},
        {"Method": "GET/DELETE", "Path": "/api/predictions", "Purpose": "Prediction history"},
        {"Method": "GET/POST", "Path": "/api/segments", "Purpose": "Segmentation results and refit"},
        {"Method": "GET", "Path": "/api/insights", "Purpose": "Dashboard and analysis aggregates"},
    ])
    st.dataframe(api_df, use_container_width=True, hide_index=True)

with e2, ui.chart_card("Dataset"):
    st.write(config.DATA_SOURCE)
    st.markdown("- **Source:** [Kaggle — Retailrocket Recommender System Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)")
    st.markdown("- **Note:** Retailrocket logs only `view`, `addtocart` and `transaction`; there is no remove-from-cart event.")
