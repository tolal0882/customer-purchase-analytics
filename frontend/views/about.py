"""
System Architecture & Project Information — Technical overview of the deployed
ML system, anti-leakage design, and team metadata.
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
    "System Architecture & Methodology",
    f"{config.PROJECT_TITLE} — {config.PROJECT_SUBTITLE}",
    badges=[
        ("Frontend: Streamlit + React 18 + Tailwind", "ok"),
        ("Backend: FastAPI + SQLite", "info"),
        ("ML: Scikit-learn Pipelines", "neutral"),
    ],
)

ui.kpi_row([
    ("Raw Clickstream Events", f"{ins.get('raw_rows', 2756101):,}", "data/raw/events.csv"),
    ("Engineered Sessions", f"{ins.get('total_sessions', 1761675):,}", "30-min inactivity threshold"),
    ("Supervised Snapshots", "30,000", "First 3 non-transaction events"),
    ("Clustered Profiles", "25,000", "Customer & Session K-Means"),
])

c1, c2 = st.columns(2)
with c1, ui.chart_card("Supervised Purchase Prediction Pipeline", "Early-session conversion classification (ml/train.py & ml/features.py)"):
    st.markdown(
        """
        - **Prediction Target:** `will_purchase` (`Yes` / `No`) — whether a transaction occurs **strictly after** the first 3 non-transaction events in the session.
        - **Leakage Prevention:** Identifiers (`visitorid`, `session_id`, `itemid`, `transactionid`) and post-snapshot counts are blocked from model inputs.
        - **In-Pipeline Feature Engineering:** `cart_to_view_ratio`, `cart_interaction_ratio`, `has_early_cart`, `item_focus_ratio`, `avg_seconds_per_action`, `is_returning_visitor`, and `historical_buy_rate`.
        - **Candidate Classifiers:** Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting evaluated with cross-validation and `TrainingFoldResampler`.
        """
    )

with c2, ui.chart_card("Unsupervised Behavioral Segmentation Pipeline", "Customer & session K-Means clustering (ml/clustering.py)"):
    st.markdown(
        r"""
        - **Dual Scope:** Supports both **Customer-Level Profiles** (`visitorid`) and **Session-Level Profiles** (`session_id`).
        - **Transformation Pipeline:** Skewed count/duration features undergo `np.log1p` transformation followed by `StandardScaler`, `RobustScaler`, or `MinMaxScaler`.
        - **Cluster Selection & Projection:** Evaluated across `K = 2..8` using **Within-Cluster Inertia (Elbow Method)**, **Silhouette Score**, and **Davies–Bouldin Index**, projected onto 2-D **PCA**.
        - **Data-Driven Personas:** Segment names and z-score bars are computed dynamically from cluster centroids.
        """
    )

with ui.chart_card("End-to-End CRISP-DM System Architecture", "Layered full-stack separation between UI, REST API, ML service, and storage"):
    arch_df = pd.DataFrame([
        {"Layer": "1. Data Engineering", "Module": "ml/data/make_dataset.py", "Responsibility": "Validates raw events.csv, removes 460 exact duplicates, engineers 30-min sessions, and builds clean.csv & behavioral profiles."},
        {"Layer": "2. Feature Pipeline", "Module": "ml/features.py", "Responsibility": "Scikit-learn FunctionTransformer + ColumnTransformer fitted strictly on training folds."},
        {"Layer": "3. Supervised & Unsupervised ML", "Module": "ml/train.py & ml/clustering.py", "Responsibility": "Trains 4 classifiers + K-Means (K=2..8) and persists experiments to models/."},
        {"Layer": "4. Service & SQLite Persistence", "Module": "ml/service.py & data/predictions.db", "Responsibility": "Unified inference, local feature-perturbation explanation, and persistent SQLite prediction log."},
        {"Layer": "5. REST API Backend", "Module": "backend_fastapi/main.py", "Responsibility": "FastAPI endpoints (/api/health, /schema, /data, /preprocess, /train, /predict, /predictions, /segments, /insights)."},
        {"Layer": "6. Hybrid Web Frontend", "Module": "frontend/app.py & frontend/components/", "Responsibility": "Streamlit + compiled React/TypeScript/Tailwind interactive components (PredictionForm, ClusterExplorer, ModelCompare)."},
    ])
    st.dataframe(arch_df, use_container_width=True, hide_index=True)

d1, d2 = st.columns(2)
with d1, ui.chart_card("Dataset Reference"):
    st.write(config.DATA_SOURCE)
    st.markdown("- **Source:** [Kaggle — Retailrocket Recommender System Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)")

with d2, ui.chart_card("Project Team"):
    st.write(f"**Course:** {config.CLASS_NAME}")
    for member in config.TEAM_MEMBERS:
        st.markdown(f"- {member}")
