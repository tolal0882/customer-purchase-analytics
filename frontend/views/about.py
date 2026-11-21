"""
System Architecture & Team Task Allocation — Technical overview of the deployed
ML system, anti-leakage design, and the 5-member team responsibility matrix (25 tasks).
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
    "System Architecture & Team Task Matrix",
    f"{config.PROJECT_TITLE} — {config.PROJECT_SUBTITLE}",
    badges=[
        ("Team Size: 5 Members (25 Tasks)", "ok"),
        ("Frontend: Streamlit + React 18 + Tailwind", "info"),
        ("Backend & ML: FastAPI + Scikit-learn + SQLite", "neutral"),
    ],
)

ui.kpi_row([
    ("Raw Clickstream Events", f"{ins.get('raw_rows', 2756101):,}", "data/raw/events.csv"),
    ("Engineered Sessions", f"{ins.get('total_sessions', 1761675):,}", "30-min inactivity threshold"),
    ("Supervised Snapshots", "30,000", "First 3 non-transaction events"),
    ("Clustered Profiles", "25,000", "Customer & Session K-Means"),
])

# =============================================================================
# 1. 5-MEMBER TEAM TASK ALLOCATION (5 TASKS PER MEMBER = 25 TASKS)
# =============================================================================
with ui.chart_card(
    "Team Members & 25-Task Responsibility Matrix (5 Tasks per Member)",
    "Equal workload distribution across Architecture/Backend, Data Engineering, EDA/Features, Supervised ML, and Unsupervised ML/UI",
):
    summary_rows = []
    for member_info in config.TEAM_TASK_ASSIGNMENTS:
        summary_rows.append({
            "Team Member": member_info["member"],
            "Primary Role": member_info["role"],
            "Assigned Tasks": "5 Tasks",
            "Owned Modules & Files": member_info["files"],
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    cols = st.columns(2)
    for idx, member_info in enumerate(config.TEAM_TASK_ASSIGNMENTS):
        target_col = cols[idx % 2] if idx < 4 else st.container()
        with target_col:
            with st.expander(f"👤 {member_info['member']} — {member_info['role']}", expanded=True):
                st.caption(f"**Owned Files:** `{member_info['files']}`")
                for t in member_info["tasks"]:
                    st.markdown(f"- {t}")

# =============================================================================
# 2. SUPERVISED & UNSUPERVISED ML METHODOLOGY
# =============================================================================
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
        {"Layer": "1. Data Engineering", "Owner": "Seng Ranut", "Module": "ml/data/make_dataset.py", "Responsibility": "Validates raw events.csv, removes 460 exact duplicates, engineers 30-min sessions, and builds clean.csv & behavioral profiles."},
        {"Layer": "2. Feature Pipeline & EDA", "Owner": "Phat Vandet", "Module": "ml/features.py & notebooks/01_eda.ipynb", "Responsibility": "Scikit-learn FunctionTransformer + ColumnTransformer fitted strictly on training folds + 6-part EDA."},
        {"Layer": "3. Supervised Classification", "Owner": "Vy Soksamnang", "Module": "ml/train.py & ml/evaluate.py", "Responsibility": "Trains 4 classifiers with TrainingFoldResampler, evaluates ROC/PR/F1, and computes local feature perturbations."},
        {"Layer": "4. Unsupervised Clustering & React UI", "Owner": "Yos Sak", "Module": "ml/clustering.py & frontend/components/", "Responsibility": "Fits Customer & Session K-Means (K=2..8) + PCA and bridges React 18 + Tailwind components."},
        {"Layer": "5. Service, FastAPI & Cloud Deployment", "Owner": "Long Tola (Leader)", "Module": "backend_fastapi/main.py & ml/service.py", "Responsibility": "FastAPI REST endpoints, SQLite prediction history, 13-test pytest suite, and GitHub / Streamlit Cloud deployment."},
    ])
    st.dataframe(arch_df, use_container_width=True, hide_index=True)
