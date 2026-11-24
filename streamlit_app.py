"""
Main Streamlit Entrypoint (Section 8, 9 & 11):
**Online Shopping Purchase Intention Prediction and Session Behavior Segmentation Using Machine Learning**
Subtitle: An Interactive Data Science Application Using Streamlit

Run from the project root:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from src import data_loader, model_manager, ui

st.set_page_config(
    page_title="Online Shopping Purchase Intention & Session Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="auto",
)
ui.inject_css()

# =============================================================================
# 1. SIX MAIN APPLICATION PAGES (CATEGORIZED NATIVE NAVIGATION)
# =============================================================================
pg_dashboard = st.Page("app_pages/dashboard.py", title="1. Dashboard", icon="📊", default=True)
pg_eda = st.Page("app_pages/dataset_eda.py", title="2. Dataset Explorer & EDA", icon="🔍")
pg_prep = st.Page("app_pages/preprocessing.py", title="3. Data Preprocessing", icon="🧹")
pg_pred = st.Page("app_pages/purchase_prediction.py", title="4. Purchase Prediction", icon="🎯")
pg_seg = st.Page("app_pages/session_segmentation.py", title="5. Session Segmentation", icon="🧩")
pg_about = st.Page("app_pages/about.py", title="6. About Project", icon="ℹ️")

nav = st.navigation(
    {
        "Overview & Data Analysis": [pg_dashboard, pg_eda, pg_prep],
        "Machine Learning Tasks": [pg_pred, pg_seg],
        "Documentation": [pg_about],
    },
    position="sidebar",
)

# =============================================================================
# 2. SIDEBAR DATASET & MODEL STATUS PANEL
# =============================================================================
file_meta = data_loader.get_file_metadata()
active_cls = model_manager.load_classifier_bundle()

with st.sidebar:
    ds_status = "12,330 sessions × 18 cols" if file_meta["exists"] else "Awaiting CSV upload"
    cls_status = (
        f"{active_cls['model_name']} (F1={active_cls['test_metrics']['f1']:.3f})"
        if active_cls
        else "Ready to train on Page 4"
    )
    st.markdown(
        f'<div class="side-panel">'
        f'<b>UCI Dataset</b><br><span style="opacity:0.88">{ui.esc(ds_status)}</span>'
        f'<div style="height:0.45rem"></div>'
        f'<b>Active Classifier</b><br><span style="opacity:0.88">{ui.esc(cls_status)}</span>'
        f'<div style="height:0.45rem"></div>'
        f'<b>Target Variable</b><br><span style="opacity:0.88">Revenue (15.47% Purchase)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

nav.run()
