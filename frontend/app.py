"""
Streamlit + React Frontend Entrypoint.
Run from the project root:

    streamlit run frontend/app.py
"""
import sys
from pathlib import Path

import streamlit as st

FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent
for p in (str(FRONTEND_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Force reload of config and service modules on Streamlit Cloud hot-reload
import importlib
try:
    import ml.service as _ml_service
    importlib.reload(_ml_service)
except Exception:
    pass
import api_client
importlib.reload(api_client)
import config
importlib.reload(config)
try:
    from components import ui
except ImportError:
    from frontend.components import ui

# =============================================================================
# 1. PAGE CONFIGURATION & DESIGN SYSTEM CSS
# =============================================================================
st.set_page_config(
    page_title="Retail Purchase Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()

# =============================================================================
# 2. NAVIGATION REGISTRY (CLEAN SaaS SIDEBAR LABELS, ZERO NUMBERED GUIDE TITLES)
# =============================================================================
VIEWS_DIR = FRONTEND_DIR / "views"
PAGE_SPECS = [
    (VIEWS_DIR / "home.py", "Dashboard", "📊", True),
    (VIEWS_DIR / "data_explorer.py", "Data & Preprocessing", "🗂️", False),
    (VIEWS_DIR / "insights.py", "Exploratory Analysis", "🔍", False),
    (VIEWS_DIR / "segmentation.py", "Customer Segmentation", "👥", False),
    (VIEWS_DIR / "model_performance.py", "Model Training & Eval", "🧠", False),
    (VIEWS_DIR / "prediction.py", "Purchase Prediction", "🎯", False),
    (VIEWS_DIR / "history.py", "Prediction History", "📜", False),
    (VIEWS_DIR / "about.py", "System Architecture", "ℹ️", False),
]

pages_with_meta = []
st_pages = []
for path, title, icon, is_default in PAGE_SPECS:
    p_obj = st.Page(str(path), title=title, icon=icon, default=is_default)
    st_pages.append(p_obj)
    pages_with_meta.append((p_obj, title, icon))

# Hide Streamlit's default unstyled page list and render our branded SaaS sidebar
nav = st.navigation(st_pages, position="hidden")

try:
    h_info = api_client.health()
    ins = api_client.get_insights()
    test_m = ins.get("test_metrics", {})
    status_info = {
        "dataset_label": "Retailrocket Loaded",
        "dataset_sub": f"{ins.get('clean_event_rows', 2755641):,} events · {ins.get('total_sessions', 1761675):,} sessions",
        "model_label": ins.get("best_model") or "Classifier Active",
        "model_sub": f"ROC-AUC {test_m.get('roc_auc', 0.848):.3f} · K-Means Sil {ins.get('clustering_silhouette', 0.342):.3f}",
        "api_label": f"Connected ({h_info.get('backend', 'fastapi')})",
        "api_kind": "ok",
    }
except Exception:
    status_info = {
        "dataset_label": "Retailrocket Ready",
        "dataset_sub": "data/processed/clean.csv",
        "model_label": "Pipeline Ready",
        "model_sub": "Supervised + K-Means",
        "api_label": "Direct Service Mode",
        "api_kind": "info",
    }

ui.sidebar(pages_with_meta, status_info)
nav.run()
