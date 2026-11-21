"""
Streamlit + React Frontend Entrypoint.
Run from the project root:

    streamlit run frontend/app.py

Layout:
  - Top app bar  : brand, breadcrumb (Category > Page), live system status, quick-link navigation row
  - Sidebar      : categorized navigation (Overview / Data / Machine Learning / Records & System)
                   + live Dataset / Model / Backend status panel
"""
import importlib
import sys
from pathlib import Path

import streamlit as st

FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent
for p in (str(FRONTEND_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import api_client
import config
importlib.reload(api_client)
importlib.reload(config)
try:
    from components import ui
except ImportError:
    from frontend.components import ui
# Streamlit keeps imported submodules cached between deploys; reload so new UI helpers are picked up
importlib.reload(ui)

# =============================================================================
# 1. PAGE CONFIGURATION & DESIGN SYSTEM
# =============================================================================
st.set_page_config(
    page_title="Retail Purchase Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()

# =============================================================================
# 2. CATEGORIZED NAVIGATION
# =============================================================================
VIEWS_DIR = FRONTEND_DIR / "views"


def _page(file: str, title: str, icon: str, default: bool = False):
    return st.Page(str(VIEWS_DIR / file), title=title, icon=icon, default=default)


pg_dashboard = _page("home.py", "Dashboard", "📊", default=True)
pg_data = _page("data_explorer.py", "Data & Preprocessing", "🗂️")
pg_eda = _page("insights.py", "Exploratory Analysis", "🔍")
pg_segments = _page("segmentation.py", "Customer Segmentation", "👥")
pg_models = _page("model_performance.py", "Model Training & Evaluation", "🧠")
pg_predict = _page("prediction.py", "Purchase Prediction", "🎯")
pg_history = _page("history.py", "Prediction History", "📜")
pg_system = _page("about.py", "System Overview", "ℹ️")

SECTIONS = {
    "Overview": [pg_dashboard],
    "Data": [pg_data, pg_eda],
    "Machine Learning": [pg_segments, pg_models, pg_predict],
    "Records & System": [pg_history, pg_system],
}
nav = st.navigation(SECTIONS, position="sidebar")

current_section = next(
    (name for name, pages in SECTIONS.items() if any(pg.title == nav.title for pg in pages)),
    "Overview",
)

# =============================================================================
# 3. LIVE SYSTEM STATUS
# =============================================================================
try:
    h_info = api_client.health()
    ins = api_client.get_insights()
    test_m = ins.get("test_metrics", {})
    api_label = "FastAPI online" if "fastapi" in str(h_info.get("backend", "")).lower() else "Direct service mode"
    status = {
        "dataset": f"{ins.get('clean_event_rows', 2755641):,} events · {ins.get('total_sessions', 1761675):,} sessions",
        "model": f"{ins.get('best_model') or 'Classifier'} · ROC-AUC {test_m.get('roc_auc', 0):.3f}",
        "backend": api_label,
        "pills": [
            (ins.get("best_model") or "Classifier ready", "ok"),
            (api_label, "info"),
        ],
    }
except Exception:
    status = {
        "dataset": "Retailrocket processed data",
        "model": "Pipeline ready",
        "backend": "Direct service mode",
        "pills": [("Pipeline ready", "ok"), ("Direct service mode", "info")],
    }

# =============================================================================
# 4. TOP APP BAR + SIDEBAR STATUS PANEL
# =============================================================================
ui.top_bar(current_section, nav.title, status["pills"], icon=nav.icon or "")
ui.sidebar_status(status["dataset"], status["model"], status["backend"])

nav.run()
