"""Main Streamlit application entry point.

Run locally with:
    .venv/bin/streamlit run streamlit_app.py
"""
import streamlit as st

from app_pages import (
    about,
    common,
    customer_segmentation,
    dashboard,
    dataset_explorer,
    eda,
    model_evaluation,
    model_training,
    prediction_history,
    preprocessing,
    purchase_prediction,
)
from components.streamlit_components import ui
from config import settings

# =============================================================================
# 1. PAGE CONFIGURATION & GLOBAL SETUP
#    - Sets the browser tab title, icon, and wide dashboard layout.
#    - Ensures local folders (data/processed, models, experiments, history) exist.
#    - Injects our custom Forest Green CSS theme into Streamlit.
# =============================================================================
st.set_page_config(
    page_title="Retail Purchase Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
settings.ensure_dirs()
ui.inject_css()

# =============================================================================
# 2. PAGE ROUTING REGISTRY (10 PAGES)
#    - Maps each page key from config/settings.py to its render() function.
#    - Uses Streamlit's native st.Page + st.navigation so URLs and active states
#      work cleanly while letting us draw a custom dark-green sidebar.
# =============================================================================
PAGE_FUNCS = {
    "dashboard": dashboard.render,
    "dataset_explorer": dataset_explorer.render,
    "preprocessing": preprocessing.render,
    "eda": eda.render,
    "customer_segmentation": customer_segmentation.render,
    "model_training": model_training.render,
    "model_evaluation": model_evaluation.render,
    "purchase_prediction": purchase_prediction.render,
    "prediction_history": prediction_history.render,
    "about": about.render,
}
pages = {}
for i, (key, title, icon) in enumerate(settings.PAGES):
    pages[key] = st.Page(
        PAGE_FUNCS[key],
        title=title,
        icon=icon,
        url_path=key,
        default=(i == 0),  # Dashboard is the landing page
    )
# Store page objects in session_state so ui.sidebar() can render st.page_link()
st.session_state["_pages"] = pages

# =============================================================================
# 3. RENDER NAVIGATION SIDEBAR & ACTIVE PAGE
#    - position="hidden" hides Streamlit's default unstyled menu.
#    - ui.sidebar() draws our branded sidebar + live dataset/model status badges.
#    - nav.run() executes the currently selected page's render() function.
# =============================================================================
nav = st.navigation(list(pages.values()), position="hidden")
ui.sidebar(common.sidebar_status())
nav.run()
