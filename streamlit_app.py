"""
Main Streamlit Entrypoint (Section 8, 9 & 11):
**Online Shopping Purchase Intention Prediction and Session Behavior Segmentation Using Machine Learning**
Subtitle: An Interactive Data Science Application Using Streamlit

Run from the project root:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

import streamlit as st

# =============================================================================
# 1. WORKSPACE PATH SETUP
#    Ensures `src/` and `app_pages/` resolve regardless of working directory.
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import (
    classification,
    clustering,
    data_loader,
    data_validation,
    evaluation,
    model_manager,
    prediction,
    preprocessing,
    ui,
)

# Reload src/ modules on every rerun so local code edits update the UI immediately
for _mod in (ui, data_validation, data_loader, preprocessing, evaluation, model_manager, classification, prediction, clustering):
    importlib.reload(_mod)


# =============================================================================
# 2. SIX PAGE CALLABLE WRAPPERS (SECTION 8)
#    Using callable functions with absolute paths guarantees that Streamlit
#    locates each page file whether launched via `streamlit_app.py` or `frontend/app.py`.
# =============================================================================
def _page_dashboard():
    runpy.run_path(str(ROOT_DIR / "app_pages" / "dashboard.py"), run_name="__main__")


def _page_dataset_eda():
    runpy.run_path(str(ROOT_DIR / "app_pages" / "dataset_eda.py"), run_name="__main__")


def _page_preprocessing():
    runpy.run_path(str(ROOT_DIR / "app_pages" / "preprocessing.py"), run_name="__main__")


def _page_purchase_prediction():
    runpy.run_path(str(ROOT_DIR / "app_pages" / "purchase_prediction.py"), run_name="__main__")


def _page_session_segmentation():
    runpy.run_path(str(ROOT_DIR / "app_pages" / "session_segmentation.py"), run_name="__main__")


def _page_about():
    runpy.run_path(str(ROOT_DIR / "app_pages" / "about.py"), run_name="__main__")


# =============================================================================
# 3. APPLICATION SHELL, SIDEBAR NAVIGATION & LIVE STATUS PANEL
# =============================================================================
def main() -> None:
    # 3.1 Configure browser tab title, icon, wide layout, and inject Forest Green CSS
    st.set_page_config(
        page_title="Online Shopping Purchase Intention & Session Segmentation",
        page_icon=":material/shopping_bag:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    ui.inject_css()

    # 3.2 Register the 6 required application pages with Material UI vector icons
    pg_dashboard = st.Page(_page_dashboard, title="Dashboard", icon=":material/space_dashboard:", url_path="dashboard", default=True)
    pg_eda = st.Page(_page_dataset_eda, title="Dataset Explorer & EDA", icon=":material/query_stats:", url_path="dataset_eda")
    pg_prep = st.Page(_page_preprocessing, title="Data Preprocessing", icon=":material/tune:", url_path="preprocessing")
    pg_pred = st.Page(_page_purchase_prediction, title="Purchase Prediction", icon=":material/online_prediction:", url_path="purchase_prediction")
    pg_seg = st.Page(_page_session_segmentation, title="Session Segmentation", icon=":material/hub:", url_path="session_segmentation")
    pg_about = st.Page(_page_about, title="About Project", icon=":material/menu_book:", url_path="about")

    nav = st.navigation(
        {
            "Overview & Data Analysis": [pg_dashboard, pg_eda, pg_prep],
            "Machine Learning Tasks": [pg_pred, pg_seg],
            "Documentation": [pg_about],
        },
        position="sidebar",
    )

    # 3.3 Render live dataset & active model summary card inside the sidebar
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

    # 3.4 Execute the currently selected page
    nav.run()


if __name__ == "__main__":
    main()
