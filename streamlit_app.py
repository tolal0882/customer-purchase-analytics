"""
Main Streamlit Entrypoint (Section 8, 9 & 11):
**Online Shopping Purchase Intention Prediction and Session Behavior Segmentation Using Machine Learning**
Subtitle: An Interactive Data Science Application Using Streamlit

Run from the project root:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import data_loader, model_manager, ui


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


def main() -> None:
    st.set_page_config(
        page_title="Online Shopping Purchase Intention & Session Segmentation",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="auto",
    )
    ui.inject_css()

    pg_dashboard = st.Page(_page_dashboard, title="1. Dashboard", icon="📊", url_path="dashboard", default=True)
    pg_eda = st.Page(_page_dataset_eda, title="2. Dataset Explorer & EDA", icon="🔍", url_path="dataset_eda")
    pg_prep = st.Page(_page_preprocessing, title="3. Data Preprocessing", icon="🧹", url_path="preprocessing")
    pg_pred = st.Page(_page_purchase_prediction, title="4. Purchase Prediction", icon="🎯", url_path="purchase_prediction")
    pg_seg = st.Page(_page_session_segmentation, title="5. Session Segmentation", icon="🧩", url_path="session_segmentation")
    pg_about = st.Page(_page_about, title="6. About Project", icon="ℹ️", url_path="about")

    nav = st.navigation(
        {
            "Overview & Data Analysis": [pg_dashboard, pg_eda, pg_prep],
            "Machine Learning Tasks": [pg_pred, pg_seg],
            "Documentation": [pg_about],
        },
        position="sidebar",
    )

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


if __name__ == "__main__":
    main()
