"""
Streamlit frontend entrypoint (8 pages).
Run from the project root OR from frontend/:

    streamlit run frontend/app.py
"""
import sys
from pathlib import Path

import streamlit as st

# Allow importing `frontend.*` or local sibling modules regardless of working directory
FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent
for p in (str(FRONTEND_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import api_client
import config

# =============================================================================
# 1. PAGE SETUP & MODERN E-COMMERCE ANALYTICS STYLING
# =============================================================================
st.set_page_config(
    page_title=config.PROJECT_TITLE,
    page_icon=config.PROJECT_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; color: #263238; }
    .stApp { background-color: #F4F7F4; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1380px; }
    section[data-testid="stSidebar"] { background-color: #153E2D; }
    section[data-testid="stSidebar"] * { color: #EAF2EC !important; }
    section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {
        background-color: #2F855A !important; font-weight: 600; border-radius: 8px;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #DCE6DF; border-radius: 12px;
        padding: 0.85rem 1rem; box-shadow: 0 1px 3px rgba(21,62,45,0.05);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF; border-radius: 12px; border-color: #DCE6DF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 2. SIDEBAR STATUS & PROJECT IDENTITY
# =============================================================================
with st.sidebar:
    st.markdown(f"### {config.PROJECT_ICON} Retailrocket ML")
    st.caption("Purchase Prediction & Customer Segmentation")
    try:
        info = api_client.health()
        st.success(f"Backend: {info.get('backend', 'ok')}")
    except api_client.APIError:
        st.error("Backend not connected")
    st.caption(f"API endpoint: `{api_client.API_URL}`")

# =============================================================================
# 3. 8-PAGE NAVIGATION (FOLLOWING TEACHER'S st.Page + st.navigation PATTERN)
# =============================================================================
VIEWS_DIR = FRONTEND_DIR / "views"
pages = [
    st.Page(str(VIEWS_DIR / "home.py"), title="1. Home Dashboard", icon="🏠", default=True),
    st.Page(str(VIEWS_DIR / "data_explorer.py"), title="2. Data Explorer", icon="📊"),
    st.Page(str(VIEWS_DIR / "prediction.py"), title="3. Purchase Prediction", icon="🔮"),
    st.Page(str(VIEWS_DIR / "segmentation.py"), title="4. Customer Segmentation", icon="👥"),
    st.Page(str(VIEWS_DIR / "model_performance.py"), title="5. Model Performance", icon="📈"),
    st.Page(str(VIEWS_DIR / "history.py"), title="6. Prediction History", icon="🕘"),
    st.Page(str(VIEWS_DIR / "insights.py"), title="7. Customer Insights", icon="💡"),
    st.Page(str(VIEWS_DIR / "about.py"), title="8. About Project", icon="ℹ️"),
]
st.navigation(pages).run()
