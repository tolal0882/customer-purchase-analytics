"""
Dataset Loading & Caching Module for the UCI Online Shoppers Purchasing Intention Dataset.

Preferred path: data/raw/online_shoppers_intention.csv
If absent, provides a Streamlit file uploader and clear instructions without modifying
any original files.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import data_validation, ui

# =============================================================================
# 1. WORKSPACE PATH CONSTANTS
#    Points to `data/raw/online_shoppers_intention.csv` and `data/processed/`.
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = ROOT_DIR / "data" / "raw" / "online_shoppers_intention.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


# =============================================================================
# 2. CACHED CSV LOADER (SECTION 4 & SECTION 12)
#    Uses @st.cache_data so the 12,330-row CSV is read from disk once and reused
#    across all 6 pages without modifying the original file.
# =============================================================================
@st.cache_data(show_spinner=False)
def load_raw_csv(path_str: str | None = None) -> pd.DataFrame:
    """Load the original UCI Online Shoppers Purchasing Intention CSV without modifying it."""
    path = Path(path_str) if path_str else RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path}")
    df = pd.read_csv(path)
    return df


# =============================================================================
# 3. FILE METADATA HELPER
#    Returns file existence, relative path, and file size in bytes and megabytes.
# =============================================================================
def get_file_metadata(path: Path = RAW_CSV_PATH) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path), "size_bytes": 0, "size_mb": 0.0}
    sz = path.stat().st_size
    return {
        "exists": True,
        "path": str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path),
        "size_bytes": sz,
        "size_mb": round(sz / (1024 * 1024), 2),
    }


# =============================================================================
# 4. STREAMLIT DATASET REQUIREMENT & FILE UPLOADER FALLBACK (SECTION 4)
#    Loads `data/raw/online_shoppers_intention.csv` automatically if present;
#    if missing, displays a friendly empty state and file uploader.
# =============================================================================
def require_dataset() -> pd.DataFrame:
    """Return the active DataFrame in Streamlit (from disk or user upload if disk file is missing)."""
    if RAW_CSV_PATH.exists():
        return load_raw_csv(str(RAW_CSV_PATH))

    if "uploaded_raw_df" in st.session_state:
        return st.session_state["uploaded_raw_df"]

    ui.empty_state(
        "UCI Online Shoppers Dataset Not Found at data/raw/online_shoppers_intention.csv",
        "Place <code>online_shoppers_intention.csv</code> inside <code>data/raw/</code> or upload the official CSV below.",
        "📂",
    )
    uploaded = st.file_uploader("Upload online_shoppers_intention.csv", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        val = data_validation.validate_dataset(df)
        if not val["is_valid_schema"]:
            st.error(f"Uploaded CSV is missing required columns: {val['missing_columns']}")
            st.stop()
        st.session_state["uploaded_raw_df"] = df
        st.rerun()
    st.stop()
