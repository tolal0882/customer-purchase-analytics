"""
Streamlit Community Cloud Entrypoint (`frontend/app.py`):
Reloads `src.ui` and `streamlit_app` on every run so new commits take effect immediately
even when `fileWatcherType = "none"`, then executes `streamlit_app.main()`.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.ui as _ui_mod
import streamlit_app as _app_mod

importlib.reload(_ui_mod)
importlib.reload(_app_mod)

_app_mod.main()
