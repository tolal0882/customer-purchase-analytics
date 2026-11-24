"""
Streamlit Community Cloud Entrypoint (`frontend/app.py`):
Directly configures the 6-page navigation and executes `streamlit_app.py` logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit_app import main

main()
