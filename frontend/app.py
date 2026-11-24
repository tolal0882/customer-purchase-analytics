"""
Compatibility wrapper so any cloud deployment configured with `frontend/app.py`
as the entrypoint runs the root `streamlit_app.py` seamlessly.
"""
from pathlib import Path
import runpy
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

runpy.run_path(str(ROOT_DIR / "streamlit_app.py"), run_name="__main__")
