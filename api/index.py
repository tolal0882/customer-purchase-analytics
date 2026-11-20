"""
Vercel ASGI entrypoint for the FastAPI backend (`backend_fastapi/main.py`).
Allows deploying the REST API (/api/health, /api/schema, /api/data, /api/metrics,
/api/predict, /api/segments, /api/insights) on Vercel or any ASGI host.
"""
from backend_fastapi.main import app

__all__ = ["app"]
