"""
FastAPI backend.  Run from the PROJECT ROOT folder:

    uvicorn backend_fastapi.main:app --reload --port 8000

Interactive API docs: http://localhost:8000/docs
"""
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml import service

app = FastAPI(
    title="E-commerce Purchase Behavior Prediction & Customer Segmentation API",
    version="1.0",
)

# =============================================================================
# 1. CORS CONFIGURATION
#    - Allows the Streamlit frontend (and React components) to call this API.
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 2. REQUEST SCHEMAS (PYDANTIC MODELS)
# =============================================================================
class PredictRequest(BaseModel):
    features: dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "views_count": 2,
                    "unique_items_viewed": 2,
                    "add_to_cart_count": 1,
                    "elapsed_seconds": 90.0,
                    "ends_with_cart": "Yes",
                    "start_hour": 14,
                    "is_weekend": "No",
                    "prior_sessions": 1,
                    "prior_purchases": 0,
                }
            }
        }
    }


class SegmentPredictRequest(BaseModel):
    features: dict[str, Any]


# =============================================================================
# 3. CORE TEACHER TEMPLATE ENDPOINTS (SUPERVISED LEARNING + SQLITE HISTORY)
# =============================================================================
@app.get("/api/health")
def health():
    return {"status": "ok", "backend": "fastapi"}


@app.get("/api/schema")
def schema():
    return service.get_schema()


@app.get("/api/data")
def data(limit: int = 100, dataset: str = "modeling"):
    return service.get_data(limit=limit, dataset=dataset)


@app.get("/api/metrics")
def metrics():
    report = service.get_metrics()
    if report is None:
        raise HTTPException(404, "Model is not trained yet.")
    return report


@app.post("/api/train")
def train():
    return service.train_model()


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        result = service.make_prediction(req.features)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    service.log_prediction(req.features, result)
    return result


@app.get("/api/predictions")
def prediction_history(limit: int = 100):
    return {"items": service.get_predictions(limit=limit)}


@app.delete("/api/predictions")
def clear_prediction_history():
    deleted = service.clear_predictions()
    return {"deleted": deleted}


# =============================================================================
# 4. CUSTOMER SEGMENTATION (K-MEANS) & BUSINESS INSIGHTS ENDPOINTS
# =============================================================================
@app.get("/api/segments")
def segments(k: int | None = None):
    return service.get_segments(n_clusters=k)


@app.post("/api/segment/predict")
def predict_segment(req: SegmentPredictRequest):
    return service.predict_customer_segment(req.features)


@app.get("/api/insights")
def insights():
    return service.get_insights()
