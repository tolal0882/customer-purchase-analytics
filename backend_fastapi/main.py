"""
FastAPI backend.  Run from the PROJECT ROOT folder:

    uvicorn backend_fastapi.main:app --reload --port 8000

Interactive API docs: http://localhost:8000/docs
"""
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml import service

app = FastAPI(
    title="E-commerce Purchase Behavior Prediction & Customer Segmentation API",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: dict[str, Any]


class TrainRequest(BaseModel):
    model_name: str = "all"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    selected_features: list[str] | None = None
    balanced: bool = True
    sampling: str = "none"
    threshold: float = 0.5
    split_mode: str = "chronological"
    hyperparams: dict[str, Any] | None = None


class ActivateExperimentRequest(BaseModel):
    experiment_id: str


class PreprocessRequest(BaseModel):
    timeout_minutes: float = 30.0


class SegmentFitRequest(BaseModel):
    n_clusters: int = 4
    scope: str = "customer"
    features: list[str] | None = None
    scaler: str = "standard"
    use_log1p: bool = True
    random_state: int = 42


class SegmentPredictRequest(BaseModel):
    features: dict[str, Any]


@app.get("/api/health")
def health():
    return {"status": "ok", "backend": "fastapi"}


@app.get("/api/schema")
def schema():
    return service.get_schema()


@app.get("/api/data")
def data(limit: int = 100, dataset: str = "modeling"):
    return service.get_data(limit=limit, dataset=dataset)


@app.post("/api/preprocess")
def preprocess(req: PreprocessRequest):
    try:
        return service.run_preprocessing(timeout_minutes=req.timeout_minutes)
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/api/metrics")
def metrics():
    report = service.get_metrics()
    if report is None:
        raise HTTPException(404, "Model is not trained yet.")
    return report


@app.post("/api/train")
def train(req: TrainRequest | None = None):
    params = req.model_dump() if req is not None else None
    return service.train_model(params=params)


@app.post("/api/experiments/activate")
def activate_experiment(req: ActivateExperimentRequest):
    try:
        return service.activate_experiment(req.experiment_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


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


@app.get("/api/segments")
def segments(
    k: int | None = None,
    scope: str = "customer",
    scaler: str = "standard",
    use_log1p: bool = True,
    random_state: int = 42,
    features: list[str] | None = Query(default=None),
):
    return service.get_segments(
        n_clusters=k,
        scope=scope,
        features=features,
        scaler=scaler,
        use_log1p=use_log1p,
        random_state=random_state,
    )


@app.post("/api/segments/fit")
def fit_segments(req: SegmentFitRequest):
    return service.get_segments(
        n_clusters=req.n_clusters,
        scope=req.scope,
        features=req.features,
        scaler=req.scaler,
        use_log1p=req.use_log1p,
        random_state=req.random_state,
    )


@app.post("/api/segment/predict")
def predict_segment(req: SegmentPredictRequest):
    return service.predict_customer_segment(req.features)


@app.get("/api/insights")
def insights():
    return service.get_insights()
