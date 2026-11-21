# Team Task Allocation (5 members × 5 tasks)

Course: Y3S1 – Web Application Development / Data Science

| Member | Role | Main files |
|---|---|---|
| Long Tola (Leader) | Architecture, FastAPI, SQLite, deployment | `ml/config.py`, `ml/service.py`, `backend_fastapi/main.py`, `frontend/app.py`, `frontend/api_client.py`, `frontend/views/history.py`, `frontend/views/about.py`, `tests/test_pipeline.py` |
| Seng Ranut | Data engineering, cleaning, sessions | `ml/data/make_dataset.py`, `notebooks/02_data_cleaning.ipynb`, `frontend/views/data_explorer.py` |
| Phat Vandet | EDA, feature engineering, insights | `ml/features.py`, `notebooks/01_eda.ipynb`, `frontend/views/home.py`, `frontend/views/insights.py` |
| Vy Soksamnang | Supervised ML, evaluation, prediction | `ml/train.py`, `ml/evaluate.py`, `notebooks/03_modeling.ipynb`, `notebooks/04_evaluation.ipynb`, `frontend/views/model_performance.py`, `frontend/views/prediction.py` |
| Yos Sak | K-Means segmentation, React/UI components | `ml/clustering.py`, `frontend/views/segmentation.py`, `frontend/components/ui.py`, `frontend/components/react_bridge.py` |

## Long Tola
1. System architecture and central configuration.
2. FastAPI REST backend (12 endpoints).
3. Service layer and SQLite prediction history.
4. API client with direct-service fallback; Prediction History and System Overview pages.
5. Pytest suite (13 tests) and deployment to GitHub / Streamlit Cloud.

## Seng Ranut
1. Audit of the four raw Retailrocket files.
2. Cleaning pipeline (460 duplicates removed, timestamp validation, memory optimisation).
3. Session construction with configurable timeout and sensitivity (15/30/60 min).
4. Leakage-safe early-session snapshot builder (`clean.csv`).
5. Notebook 02 and the Data & Preprocessing page.

## Phat Vandet
1. Exploratory analysis notebook (event mix, funnel, class imbalance).
2. Behavioral ratio features in `ml/features.py`.
3. Scikit-learn preprocessing pipeline (imputer, scaler, encoder).
4. Dataset-wide aggregations (`insights_summary.json`).
5. Dashboard and Exploratory Analysis pages.

## Vy Soksamnang
1. Four classifiers with chronological / stratified split.
2. `TrainingFoldResampler` (class weights and in-fold resampling without leakage).
3. Evaluation suite and notebooks 03–04.
4. Prediction service with local explanations and the Purchase Prediction page.
5. Model Training & Evaluation page and experiment registry.

## Yos Sak
1. Customer and session behavioral profiles.
2. K-Means with log1p, scalers, Elbow / Silhouette over K = 2..8.
3. PCA projection, z-score profile naming, post-clustering purchase rate.
4. React + Tailwind components via `react_bridge.py` and the UI design system.
5. Customer Segmentation page.
