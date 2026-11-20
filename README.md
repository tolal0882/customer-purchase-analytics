---
title: Retail Purchase Analytics
emoji: 🛒
colorFrom: green
colorTo: gray
sdk: streamlit
sdk_version: 1.44.0
app_file: frontend/app.py
pinned: false
---

# E-commerce Purchase Behavior Prediction and Customer Segmentation Using Machine Learning

*An Interactive Data Science Application Using Streamlit, FastAPI, and React*

Built by adapting our teacher's **CRISP-DM Data Science Web App Template**:
**raw data (`events.csv`) ➜ cleaning (`make_dataset.py`) ➜ EDA & notebooks (`01..04`) ➜ supervised + unsupervised modelling (`train.py` & `clustering.py`) ➜ evaluation (`evaluate.py`) ➜ FastAPI (`backend_fastapi/`) ➜ 8-page Streamlit + React web app (`frontend/`).**

```text
┌────────────────────┐   HTTP / JSON   ┌──────────────────────┐   Python calls   ┌─────────────────────────┐
│ Streamlit + React  │ ──────────────▶ │   FastAPI Backend    │ ───────────────▶ │   ml/ Pipeline Layer    │
│ frontend/app.py    │ ◀────────────── │ backend_fastapi/     │ ◀─────────────── │ models/model.joblib     │
│ (8 views)          │                 │ + SQLite history     │                  │ models/clustering.joblib│
└────────────────────┘                 └──────────────────────┘                  └─────────────────────────┘
```

---

## 1. The Data-Science Workflow (CRISP-DM)

| # | Phase | Where | Output |
|---|---|---|---|
| 1 | Business understanding | `reports/final_report_template.md`, `frontend/views/about.py` | Problem, objectives, success criteria |
| 2 | Data understanding | `notebooks/01_eda.ipynb` | Data quality audit, funnel & target distribution |
| 3 | Data preparation | `notebooks/02_data_cleaning.ipynb` ➜ `ml/data/make_dataset.py`, `ml/features.py` | `data/processed/clean.csv`, `customer_profiles.csv` |
| 4 | Modelling | `notebooks/03_modeling.ipynb` ➜ `ml/train.py`, `ml/clustering.py` | 4 classifiers (5-fold CV + tuned) & K-Means ($K=2\dots8$) |
| 5 | Evaluation | `notebooks/04_evaluation.ipynb` ➜ `ml/evaluate.py` | `models/metrics.json`, `model_card.md`, `reports/figures/*.png` |
| 6 | Deployment | `backend_fastapi/main.py`, `frontend/app.py` (8 pages) | Working full-stack web application |

---

## 2. Project Structure (Teacher's Template Architecture)

```text
customer-purchase-analytics/
├── data/
│   ├── raw/events.csv                # Original Retailrocket clickstream dataset (2.75M rows, read-only)
│   ├── processed/                    # clean.csv, customer_profiles.csv, insights_summary.json
│   ├── predictions.db                # SQLite persistent prediction history
│   └── README.md                     # Retailrocket data dictionary
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_data_cleaning.ipynb        # Duplicate removal, 30-min sessions & snapshot engineering
│   ├── 03_modeling.ipynb             # 4-classifier CV comparison, tuning & K-Means Elbow/Silhouette
│   └── 04_evaluation.ipynb           # Held-out test evaluation, ROC/PR curves & cluster profiling
├── ml/
│   ├── config.py                     # Paths, target ("will_purchase"), features & hyperparameters
│   ├── data/make_dataset.py          # Raw events.csv ➜ clean.csv & customer_profiles.csv
│   ├── features.py                   # In-pipeline FunctionTransformer + ColumnTransformer
│   ├── train.py                      # 5-fold Stratified CV, RandomizedSearchCV & model card generator
│   ├── clustering.py                 # K-Means customer segmentation (log1p + StandardScaler + PCA)
│   ├── evaluate.py                   # Test metrics & report figures
│   └── service.py                    # Unified service layer called by FastAPI (with SQLite history)
├── models/                           # model.joblib, clustering.joblib, metrics.json, model_card.md
├── reports/
│   ├── figures/                      # model_comparison.png, confusion_matrix.png, roc_curve.png,
│   │                                 # feature_importance.png, elbow_silhouette.png, clusters_pca.png
│   └── final_report_template.md      # Populated academic project report
├── backend_fastapi/                  # FastAPI REST backend (/api/health, /schema, /data, /metrics,
│                                     # /train, /predict, /predictions, /segments, /insights)
├── backend_django/                   # Optional Django REST backend from template
├── frontend/                         # 8-page Streamlit + React application
│   ├── app.py                        # Main Streamlit entrypoint
│   ├── config.py                     # Project title, description & team info
│   ├── api_client.py                 # HTTP client for FastAPI (with automatic direct-service fallback)
│   ├── components/                   # Compiled React + TypeScript + Tailwind bundle & bridge
│   └── views/                        # 8 pages: home, data_explorer, prediction, segmentation,
│                                     # model_performance, history, insights, about
├── tests/test_pipeline.py            # Pytest suite (cleaning, features, classification, clustering, API)
└── requirements.txt
```

---

## 3. Setup & Installation

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -r backend_fastapi/requirements.txt
pip install -r frontend/requirements.txt
```

Place the Retailrocket `events.csv` file from [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) into `data/raw/events.csv`.

---

## 4. Run the ML Pipeline (From Project Root)

```bash
python -m ml.data.make_dataset    # clean raw events.csv ➜ data/processed/clean.csv & customer_profiles.csv
python -m ml.train                # compare 4 classifiers (5-fold CV), tune best, run K-Means (K=4), save all
pytest                            # run automated test suite
```

Actual output of `python -m ml.train` on Retailrocket:

```text
Data: 24,000 train / 6,000 test rows, 9 input columns

1) Comparing models (5-fold CV on training set):
  Logistic Regression    f1 = 0.4266 ± 0.0087
  Random Forest          f1 = 0.4227 ± 0.0117
  Gradient Boosting      f1 = 0.3466 ± 0.0240
  Decision Tree          f1 = 0.3092 ± 0.0164

2) Tuning best model: Logistic Regression  (best params: {'C': 3.0})
3) Evaluating on the TEST set:
   accuracy 0.8650 · precision 0.3118 · recall 0.6462 · f1 0.4206 · roc_auc 0.8165 · pr_auc 0.3758
4) Running K-Means Customer Segmentation:
   K-Means (K=4) trained on 25,000 customers · Silhouette = 0.3413
```

---

## 5. Run the Web Application

**Terminal 1 – FastAPI Backend** (from project root):
```bash
.venv/bin/uvicorn backend_fastapi.main:app --reload --port 8000
```
*(Interactive Swagger docs: http://localhost:8000/docs)*

**Terminal 2 – Streamlit + React Frontend** (from project root):
```bash
.venv/bin/streamlit run frontend/app.py
```
*(App URL: http://localhost:8501 — Note: if FastAPI is not started separately, `frontend/api_client.py` automatically falls back to calling `ml.service` directly so single-command demos work seamlessly!)*
