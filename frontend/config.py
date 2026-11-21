"""
Project metadata and 5-member team task allocation (5 tasks per member = 25 tasks)
shown on the Dashboard and System Architecture / About pages.
"""
PROJECT_TITLE = "E-commerce Purchase Behavior Prediction & Customer Segmentation"
PROJECT_SUBTITLE = "An Interactive Data Science Application Using Streamlit, FastAPI, and React"
PROJECT_ICON = "🛒"
PROJECT_DESCRIPTION = (
    "A CRISP-DM Machine Learning application built on the Retailrocket E-commerce Dataset "
    "that predicts early-session purchase conversion (Supervised Learning) and groups customers "
    "into actionable behavioral segments using K-Means (Unsupervised Learning)."
)
CLASS_NAME = "Y3S1 – Web Application Development / Data Science"

TEAM_MEMBERS = [
    "Long Tola (Team Leader – Full-Stack Architecture, FastAPI, SQLite & Cloud Deployment)",
    "Seng Ranut (Data Engineering, Preprocessing & Session Construction)",
    "Phat Vandet (Exploratory Data Analysis, Feature Engineering & Business Insights)",
    "Vy Soksamnang (Supervised Machine Learning – Purchase Prediction & Model Evaluation)",
    "Yos Sak (Unsupervised Machine Learning – K-Means Segmentation & React UI Components)",
]

TEAM_TASK_ASSIGNMENTS = [
    {
        "member": "1. Long Tola (Team Leader)",
        "role": "System Architect, FastAPI Backend, SQLite Persistence & Cloud Deployment",
        "files": "ml/config.py, ml/service.py, backend_fastapi/main.py, frontend/app.py, frontend/api_client.py, tests/test_pipeline.py",
        "tasks": [
            "Task 1.1: Design the end-to-end CRISP-DM system architecture and central configuration (`ml/config.py`, `frontend/config.py`).",
            "Task 1.2: Implement the FastAPI REST backend (`backend_fastapi/main.py`) with 12 endpoints (`/api/health`, `/schema`, `/data`, `/preprocess`, `/train`, `/predict`, `/predictions`, `/segments`, etc.).",
            "Task 1.3: Build the unified ML service layer and persistent SQLite prediction history database (`ml/service.py`, `data/predictions.db`).",
            "Task 1.4: Develop the hybrid API client with automatic direct-service cloud fallback (`frontend/api_client.py`) and the Prediction History & System Architecture pages (`frontend/views/history.py`, `frontend/views/about.py`).",
            "Task 1.5: Lead automated quality assurance (`tests/test_pipeline.py` – 13 pytest tests) and deploy the application to GitHub & Streamlit Community Cloud.",
        ],
    },
    {
        "member": "2. Seng Ranut",
        "role": "Data Engineer — Raw Data Validation, Cleaning & Session Construction",
        "files": "ml/data/make_dataset.py, notebooks/02_data_cleaning.ipynb, frontend/views/data_explorer.py",
        "tasks": [
            "Task 2.1: Audit and validate the 4 raw Retailrocket dataset files (`events.csv` 2.75M rows, `category_tree.csv`, `item_properties_part1/part2.csv`).",
            "Task 2.2: Implement data cleaning in `ml/data/make_dataset.py` (`clean_events`) to remove 460 exact duplicate events, validate UTC timestamps, and optimize memory types.",
            "Task 2.3: Engineer session boundaries using a configurable inactivity timeout threshold (default 30 minutes → 1,761,675 sessions) and compute timeout sensitivity (15 / 30 / 60 min).",
            "Task 2.4: Construct the leakage-safe early-session snapshot builder (`build_modeling_snapshots` → `data/processed/clean.csv`) with eligibility filtering.",
            "Task 2.5: Author `notebooks/02_data_cleaning.ipynb` and build the Data Explorer & Preprocessing Pipeline UI (`frontend/views/data_explorer.py`).",
        ],
    },
    {
        "member": "3. Phat Vandet",
        "role": "Data Analyst & Feature Engineer — EDA, In-Pipeline Features & Insights",
        "files": "ml/features.py, notebooks/01_eda.ipynb, frontend/views/home.py, frontend/views/insights.py",
        "tasks": [
            "Task 3.1: Conduct Exploratory Data Analysis in `notebooks/01_eda.ipynb` covering event distribution, conversion funnel, and 92.4% class imbalance.",
            "Task 3.2: Engineer in-pipeline behavioral ratio features in `ml/features.py` (`cart_to_view_ratio`, `cart_interaction_ratio`, `has_early_cart`, `item_focus_ratio`, `avg_seconds_per_action`, `is_returning_visitor`, `historical_buy_rate`).",
            "Task 3.3: Build the Scikit-learn `ColumnTransformer` preprocessing pipeline (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) inside `ml/features.py`.",
            "Task 3.4: Compute dataset-wide analytical aggregations (`build_insights_summary` → `data/processed/insights_summary.json`), including daily trends, within-session event transition matrix, and top 15 products.",
            "Task 3.5: Build the Executive Analytics Dashboard (`frontend/views/home.py`) and 6-tab Exploratory Analysis & Behavioral Insights page (`frontend/views/insights.py`).",
        ],
    },
    {
        "member": "4. Vy Soksamnang",
        "role": "Supervised ML Engineer — Purchase Behavior Prediction & Model Evaluation",
        "files": "ml/train.py, ml/evaluate.py, notebooks/03_modeling.ipynb, notebooks/04_evaluation.ipynb, frontend/views/model_performance.py, frontend/views/prediction.py",
        "tasks": [
            "Task 4.1: Implement the 4 supervised classification algorithms in `ml/train.py` (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) with chronological and stratified splitting.",
            "Task 4.2: Implement `TrainingFoldResampler` in `ml/train.py` for balanced class weighting and training-fold under/over-sampling with zero test-fold data leakage.",
            "Task 4.3: Build the evaluation suite in `ml/evaluate.py` and `notebooks/04_evaluation.ipynb` (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, ROC & PR curves, feature importance).",
            "Task 4.4: Implement local feature-perturbation prediction explanations (`make_prediction` in `ml/service.py`) and the interactive Purchase Prediction page (`frontend/views/prediction.py`).",
            "Task 4.5: Build the Interactive Model Training & Evaluation Studio and Saved Experiments Registry (`frontend/views/model_performance.py`, `models/experiments.json`).",
        ],
    },
    {
        "member": "5. Yos Sak",
        "role": "Unsupervised ML & Frontend UI Engineer — K-Means Segmentation & React Components",
        "files": "ml/clustering.py, frontend/views/segmentation.py, frontend/components/ui.py, frontend/components/react_bridge.py",
        "tasks": [
            "Task 5.1: Aggregate customer-level (`customer_profiles.csv`) and session-level (`session_profiles.csv`) behavioral profiles for unsupervised learning.",
            "Task 5.2: Implement K-Means clustering in `ml/clustering.py` with `np.log1p` skewness transformation, configurable scalers (`Standard`, `Robust`, `MinMax`), and Elbow/Silhouette evaluation across `K = 2..8`.",
            "Task 5.3: Implement 2-D PCA cluster projection, automated z-score persona naming, and post-clustering purchase conversion analysis in `ml/clustering.py`.",
            "Task 5.4: Integrate the compiled React 18 + TypeScript + Tailwind components (`PredictionForm`, `ClusterExplorer`, `ModelCompare`) via `frontend/components/react_bridge.py` and the SaaS design system (`frontend/components/ui.py`).",
            "Task 5.5: Build the dual-scope Customer & Session Segmentation page (`frontend/views/segmentation.py`) with interactive K-Means refitting, segment assignment, and CSV export.",
        ],
    },
]

DATA_SOURCE = (
    "Retailrocket Recommender System Dataset (Kaggle) – 2,756,101 real e-commerce clickstream "
    "events (`view`, `addtocart`, `transaction`) across 1,407,580 visitors."
)
