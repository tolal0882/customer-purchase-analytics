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
        "ui_features": "📜 Prediction History & ℹ️ System Architecture (+ Sidebar & Backend API)",
        "files": "ml/config.py, ml/service.py, backend_fastapi/main.py, frontend/app.py, frontend/api_client.py, frontend/views/history.py, frontend/views/about.py, tests/test_pipeline.py",
        "tasks": [
            "Task 1.1: Design the end-to-end CRISP-DM system architecture and central configuration (`ml/config.py`, `frontend/config.py`).",
            "Task 1.2: Implement the FastAPI REST backend (`backend_fastapi/main.py`) with 12 endpoints (`/api/health`, `/schema`, `/data`, `/preprocess`, `/train`, `/predict`, `/predictions`, `/segments`, etc.).",
            "Task 1.3: Build the unified ML service layer and persistent SQLite prediction history database (`ml/service.py`, `data/predictions.db`).",
            "Task 1.4: Develop the hybrid API client (`frontend/api_client.py`) and build the **📜 Prediction History** (`frontend/views/history.py`) & **ℹ️ System Architecture** (`frontend/views/about.py`) UI pages.",
            "Task 1.5: Lead automated quality assurance (`tests/test_pipeline.py` – 13 pytest tests) and deploy the application to GitHub & Streamlit Community Cloud.",
        ],
    },
    {
        "member": "2. Seng Ranut",
        "role": "Data Engineer — Raw Data Validation, Cleaning & Session Construction",
        "ui_features": "🗂️ Data & Preprocessing (Dataset Explorer, Session Timeout Builder & Data Dictionary)",
        "files": "ml/data/make_dataset.py, notebooks/02_data_cleaning.ipynb, frontend/views/data_explorer.py",
        "tasks": [
            "Task 2.1: Audit and validate the 4 raw Retailrocket dataset files (`events.csv` 2.75M rows, `category_tree.csv`, `item_properties_part1/part2.csv`).",
            "Task 2.2: Implement data cleaning in `ml/data/make_dataset.py` (`clean_events`) to remove 460 exact duplicate events, validate UTC timestamps, and optimize memory types.",
            "Task 2.3: Engineer session boundaries using a configurable inactivity timeout threshold (default 30 minutes → 1,761,675 sessions) and compute timeout sensitivity (15 / 30 / 60 min).",
            "Task 2.4: Construct the leakage-safe early-session snapshot builder (`build_modeling_snapshots` → `data/processed/clean.csv`) with eligibility filtering.",
            "Task 2.5: Author `notebooks/02_data_cleaning.ipynb` and build the **🗂️ Data & Preprocessing** UI page (`frontend/views/data_explorer.py`).",
        ],
    },
    {
        "member": "3. Phat Vandet",
        "role": "Data Analyst & Feature Engineer — EDA, In-Pipeline Features & Insights",
        "ui_features": "📊 Dashboard & 🔍 Exploratory Analysis (6-Tab EDA & Segment Playbook)",
        "files": "ml/features.py, notebooks/01_eda.ipynb, frontend/views/home.py, frontend/views/insights.py",
        "tasks": [
            "Task 3.1: Conduct Exploratory Data Analysis in `notebooks/01_eda.ipynb` covering event distribution, conversion funnel, and 92.4% class imbalance.",
            "Task 3.2: Engineer in-pipeline behavioral ratio features in `ml/features.py` (`cart_to_view_ratio`, `cart_interaction_ratio`, `has_early_cart`, `item_focus_ratio`, `avg_seconds_per_action`, `is_returning_visitor`, `historical_buy_rate`).",
            "Task 3.3: Build the Scikit-learn `ColumnTransformer` preprocessing pipeline (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) inside `ml/features.py`.",
            "Task 3.4: Compute dataset-wide analytical aggregations (`build_insights_summary` → `data/processed/insights_summary.json`), including daily trends, within-session event transition matrix, and top 15 products.",
            "Task 3.5: Build the **📊 Dashboard** (`frontend/views/home.py`) and 6-tab **🔍 Exploratory Analysis** UI page (`frontend/views/insights.py`).",
        ],
    },
    {
        "member": "4. Vy Soksamnang",
        "role": "Supervised ML Engineer — Purchase Behavior Prediction & Model Evaluation",
        "ui_features": "🧠 Model Training & Eval & 🎯 Purchase Prediction",
        "files": "ml/train.py, ml/evaluate.py, notebooks/03_modeling.ipynb, notebooks/04_evaluation.ipynb, frontend/views/model_performance.py, frontend/views/prediction.py",
        "tasks": [
            "Task 4.1: Implement the 4 supervised classification algorithms in `ml/train.py` (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) with chronological and stratified splitting.",
            "Task 4.2: Implement `TrainingFoldResampler` in `ml/train.py` for balanced class weighting and training-fold under/over-sampling with zero test-fold data leakage.",
            "Task 4.3: Build the evaluation suite in `ml/evaluate.py` and `notebooks/04_evaluation.ipynb` (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, ROC & PR curves, feature importance).",
            "Task 4.4: Implement local feature-perturbation prediction explanations (`make_prediction` in `ml/service.py`) and build the **🎯 Purchase Prediction** UI page (`frontend/views/prediction.py`).",
            "Task 4.5: Build the **🧠 Model Training & Eval** UI page and Saved Experiments Registry (`frontend/views/model_performance.py`, `models/experiments.json`).",
        ],
    },
    {
        "member": "5. Yos Sak",
        "role": "Unsupervised ML & Frontend UI Engineer — K-Means Segmentation & React Components",
        "ui_features": "👥 Customer Segmentation (+ React 18 / Tailwind Components & SaaS UI Design System)",
        "files": "ml/clustering.py, frontend/views/segmentation.py, frontend/components/ui.py, frontend/components/react_bridge.py",
        "tasks": [
            "Task 5.1: Aggregate customer-level (`customer_profiles.csv`) and session-level (`session_profiles.csv`) behavioral profiles for unsupervised learning.",
            "Task 5.2: Implement K-Means clustering in `ml/clustering.py` with `np.log1p` skewness transformation, configurable scalers (`Standard`, `Robust`, `MinMax`), and Elbow/Silhouette evaluation across `K = 2..8`.",
            "Task 5.3: Implement 2-D PCA cluster projection, automated z-score persona naming, and post-clustering purchase conversion analysis in `ml/clustering.py`.",
            "Task 5.4: Integrate the compiled React 18 + TypeScript + Tailwind components (`PredictionForm`, `ClusterExplorer`, `ModelCompare`) via `frontend/components/react_bridge.py` and the SaaS design system (`frontend/components/ui.py`).",
            "Task 5.5: Build the dual-scope **👥 Customer Segmentation** UI page (`frontend/views/segmentation.py`) with interactive K-Means refitting, segment assignment, and CSV export.",
        ],
    },
]

UI_PAGE_OWNERS = [
    {"UI Sidebar Menu": "📊 Dashboard", "Page File": "frontend/views/home.py", "Handled By": "Phat Vandet", "Key Features Handled": "Interactive date/event filters, 8 real Retailrocket KPIs, Event Funnel, Daily/Weekly trends, Top 15 items"},
    {"UI Sidebar Menu": "🗂️ Data & Preprocessing", "Page File": "frontend/views/data_explorer.py", "Handled By": "Seng Ranut", "Key Features Handled": "Dataset Explorer (search/filter/sort/paginate/CSV export), Cleaning Log, Interactive Session Timeout Builder, Data Dictionary"},
    {"UI Sidebar Menu": "🔍 Exploratory Analysis", "Page File": "frontend/views/insights.py", "Handled By": "Phat Vandet", "Key Features Handled": "6-Tab EDA (Events & Transition Matrix, Visitors, Sessions, Purchases & Cart Abandonment, Products, Segment Playbook)"},
    {"UI Sidebar Menu": "👥 Customer Segmentation", "Page File": "frontend/views/segmentation.py", "Handled By": "Yos Sak", "Key Features Handled": "Customer & Session K-Means Studio, Elbow & Silhouette charts, React ClusterExplorer cards, 2-D PCA scatter, CSV export"},
    {"UI Sidebar Menu": "🧠 Model Training & Eval", "Page File": "frontend/views/model_performance.py", "Handled By": "Vy Soksamnang", "Key Features Handled": "Held-Out Test Evaluation, React ModelCompare leaderboard, ROC/PR curves, Interactive Training Studio, Saved Experiments"},
    {"UI Sidebar Menu": "🎯 Purchase Prediction", "Page File": "frontend/views/prediction.py", "Handled By": "Vy Soksamnang (+ Yos Sak React Form)", "Key Features Handled": "Saved pipeline selector, React + Tailwind PredictionForm with presets, Purchase Likelihood Hero Banner, Local Feature Perturbation chart"},
    {"UI Sidebar Menu": "📜 Prediction History", "Page File": "frontend/views/history.py", "Handled By": "Long Tola (Team Leader)", "Key Features Handled": "SQLite persistent prediction log, outcome/model filters, CSV export, single record JSON inspector, clear confirmation"},
    {"UI Sidebar Menu": "ℹ️ System Architecture", "Page File": "frontend/views/about.py", "Handled By": "Long Tola (Team Leader)", "Key Features Handled": "5-Member Team Task Matrix (25 tasks), UI Feature Ownership Map, CRISP-DM Full-Stack Architecture"},
]

DATA_SOURCE = (
    "Retailrocket Recommender System Dataset (Kaggle) – 2,756,101 real e-commerce clickstream "
    "events (`view`, `addtocart`, `transaction`) across 1,407,580 visitors."
)
