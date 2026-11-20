# E-commerce Purchase Behavior Prediction and Customer Segmentation Using Machine Learning

An interactive Data Science application built with **Streamlit** and **React + TypeScript + Tailwind CSS** for analyzing the **Retailrocket E-commerce Dataset** using the **CRISP-DM** methodology.

## Key Capabilities

- **10 Complete Application Pages:** Dashboard, Dataset Explorer, Data Preprocessing, Exploratory Data Analysis (6 analytical sections), Customer & Session Segmentation, Model Training, Model Evaluation, Purchase Prediction, Prediction History, and About Project.
- **Hybrid Frontend:** Streamlit is the primary framework, integrated with custom React + TypeScript + Tailwind CSS components (`PredictionForm`, `ClusterExplorer`, `ModelCompare`) built with Vite and served via `streamlit.components.v1.declare_component`.
- **Leakage-Safe Machine Learning:**
  - Early-session prediction snapshots (default: after the 3rd non-transaction event; configurable time-based alternative).
  - Chronological train/test split and expanding-window `TimeSeriesSplit` cross-validation.
  - All preprocessing (`SimpleImputer`, `StandardScaler`) and optional training-fold resampling wrapped inside a single Scikit-learn pipeline (`ResamplingClassifier`).
  - Explicit guards blocking `visitorid`, `itemid`, `transactionid`, and `transaction_count` from model inputs.
- **Behavioural Segmentation:** K-Means clustering for both customers (visitors) and sessions, with Elbow Method, Silhouette analysis, 2-D PCA visualization, z-score-derived cluster names, and post-clustering purchase-rate comparison.

## Project Structure

```text
customer-purchase-analytics/
├── streamlit_app.py              # Main Streamlit entry point
├── requirements.txt
├── pytest.ini
├── config/settings.py            # Paths, defaults, design system tokens
├── app_pages/                    # 10 Streamlit pages
├── src/                          # Pure Python data, ML, analytics & chart modules
├── components/
│   ├── streamlit_components/     # Reusable Streamlit UI + React component wrappers
│   └── react/                    # Vite + React + TypeScript + Tailwind CSS components
├── data/
│   ├── raw/ (or row/)            # Original Retailrocket CSVs (never modified)
│   ├── processed/                # Cached parquet/CSV processed tables
│   └── history/                  # Local prediction history log (CSV)
├── models/                       # Saved joblib pipelines & clustering bundles
├── experiments/                  # Saved experiment metadata (JSON)
└── tests/                        # Pytest unit, leakage, ML & real-data tests
```

## Dataset Setup

Download the [Retailrocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) from Kaggle and place the files into `data/raw/` (the app also automatically detects `data/row/` if already present):

- `events.csv` (required)
- `category_tree.csv` (optional)
- `item_properties_part1.csv` & `item_properties_part2.csv` (optional enrichment; streamed in chunks only when enabled)

## Installation & Execution

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build React components (optional if `components/react/dist` is already built)

```bash
cd components/react
npm install
npm run build
cd ../..
```

*(If `components/react/dist` is absent, the app automatically falls back to native Streamlit controls.)*

### 3. Run the application

```bash
.venv/bin/streamlit run streamlit_app.py
```

### 4. Run tests

```bash
.venv/bin/python -m pytest
```

## Methodological Notes & Limitations

- **Session IDs** are not provided by Retailrocket and are engineered via a configurable inactivity timeout (default: 30 minutes).
- **Cart removals** (`removefromcart`) do not exist in the Retailrocket dataset; `historical_cart_abandonment_rate` is defined as the share of cart sessions without a transaction in the same session.
- **Imbalanced target:** Only ~7.5% of eligible 3-event prediction snapshots result in a transaction; always evaluate models using PR-AUC, Recall, Precision, and F1 rather than accuracy alone.
