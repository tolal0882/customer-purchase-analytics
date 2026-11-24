---
title: Online Shopping Purchase Intention & Session Segmentation
emoji: 🛒
colorFrom: green
colorTo: gray
sdk: streamlit
sdk_version: 1.44.0
app_file: streamlit_app.py
pinned: false
---

# Online Shopping Purchase Intention Prediction and Session Behavior Segmentation Using Machine Learning

*An Interactive Data Science Application Using Streamlit*

Built on the **UCI Online Shoppers Purchasing Intention Dataset** (`12,330` shopping sessions, `17` input features, and binary target `Revenue`), this academic Data Science application demonstrates two complete machine learning workflows following **CRISP-DM**:

1. **Purchase Intention Prediction (Supervised Learning / Binary Classification):**
   - Compares **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier** using an **80/20 Stratified Train/Test Split** and **Stratified 5-Fold Cross-Validation**.
   - Addresses class imbalance (`15.47%` purchase vs. `84.53%` non-purchase) with optional balanced class weights and evaluates **Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix, Classification Report, ROC Curve, and Precision–Recall Curve**.
   - Includes a dedicated **`PageValues` Investigation** comparing a 17-feature pipeline against a 16-feature pipeline excluding `PageValues`.
   - Persists fitted Scikit-learn `Pipeline` + `ColumnTransformer` bundles with **Joblib** (`models/classification/`) for live session purchase prediction and local prediction history logging.

2. **Session Behavior Segmentation (Unsupervised Learning / K-Means Clustering):**
   - Segments shopping sessions using **K-Means Clustering** on browsing activity, duration, bounce/exit rates, and special-day proximity (**strictly excluding `Revenue` from clustering inputs**).
   - Evaluates $K = 2 \dots 8$ using the **Elbow Method (Within-Cluster Inertia)** and **Silhouette Score**, projects sessions onto **2-D PCA**, generates data-driven segment interpretations from actual cluster centroids, and compares post-clustering purchase rates (`Revenue`) across segments strictly for interpretation.

---

## 1. Project Folder Structure

```text
customer-purchase-analytics/
├── streamlit_app.py                     # Main Streamlit entrypoint (6 pages)
├── requirements.txt                     # Minimal dependencies
├── README.md                            # Setup & methodological documentation
├── pytest.ini                           # Pytest configuration
│
├── app_pages/
│   ├── dashboard.py                     # PAGE 1: Dashboard (4 filters, 6 KPIs, 6 Plotly charts)
│   ├── dataset_eda.py                   # PAGE 2: Dataset Explorer & 4-Section EDA (A, B, C, D)
│   ├── preprocessing.py                 # PAGE 3: Data Preprocessing (Validation, duplicate toggle, pipelines, CSV export)
│   ├── purchase_prediction.py           # PAGE 4: Purchase Prediction (Training, PageValues study, Evaluation, Prediction, History)
│   ├── session_segmentation.py          # PAGE 5: Session Segmentation (K-Means, Elbow/Silhouette, 2D PCA, Profiles, CSV export)
│   └── about.py                         # PAGE 6: About Project (Objectives, UCI citation, CRISP-DM, live results, limitations)
│
├── src/
│   ├── ui.py                            # Forest Green (#153E2D / #2F855A / #A7C957 / #F3F7F3) design system
│   ├── data_loader.py                   # Cached CSV loading + file uploader fallback
│   ├── data_validation.py               # Schema, dtype, missing, duplicate & domain validation
│   ├── preprocessing.py                 # Explicit cleaning options & classification/clustering ColumnTransformers
│   ├── classification.py                # Logistic Regression, Decision Tree, Random Forest & PageValues study
│   ├── evaluation.py                    # Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, ROC/PR curves
│   ├── prediction.py                    # Single-session inference & local prediction history
│   ├── clustering.py                    # K-Means workflow, Elbow/Silhouette, 2D PCA, data-driven segment profiler
│   └── model_manager.py                 # Joblib persistence for classification & clustering bundles
│
├── data/
│   ├── raw/
│   │   └── online_shoppers_intention.csv  # Official UCI dataset (12,330 × 18, read-only)
│   └── processed/                       # Exported processed CSVs & local prediction_history.json
│
├── models/
│   ├── classification/                  # Saved Joblib classification pipelines & metadata JSON
│   └── clustering/                      # Saved Joblib K-Means clustering bundle & metadata JSON
│
└── tests/
    └── test_project.py                  # Automated pytest suite verifying all acceptance criteria
```

---

## 2. Installation & Local Execution

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install minimal dependencies
pip install -r requirements.txt

# 3. Run the automated test suite
pytest -v

# 4. Launch the 6-page Streamlit application
streamlit run streamlit_app.py
```

---

## 3. Dataset Citation & License

- **Source:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**)
- **Citation:**
  > Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset.* UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q.
