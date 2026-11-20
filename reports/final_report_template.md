# E-commerce Purchase Behavior Prediction and Customer Segmentation Using Machine Learning – Final Report

**Class:** Y3S1 · **Team:** Tola Long (Team Leader), Team Member 2, Team Member 3, Team Member 4 · **Date:** November 2026

---

## 1. Business Understanding

- **Problem:** Online retailers record millions of anonymous browsing interactions, yet fewer than 1% of all sessions—and only ~7.6% of engaged multi-event sessions—result in a purchase. Furthermore, treating all visitors identically wastes marketing budget on casual bouncers while failing to support hesitant cart builders.
- **Users:** E-commerce marketing analysts, personalization engines, and store managers.
- **Goal:**
  1. **Supervised Learning (Purchase Prediction):** Predict early in an active shopping session (after the first 3 non-transaction interactions) whether the session will result in a transaction (`will_purchase` = `Yes` / `No`).
  2. **Unsupervised Learning (Customer Segmentation):** Group engaged visitors into actionable behavioral personas using K-Means clustering to tailor promotions and UX interventions.
- **Success Criteria:** Achieve a held-out test **ROC-AUC $\ge 0.80$** and **Recall $\ge 0.60$** on early-session purchase prediction, and discover interpretable K-Means customer segments with **Silhouette Score $\ge 0.30$**.

## 2. Data Understanding *(notebook 01)*

- **Source:** Retailrocket Recommender System Dataset (Kaggle), collected from a real e-commerce storefront over 4.5 months (`2015-05-03` to `2015-09-18`).
- **Size:** `2,756,101` raw events across `1,407,580` unique visitors and `235,061` distinct items (`data/raw/events.csv`).
- **Data Quality Issues Found:**
  - `460` exact duplicate rows.
  - Timestamps were not sorted chronologically.
  - `transactionid` is empty on `99.19%` of rows by design (populated only on `transaction` events).
  - No native `session_id` column exists in the raw file.
- **Key EDA Findings:**
  - `view` accounts for `96.67%` (`2,664,312`) of events, `addtocart` for `2.52%` (`69,332`), and `transaction` for `0.81%` (`22,457`).
  - `83.9%` of sessions that add an item to the cart abandon it without purchasing in that session.
  - Among sessions reaching at least 3 non-transaction interactions with subsequent activity, the conversion rate is `7.59%` (imbalanced binary target).

## 3. Data Preparation *(notebook 02, `ml/data/make_dataset.py`, `ml/features.py`)*

| Problem | Rows Affected | Decision | Reason |
|---|---|---|---|
| Exact duplicate rows | `460` | Removed via `drop_duplicates()` | Prevent double-counting repeated logging events |
| Unsorted timestamps | `2,755,641` | Sorted by `(visitorid, timestamp)` | Required for accurate inter-event time gaps |
| Missing session boundaries | `1,761,675` sessions | Engineered 30-min inactivity timeout | Standard e-commerce session definition |
| Post-purchase leakage risk | `1,574` sessions | Excluded sessions purchasing at/before event 3 | Prevent outcome leakage into input features |

- **Engineered Features (`ml/features.py → add_features()`):**
  - `cart_to_view_ratio = add_to_cart_count / max(views_count, 1)`
  - `item_focus_ratio = 1 - (unique_items_viewed / max(views_count, 1))`
  - `avg_seconds_per_action = elapsed_seconds / 2`
  - `is_returning_visitor = (prior_sessions > 0).astype(int)`
- **Leakage Prevention:** All imputation (`SimpleImputer`), scaling (`StandardScaler`), one-hot encoding (`OneHotEncoder`), and feature transformations (`FunctionTransformer(add_features)`) live inside the Scikit-learn `Pipeline` and are fitted strictly on the training set. Identifier columns (`session_id`, `visitorid`, `session_start_ts`) are dropped via `DROP_COLUMNS`.

## 4. Modelling *(notebook 03, `ml/train.py`, `ml/clustering.py`)*

### 4.1 Supervised Classification (5-Fold Stratified Cross-Validation on 24,000 Training Rows)

| Model | CV F1 (mean ± std) | CV ROC-AUC (mean ± std) | CV Recall (mean ± std) |
|---|---|---|---|
| **Logistic Regression (Winner)** | **0.4266 ± 0.0087** | **0.8054 ± 0.0094** | **0.6284 ± 0.0177** |
| Random Forest | 0.4227 ± 0.0117 | 0.8034 ± 0.0078 | 0.5944 ± 0.0195 |
| Gradient Boosting | 0.3466 ± 0.0240 | 0.8152 ± 0.0079 | 0.2338 ± 0.0200 |
| Decision Tree | 0.3092 ± 0.0164 | 0.7850 ± 0.0086 | 0.6789 ± 0.0446 |

- **Best Tuned Model:** `Logistic Regression` (`C = 3.0`, `class_weight = "balanced"`).

### 4.2 Unsupervised Customer Segmentation (K-Means on 25,000 Engaged Customers)

- **Preprocessing:** Median imputation $\rightarrow$ `np.log1p` transformation on right-skewed count/duration features $\rightarrow$ `StandardScaler`.
- **Selected $K = 4$:** Achieved **Silhouette Score = `0.3413`** and **Davies–Bouldin Index = `1.196`**, discovering 4 behavioral personas (*Quick Bouncers / Casual Browsers*, *Focused Everyday Shoppers*, *Deep Window Explorers*, and *Hesitant Cart Abandoners*).

## 5. Evaluation *(notebook 04, `models/model_card.md`)*

- **Held-Out Test Set Performance (6,000 unseen sessions):**
  - **ROC-AUC:** `0.8165` (exceeds our `0.80` success criterion)
  - **PR-AUC:** `0.3758` (~5× the `0.0758` random baseline prevalence)
  - **Recall:** `0.6462` (catches **64.6%** of all future buyers after just 3 interactions)
  - **Precision:** `0.3118` (~4.1× lift over random targeting)
  - **F1-Score:** `0.4206`
  - **Accuracy:** `0.8650`
- **Charts Generated in `reports/figures/`:** `model_comparison.png`, `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`, `elbow_silhouette.png`, `clusters_pca.png`.

## 6. Deployment *(FastAPI Backend + 8-Page Streamlit & React Frontend)*

- **Backend (`backend_fastapi/main.py`):** Exposes `/api/health`, `/api/schema`, `/api/data`, `/api/metrics`, `/api/train`, `/api/predict`, `/api/predictions` (persisted in SQLite `data/predictions.db`), `/api/segments`, `/api/segment/predict`, and `/api/insights`.
- **Frontend (`frontend/app.py`):** 8 pages (**Home Dashboard**, **Data Explorer**, **Purchase Prediction**, **Customer Segmentation**, **Model Performance**, **Prediction History**, **Customer Insights**, **About Project**) with integrated React + TypeScript + Tailwind CSS components (`frontend/components/react_dist/`).

## 7. Conclusion & Limitations

- **Achievements:** Built a reproducible, leakage-safe CRISP-DM pipeline combining early-session conversion prediction (`ROC-AUC = 0.8165`) and K-Means behavioral customer segmentation (`Silhouette = 0.3413`), deployed via FastAPI and an 8-page Streamlit + React application.
- **Limitations:**
  1. Applies to sessions reaching at least 3 interactions; 1-click bounce sessions cannot be scored via an early 3-event window.
  2. Retailrocket does not log price, demographic, or cart-removal events, so cart abandonment is measured at the session level.

## References

- Retailrocket Recommender System Dataset (Kaggle): https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide.*
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
