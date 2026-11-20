"""
Project configuration – ONE place to change settings.

Adapted from the teacher's template for:
"E-commerce Purchase Behavior Prediction and Customer Segmentation Using Machine Learning"
Dataset: Retailrocket E-commerce Clickstream Dataset (data/raw/events.csv)
"""
from pathlib import Path

# =============================================================================
# 1. PROJECT PATHS
#    - Keeps every path from the teacher's template (RAW_DATA_PATH,
#      PROCESSED_DATA_PATH, MODEL_PATH, METRICS_PATH, MODEL_CARD_PATH, FIGURES_DIR)
#    - Adds paths for Customer Segmentation (K-Means) and SQLite history.
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "events.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "clean.csv"
CUSTOMER_PROFILES_PATH = ROOT_DIR / "data" / "processed" / "customer_profiles.csv"
INSIGHTS_SUMMARY_PATH = ROOT_DIR / "data" / "processed" / "insights_summary.json"
PREDICTIONS_DB_PATH = ROOT_DIR / "data" / "predictions.db"

MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
MODEL_CARD_PATH = ROOT_DIR / "models" / "model_card.md"
CLUSTERING_MODEL_PATH = ROOT_DIR / "models" / "clustering.joblib"
CLUSTERING_METRICS_PATH = ROOT_DIR / "models" / "clustering_metrics.json"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# =============================================================================
# 2. SUPERVISED LEARNING PROBLEM DEFINITION (PURCHASE PREDICTION)
#    - Binary classification: given a customer's early-session browsing actions
#      (first 3 non-transaction events) + prior session history, predict whether
#      the session will result in a transaction ("Yes" / "No").
# =============================================================================
TASK_TYPE = "classification"

TARGET_COLUMN = "will_purchase"
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"

# Columns that must NOT be used as model inputs (IDs, timestamps, leakage prevention)
DROP_COLUMNS = ["session_id", "visitorid", "session_start_ts"]

# =============================================================================
# 3. RETAILROCKET SESSION & SNAPSHOT ENGINEERING SETTINGS
# =============================================================================
VALID_EVENTS = ("view", "addtocart", "transaction")
SESSION_TIMEOUT_MINUTES = 30.0   # New session starts after 30 minutes of inactivity
SNAPSHOT_EVENTS = 3              # Early-session observation window (first 3 non-tx events)
MAX_MODELING_ROWS = 30_000       # Stratified chronological sample for fast CV & tuning
MAX_CUSTOMER_PROFILES = 25_000   # Engaged customer profiles for K-Means segmentation

# =============================================================================
# 4. TRAINING & CLUSTERING HYPERPARAMETERS
# =============================================================================
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42
PRIMARY_METRIC = "f1" if TASK_TYPE == "classification" else "neg_root_mean_squared_error"
TUNING_ITERATIONS = 10

# Unsupervised K-Means Customer Segmentation settings
DEFAULT_K_CLUSTERS = 4
K_RANGE = (2, 3, 4, 5, 6, 7, 8)
CLUSTERING_FEATURES = [
    "total_sessions",
    "total_views",
    "unique_items_viewed",
    "total_add_to_cart",
    "avg_session_duration_sec",
    "avg_items_per_session",
    "cart_abandonment_rate",
    "recency_days",
]
