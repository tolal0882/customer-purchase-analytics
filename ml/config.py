"""
Project configuration – ONE place to change settings.

Follows the teacher's ds_project template architecture (FastAPI + Streamlit + ml/)
for: "E-commerce Purchase Behavior Prediction and Customer Segmentation Using Machine Learning"
Dataset: Retailrocket E-commerce Dataset (data/raw/events.csv, category_tree.csv)
"""
from pathlib import Path

# =============================================================================
# 1. PROJECT PATHS (TEACHER TEMPLATE + EXTENDED ARTIFACTS)
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "events.csv"
CATEGORY_TREE_PATH = ROOT_DIR / "data" / "raw" / "category_tree.csv"
ITEM_PROPS_PART1_PATH = ROOT_DIR / "data" / "raw" / "item_properties_part1.csv"
ITEM_PROPS_PART2_PATH = ROOT_DIR / "data" / "raw" / "item_properties_part2.csv"

PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "clean.csv"
CUSTOMER_PROFILES_PATH = ROOT_DIR / "data" / "processed" / "customer_profiles.csv"
SESSION_PROFILES_PATH = ROOT_DIR / "data" / "processed" / "session_profiles.csv"
INSIGHTS_SUMMARY_PATH = ROOT_DIR / "data" / "processed" / "insights_summary.json"
PREDICTIONS_DB_PATH = ROOT_DIR / "data" / "predictions.db"

MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
EXPERIMENTS_PATH = ROOT_DIR / "models" / "experiments.json"
MODEL_CARD_PATH = ROOT_DIR / "models" / "model_card.md"
CLUSTERING_MODEL_PATH = ROOT_DIR / "models" / "clustering.joblib"
CLUSTERING_METRICS_PATH = ROOT_DIR / "models" / "clustering_metrics.json"
SESSION_CLUSTERING_METRICS_PATH = ROOT_DIR / "models" / "session_clustering_metrics.json"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# =============================================================================
# 2. SUPERVISED LEARNING PROBLEM DEFINITION (PURCHASE PREDICTION)
# =============================================================================
TASK_TYPE = "classification"

TARGET_COLUMN = "will_purchase"
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"

# Identifier / timestamp columns blocked from model inputs (leakage prevention)
DROP_COLUMNS = ["session_id", "visitorid", "session_start_ts"]

# =============================================================================
# 3. RETAILROCKET SESSION & SNAPSHOT ENGINEERING SETTINGS
# =============================================================================
VALID_EVENTS = ("view", "addtocart", "transaction")
SESSION_TIMEOUT_MINUTES = 30.0   # 30-min inactivity threshold (analytical assumption)
SNAPSHOT_EVENTS = 3              # Early-session cutoff: first 3 non-transaction events
MAX_MODELING_ROWS = 30_000       # Chronological stratified sample for interactive training
MAX_CUSTOMER_PROFILES = 25_000   # Engaged visitor profiles for K-Means customer clustering
MAX_SESSION_PROFILES = 25_000    # Multi-event session profiles for K-Means session clustering

# =============================================================================
# 4. TRAINING & CLUSTERING HYPERPARAMETERS
# =============================================================================
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42
PRIMARY_METRIC = "f1" if TASK_TYPE == "classification" else "neg_root_mean_squared_error"
TUNING_ITERATIONS = 10

DEFAULT_K_CLUSTERS = 4
K_RANGE = (2, 3, 4, 5, 6, 7, 8)

CUSTOMER_CLUSTERING_FEATURES = [
    "total_sessions",
    "total_views",
    "unique_items_viewed",
    "total_add_to_cart",
    "avg_session_duration_sec",
    "avg_items_per_session",
    "avg_cart_additions_per_session",
    "cart_abandonment_rate",
    "recency_days",
]

# Backward-compatible alias used by legacy callers/tests
CLUSTERING_FEATURES = CUSTOMER_CLUSTERING_FEATURES

SESSION_CLUSTERING_FEATURES = [
    "total_events",
    "total_views",
    "unique_items_viewed",
    "add_to_cart_count",
    "session_duration_sec",
    "avg_time_between_events_sec",
    "cart_interaction_ratio",
    "item_revisit_ratio",
]
