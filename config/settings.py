"""Central configuration file for the entire application.

Design rule: Keep only constants and pure path-lookup functions here (no mutable
global state), so every module shares the same single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# =============================================================================
# 1. PROJECT DIRECTORY PATHS
#    - Resolves all data, model, and experiment folders relative to project root.
#    - Checks data/raw/ first and falls back to data/row/ if needed.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR_CANDIDATES = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data" / "row"]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HISTORY_DIR = PROJECT_ROOT / "data" / "history"
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFICATION_MODEL_DIR = MODELS_DIR / "classification"
CUSTOMER_CLUSTER_DIR = MODELS_DIR / "customer_clustering"
SESSION_CLUSTER_DIR = MODELS_DIR / "session_clustering"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
ASSETS_DIR = PROJECT_ROOT / "assets"
PREDICTION_HISTORY_FILE = HISTORY_DIR / "prediction_history.csv"

# =============================================================================
# 2. DATASET SCHEMA & VALIDATION CONSTANTS
#    - Only events.csv is required; category_tree & item_properties are optional.
#    - Timestamps are validated against plausible epoch-ms bounds (2000–2100).
# =============================================================================
DATASET_FILES = {
    "events": "events.csv",
    "category_tree": "category_tree.csv",
    "item_properties_part1": "item_properties_part1.csv",
    "item_properties_part2": "item_properties_part2.csv",
}
REQUIRED_FILES = ["events"]

EVENTS_COLUMNS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
VALID_EVENTS = ["view", "addtocart", "transaction"]
TS_MIN_MS = 946_684_800_000      # 2000-01-01 00:00:00 UTC in milliseconds
TS_MAX_MS = 4_102_444_800_000    # 2100-01-01 00:00:00 UTC in milliseconds

# =============================================================================
# 3. ANALYTICAL & MACHINE LEARNING DEFAULTS
#    - 30-minute session timeout is an analytical assumption (not in raw data).
#    - 3-event snapshot prevents future-data leakage in purchase prediction.
#    - Sample caps keep Silhouette and 2-D PCA fast on a 16 GB laptop.
# =============================================================================
DEFAULT_SESSION_TIMEOUT_MIN = 30.0
DEFAULT_SNAPSHOT_EVENTS = 3
DEFAULT_SNAPSHOT_CUTOFF_MIN = 5.0
DEFAULT_RANDOM_SEED = 42
DEFAULT_TEST_RATIO = 0.2
DEFAULT_REMOVE_DUPLICATES = True
DEFAULT_MIN_CUSTOMER_SESSIONS = 2
DEFAULT_MIN_CUSTOMER_EVENTS = 3
SILHOUETTE_SAMPLE_SIZE = 20_000
PLOT_SAMPLE_SIZE = 20_000

# =============================================================================
# 4. DESIGN SYSTEM & NAVIGATION REGISTRY
#    - Forest Green palette used across CSS, Plotly charts, and React views.
# =============================================================================
COLORS = {
    "forest": "#153E2D",
    "primary": "#2F855A",
    "accent": "#A7C957",
    "background": "#F3F7F3",
    "card": "#FFFFFF",
    "text": "#263238",
    "muted": "#6B7C74",
    "border": "#DDE7DF",
    "danger": "#C0392B",
    "warning": "#D69E2E",
}
CHART_COLORS = ["#2F855A", "#A7C957", "#153E2D", "#5BA67C", "#D69E2E", "#4A7C8C", "#8E6C8A", "#C0785A"]

PAGES = [
    ("dashboard", "Dashboard", "📊"),
    ("dataset_explorer", "Dataset Explorer", "🗂️"),
    ("preprocessing", "Data Preprocessing", "🧹"),
    ("eda", "Exploratory Analysis", "🔎"),
    ("customer_segmentation", "Segmentation", "👥"),
    ("model_training", "Model Training", "🧠"),
    ("model_evaluation", "Model Evaluation", "✅"),
    ("purchase_prediction", "Purchase Prediction", "🛒"),
    ("prediction_history", "Prediction History", "🕘"),
    ("about", "About Project", "ℹ️"),
]


# =============================================================================
# 5. DIRECTORY HELPERS
# =============================================================================
def resolve_raw_dir() -> Path:
    """Return the first candidate directory that contains events.csv, else data/raw."""
    for d in RAW_DIR_CANDIDATES:
        if (d / DATASET_FILES["events"]).exists():
            return d
    return RAW_DIR_CANDIDATES[0]


def ensure_dirs() -> None:
    """Create all local output directories if they do not exist yet."""
    for d in (
        PROCESSED_DIR,
        HISTORY_DIR,
        CLASSIFICATION_MODEL_DIR,
        CUSTOMER_CLUSTER_DIR,
        SESSION_CLUSTER_DIR,
        EXPERIMENTS_DIR,
        ASSETS_DIR,
        RAW_DIR_CANDIDATES[0],
    ):
        d.mkdir(parents=True, exist_ok=True)
