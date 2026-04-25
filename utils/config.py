"""Configuration and constants for Streamlit app."""
from pathlib import Path
import os

# ========== Directory Configuration ==========
APP_DIR = Path(__file__).parent.parent
STORAGE_DIR = APP_DIR / "storage"
DB_PATH = STORAGE_DIR / "app.db"
DATASETS_DIR = STORAGE_DIR / "datasets"
MODELS_DIR = STORAGE_DIR / "models"
PREDICTIONS_DIR = STORAGE_DIR / "predictions"

# Ensure directories exist
for d in [STORAGE_DIR, DATASETS_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========== Model Configuration ==========
# Available models for each problem type
REGRESSION_MODELS = ["auto", "linear_regression", "random_forest", "xgboost"]
CLASSIFICATION_MODELS = ["auto", "logistic_regression", "random_forest", "xgboost"]
TIMESERIES_MODELS = ["random_forest"]

# ========== Feature Engineering Configuration ==========
# Strategies for handling missing values
MISSING_STRATEGIES = ["drop", "mean", "median", "mode"]

# ========== AI Provider Configuration ==========
# Supported AI providers with their API key prefixes and available models
AI_PROVIDERS = {
    "openai": {
        "prefix": "sk-",
        "models": ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        "default": "gpt-4o-mini",
    },
    "groq": {
        "prefix": "gsk_",
        "models": ["mixtral-8x7b-32768", "llama2-70b", "gemma-7b"],
        "default": "mixtral-8x7b-32768",
    },
    "anthropic": {
        "prefix": "sk-ant-",
        "models": ["claude-opus", "claude-sonnet", "claude-haiku"],
        "default": "claude-haiku",
    },
}

# ========== Training & Validation Thresholds ==========
# Minimum samples required for model training (prevents overfitting on tiny datasets)
MIN_ROWS_FOR_TRAINING = 10

# Threshold for flagging high-cardinality categorical features
# Features with more unique values than this will be flagged for review
HIGH_CARDINALITY_THRESHOLD = 50

# Threshold for flagging columns with excessive missing values (in percent)
# Columns with more than this % missing will be flagged
HIGH_NULL_PCT_THRESHOLD = 30

# ========== DateTime Feature Detection ==========
# Minimum samples to check for datetime columns
MIN_SAMPLES_FOR_DATETIME = 3

# Threshold for determining if a column is datetime (0.0-1.0)
# If this fraction of samples parse as dates, treat as datetime
DATETIME_MATCH_THRESHOLD = 0.7

# ========== Train-Test Split ==========
# Default test set size (fraction of data reserved for validation)
DEFAULT_TEST_SIZE = 0.2

# ========== Random State ==========
# Fixed seed for reproducibility across runs
RANDOM_STATE = 42

