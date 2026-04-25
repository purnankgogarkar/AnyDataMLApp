"""Utilities package."""

from .config import (
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    TIMESERIES_MODELS,
    MISSING_STRATEGIES,
    AI_PROVIDERS,
)
from .storage import (
    init_db,
    list_datasets,
    list_runs,
    save_dataset,
    get_dataset,
    save_run,
    get_run,
)
from .data_handler import (
    load_dataset_file,
    process_and_save_dataset,
    load_dataset,
    suggest_problem_type,
    classify_column,
    compute_column_stats,
    get_numeric_columns,
    get_categorical_columns,
)
from .analysis import analyze_dataset, get_data_quality_report
from .ml_pipeline import train_model, TrainConfig, load_model, get_predictions_path, get_model_path
from .ai_handler import (
    detect_ai_provider,
    validate_api_key,
    get_model_explanation,
    get_feature_engineering_suggestions,
    get_smart_model_selection,
    get_business_recommendations,
    get_data_quality_insights,
    get_hyperparameter_suggestions,
)

__all__ = [
    # Config
    "REGRESSION_MODELS",
    "CLASSIFICATION_MODELS",
    "TIMESERIES_MODELS",
    "MISSING_STRATEGIES",
    "AI_PROVIDERS",
    # Storage
    "init_db",
    "list_datasets",
    "list_runs",
    "save_dataset",
    "get_dataset",
    "save_run",
    "get_run",
    # Data Handler
    "load_dataset_file",
    "process_and_save_dataset",
    "load_dataset",
    "suggest_problem_type",
    "classify_column",
    "compute_column_stats",
    "get_numeric_columns",
    "get_categorical_columns",
    # Analysis
    "analyze_dataset",
    "get_data_quality_report",
    # ML Pipeline
    "train_model",
    "TrainConfig",
    "load_model",
    "get_predictions_path",
    "get_model_path",
    # AI Handler
    "detect_ai_provider",
    "validate_api_key",
    "get_model_explanation",
    "get_feature_engineering_suggestions",
    "get_smart_model_selection",
    "get_business_recommendations",
    "get_data_quality_insights",
    "get_hyperparameter_suggestions",
]

