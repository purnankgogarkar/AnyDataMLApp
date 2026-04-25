"""ML pipeline for model training and evaluation."""
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .config import MODELS_DIR, PREDICTIONS_DIR, MIN_ROWS_FOR_TRAINING, DEFAULT_TEST_SIZE, RANDOM_STATE
from .data_handler import classify_column
from .storage import save_run

logger = logging.getLogger(__name__)


class TrainConfig:
    """Training configuration."""
    def __init__(
        self,
        target: str,
        features: List[str],
        problem_type: str,
        model_name: str = "auto",
        business_outcome: str = "NA",
        missing_strategy: str = "mean",
        missing_strategies_per_column: dict = None,
        encode_categorical: bool = True,
        scale_features: bool = False,
        extract_datetime: bool = True,
        test_size: float = 0.2,
    ):
        self.target = target
        self.features = features
        self.problem_type = problem_type
        self.model_name = model_name
        self.business_outcome = business_outcome
        self.missing_strategy = missing_strategy
        self.missing_strategies_per_column = missing_strategies_per_column or {}
        self.encode_categorical = encode_categorical
        self.scale_features = scale_features
        self.extract_datetime = extract_datetime
        self.test_size = test_size


def prepare_data(df: pd.DataFrame, config: TrainConfig) -> Tuple:
    """Prepare data for training."""
    data = df[config.features + [config.target]].copy()
    
    # Extract datetime features
    if config.extract_datetime:
        for col in list(data.columns):
            if col == config.target:
                continue
            if classify_column(data[col]) == "datetime":
                parsed = pd.to_datetime(data[col], errors="coerce")
                data[f"{col}_year"] = parsed.dt.year
                data[f"{col}_month"] = parsed.dt.month
                data[f"{col}_day"] = parsed.dt.day
                data = data.drop(columns=[col])
    
    # Drop target NaN rows
    data = data.dropna(subset=[config.target])
    
    # Missing value handling
    feat_cols = [c for c in data.columns if c != config.target]
    if config.missing_strategy == "drop":
        data = data.dropna()
    else:
        num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(data[c])]
        cat_cols = [c for c in feat_cols if c not in num_cols]
        
        # Apply per-column strategies if available
        for col in num_cols:
            if col in config.missing_strategies_per_column:
                strategy = config.missing_strategies_per_column[col]
                if strategy == "drop":
                    data = data.dropna(subset=[col])
                else:
                    strat = "mean" if strategy == "mean" else ("median" if strategy == "median" else "most_frequent")
                    data[[col]] = SimpleImputer(strategy=strat).fit_transform(data[[col]])
        
        for col in cat_cols:
            if col in config.missing_strategies_per_column:
                strategy = config.missing_strategies_per_column[col]
                if strategy == "drop":
                    data = data.dropna(subset=[col])
                else:
                    data[[col]] = SimpleImputer(strategy="most_frequent").fit_transform(data[[col]])
        
        # Apply default fallback strategy to remaining columns
        remaining_num = [c for c in num_cols if c not in config.missing_strategies_per_column and data[c].isnull().any()]
        remaining_cat = [c for c in cat_cols if c not in config.missing_strategies_per_column and data[c].isnull().any()]
        
        if remaining_num:
            strat = "mean" if config.missing_strategy == "mean" else ("median" if config.missing_strategy == "median" else "most_frequent")
            data[remaining_num] = SimpleImputer(strategy=strat).fit_transform(data[remaining_num])
        
        if remaining_cat:
            data[remaining_cat] = SimpleImputer(strategy="most_frequent").fit_transform(data[remaining_cat])
    
    y = data[config.target]
    X = data.drop(columns=[config.target])
    
    # Classification target encoding if object
    target_classes = None
    if config.problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        y, uniques = pd.factorize(y)
        target_classes = list(uniques)
    
    # Encode categorical
    if config.encode_categorical:
        X = pd.get_dummies(X, drop_first=False, dummy_na=False)
    else:
        X = X.select_dtypes(include=[np.number])
    
    # Scale features
    scaler = None
    if config.scale_features and len(X.columns) > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X, y, target_classes, scaler


def pick_model(problem_type: str, model_name: str) -> Tuple[Any, str]:
    """Select model based on problem type and name."""
    name = model_name.lower()
    
    if problem_type == "regression":
        if name in ("auto", "random_forest"):
            return RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE), "random_forest"
        if name == "linear_regression":
            return LinearRegression(), "linear_regression"
        if name == "xgboost":
            return XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbosity=0), "xgboost"
    
    elif problem_type == "classification":
        if name in ("auto", "random_forest"):
            return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), "random_forest"
        if name == "logistic_regression":
            return LogisticRegression(max_iter=1000), "logistic_regression"
        if name == "xgboost":
            return XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, verbosity=0, 
                               use_label_encoder=False, eval_metric="logloss"), "xgboost"
    
    elif problem_type == "time_series":
        return RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE), "random_forest_ts"
    
    raise ValueError(f"Unsupported model/problem combo: {model_name}/{problem_type}")


def train_model(df: pd.DataFrame, config: TrainConfig, dataset_id: str) -> Dict[str, Any]:
    """Train model and return results."""
    # Validate inputs
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Invalid DataFrame provided")
    
    if config.target not in df.columns:
        raise ValueError(f"Target column '{config.target}' not found in DataFrame")
    
    missing_features = [f for f in config.features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    if not config.features:
        raise ValueError("At least one feature must be selected")
    
    if config.problem_type not in ["classification", "regression", "time_series"]:
        raise ValueError(f"Invalid problem type: {config.problem_type}")
    
    try:
        # Prepare data
        logger.info(f"Preparing data with {len(df)} rows, {len(config.features)} features")
        X, y, target_classes, scaler = prepare_data(df, config)
        
        if len(X) < MIN_ROWS_FOR_TRAINING:
            raise ValueError(
                f"Insufficient data after preprocessing: {len(X)} rows. "
                f"Minimum required: {MIN_ROWS_FOR_TRAINING}"
            )
        
        logger.info(f"Data prepared: {X.shape[0]} rows, {X.shape[1]} features")
        
        # Pick model
        model, resolved_name = pick_model(config.problem_type, config.model_name)
        logger.info(f"Selected model: {resolved_name}")
        
        # Train/test split
        stratify = y if (config.problem_type == "classification" and len(np.unique(y)) > 1) else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_size, random_state=RANDOM_STATE, stratify=stratify
            )
        except Exception as e:
            logger.warning(f"Stratified split failed: {e}. Using regular split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_size, random_state=RANDOM_STATE
            )
        
        # Train
        logger.info(f"Training {resolved_name} model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Compute metrics
        metrics: Dict[str, float] = {}
        if config.problem_type == "regression" or config.problem_type == "time_series":
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))
            metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
            logger.info(f"Regression metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        else:
            acc = float(accuracy_score(y_test, y_pred))
            avg = "binary" if len(np.unique(y)) == 2 else "weighted"
            prec = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
            rec = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
            metrics = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
            logger.info(f"Classification metrics - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        # Feature importance
        importance: List[Dict[str, Any]] = []
        if hasattr(model, "feature_importances_"):
            for name, val in zip(X.columns, model.feature_importances_):
                importance.append({"feature": name, "importance": float(round(val, 6))})
        elif hasattr(model, "coef_"):
            coef = np.array(model.coef_).ravel()
            if len(coef) == len(X.columns):
                for name, val in zip(X.columns, coef):
                    importance.append({"feature": name, "importance": float(round(abs(val), 6))})
        importance.sort(key=lambda x: x["importance"], reverse=True)
        
        # Generate run ID
        run_id = str(uuid.uuid4())
        
        # Save model
        logger.info(f"Saving model {run_id[:8]} to storage...")
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_columns": list(X.columns),
                "target": config.target,
                "target_classes": target_classes,
                "problem_type": config.problem_type,
            },
            MODELS_DIR / f"{run_id}.joblib",
        )
        
        # Save predictions
        preds_df = pd.DataFrame({
            "actual": y_test.values if hasattr(y_test, "values") else y_test,
            "predicted": y_pred
        })
        if target_classes is not None:
            preds_df["actual_label"] = preds_df["actual"].apply(
                lambda i: target_classes[int(i)] if 0 <= int(i) < len(target_classes) else i
            )
            preds_df["predicted_label"] = preds_df["predicted"].apply(
                lambda i: target_classes[int(i)] if 0 <= int(i) < len(target_classes) else i
            )
        pred_path = PREDICTIONS_DIR / f"{run_id}.csv"
        preds_df.to_csv(pred_path, index=False)
        logger.info(f"Predictions saved to {pred_path}")
        
        # Save to database
        save_run(
            run_id,
            dataset_id,
            config.target,
            config.features,
            config.problem_type,
            resolved_name,
            config.business_outcome,
            metrics,
            importance[:20]
        )
        logger.info(f"Run {run_id[:8]} saved to database")
        
        return {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "target": config.target,
            "features": config.features,
            "problem_type": config.problem_type,
            "model_name": resolved_name,
            "business_outcome": config.business_outcome,
            "metrics": metrics,
            "feature_importance": importance[:20],
        }
    
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        raise
    
    # Pick model
    model, resolved_name = pick_model(config.problem_type, config.model_name)
    
    # Train/test split
    stratify = y if (config.problem_type == "classification" and len(np.unique(y)) > 1) else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=42, stratify=stratify
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=42
        )
    
    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute metrics
    metrics: Dict[str, float] = {}
    if config.problem_type == "regression" or config.problem_type == "time_series":
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
    else:
        acc = float(accuracy_score(y_test, y_pred))
        avg = "binary" if len(np.unique(y)) == 2 else "weighted"
        prec = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
        rec = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
    
    # Feature importance
    importance: List[Dict[str, Any]] = []
    if hasattr(model, "feature_importances_"):
        for name, val in zip(X.columns, model.feature_importances_):
            importance.append({"feature": name, "importance": float(round(val, 6))})
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_).ravel()
        if len(coef) == len(X.columns):
            for name, val in zip(X.columns, coef):
                importance.append({"feature": name, "importance": float(round(abs(val), 6))})
    importance.sort(key=lambda x: x["importance"], reverse=True)
    
    # Generate run ID
    run_id = str(uuid.uuid4())
    
    # Save model
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": list(X.columns),
            "target": config.target,
            "target_classes": target_classes,
            "problem_type": config.problem_type,
        },
        MODELS_DIR / f"{run_id}.joblib",
    )
    
    # Save predictions
    preds_df = pd.DataFrame({
        "actual": y_test.values if hasattr(y_test, "values") else y_test,
        "predicted": y_pred
    })
    if target_classes is not None:
        preds_df["actual_label"] = preds_df["actual"].apply(
            lambda i: target_classes[int(i)] if 0 <= int(i) < len(target_classes) else i
        )
        preds_df["predicted_label"] = preds_df["predicted"].apply(
            lambda i: target_classes[int(i)] if 0 <= int(i) < len(target_classes) else i
        )
    pred_path = PREDICTIONS_DIR / f"{run_id}.csv"
    preds_df.to_csv(pred_path, index=False)
    
    # Save to database
    save_run(
        run_id,
        dataset_id,
        config.target,
        config.features,
        config.problem_type,
        resolved_name,
        config.business_outcome,
        metrics,
        importance[:20]
    )
    
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "target": config.target,
        "features": config.features,
        "problem_type": config.problem_type,
        "model_name": resolved_name,
        "business_outcome": config.business_outcome,
        "metrics": metrics,
        "feature_importance": importance[:20],
    }


def load_model(run_id: str) -> Dict[str, Any]:
    """Load saved model."""
    path = MODELS_DIR / f"{run_id}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model {run_id} not found")
    return joblib.load(path)


def get_predictions_path(run_id: str) -> str:
    """Get path to predictions CSV."""
    path = PREDICTIONS_DIR / f"{run_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Predictions for {run_id} not found")
    return str(path)


def get_model_path(run_id: str) -> str:
    """Get path to model file."""
    path = MODELS_DIR / f"{run_id}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model {run_id} not found")
    return str(path)
