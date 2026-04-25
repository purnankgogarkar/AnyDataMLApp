"""Data handling utilities for loading, parsing, and processing datasets."""
import io
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DATASETS_DIR, MIN_SAMPLES_FOR_DATETIME, DATETIME_MATCH_THRESHOLD
from .storage import save_dataset, get_dataset

logger = logging.getLogger(__name__)


def classify_column(series: pd.Series) -> str:
    """Classify a pandas series as numeric / categorical / datetime."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # Try to parse as datetime
    if series.dtype == object:
        sample = series.dropna().astype(str).head(20)
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() >= max(MIN_SAMPLES_FOR_DATETIME, int(len(sample) * DATETIME_MATCH_THRESHOLD)):
                return "datetime"
    return "categorical"


def compute_column_stats(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute statistics for each column in dataframe."""
    stats = []
    total = len(df)
    for col in df.columns:
        s = df[col]
        ctype = classify_column(s)
        null_pct = float(s.isna().sum()) / max(total, 1) * 100.0
        unique = int(s.nunique(dropna=True))
        entry: Dict[str, Any] = {
            "name": col,
            "dtype": ctype,
            "null_pct": round(null_pct, 2),
            "unique": unique,
        }
        if ctype == "numeric":
            try:
                entry["mean"] = float(round(s.mean(), 4)) if s.notna().any() else None
                entry["min"] = float(round(s.min(), 4)) if s.notna().any() else None
                entry["max"] = float(round(s.max(), 4)) if s.notna().any() else None
            except Exception:
                pass
        stats.append(entry)
    return stats


def df_to_records(df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
    """Convert dataframe to JSON-safe records."""
    sub = df.head(limit).copy()
    for c in sub.columns:
        if pd.api.types.is_datetime64_any_dtype(sub[c]):
            sub[c] = sub[c].astype(str)
    sub = sub.replace({np.nan: None})
    return json.loads(sub.to_json(orient="records", date_format="iso"))


def load_dataset_file(file_content: bytes, filename: str) -> Tuple[pd.DataFrame, str]:
    """Load dataset from uploaded file (CSV or Excel)."""
    ext = filename.lower().rsplit(".", 1)[-1]
    
    if ext not in {"csv", "xlsx", "xls"}:
        raise ValueError("Only CSV or Excel files are supported.")
    
    try:
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(file_content))
        else:
            df = pd.read_excel(io.BytesIO(file_content))
    except Exception as exc:
        raise ValueError(f"Unable to parse file: {exc}")
    
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    
    return df, ext


def process_and_save_dataset(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Process uploaded file and save to storage."""
    try:
        logger.info(f"Processing dataset file: {filename}")
        df, ext = load_dataset_file(file_content, filename)
        logger.info(f"File loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
        dataset_id = str(uuid.uuid4())
        
        # Save dataframe as pickle
        df_path = DATASETS_DIR / f"{dataset_id}.pkl"
        df.to_pickle(df_path)
        logger.info(f"Dataset saved to {df_path}")
        
        # Create metadata
        meta = {
            "id": dataset_id,
            "filename": filename,
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "columns": compute_column_stats(df),
            "preview": df_to_records(df, 10),
        }
        
        # Save to database
        save_dataset(
            dataset_id,
            filename,
            meta["rows"],
            meta["cols"],
            meta["columns"],
            meta["preview"]
        )
        logger.info(f"Dataset metadata saved for {dataset_id[:8]}")
        
        return meta
    
    except Exception as e:
        logger.error(f"Failed to process dataset {filename}: {str(e)}", exc_info=True)
        raise


def load_dataset(dataset_id: str) -> pd.DataFrame:
    """Load dataset dataframe from storage."""
    path = DATASETS_DIR / f"{dataset_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset {dataset_id} not found")
    return pd.read_pickle(path)


def suggest_problem_type(df: pd.DataFrame, target: str) -> str:
    """Suggest problem type based on target column."""
    s = df[target]
    ctype = classify_column(s)
    
    if ctype == "datetime":
        return "time_series"
    if ctype == "numeric":
        # Low unique numeric → classification
        if s.nunique(dropna=True) <= 10 and s.dropna().apply(lambda v: float(v).is_integer()).all():
            return "classification"
        return "regression"
    return "classification"


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns."""
    return [c for c in df.columns if classify_column(df[c]) == "numeric"]


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of categorical columns."""
    return [c for c in df.columns if classify_column(df[c]) == "categorical"]
