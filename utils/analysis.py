"""Data analysis utilities."""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .data_handler import classify_column
from .config import HIGH_NULL_PCT_THRESHOLD, HIGH_CARDINALITY_THRESHOLD

logger = logging.getLogger(__name__)


def analyze_dataset(df: pd.DataFrame, target: str, features: List[str]) -> Dict[str, Any]:
    """Analyze dataset for quality issues and feature correlations."""
    warnings: List[Dict[str, str]] = []
    correlations: List[Dict[str, Any]] = []
    
    # Check for missing values and high cardinality
    for f in features:
        if f not in df.columns:
            continue
        s = df[f]
        null_pct = float(s.isna().sum()) / max(len(df), 1) * 100.0
        
        if null_pct > HIGH_NULL_PCT_THRESHOLD:
            warnings.append({
                "feature": f,
                "type": "missing",
                "message": f"{f} has {null_pct:.1f}% missing values"
            })
        
        if classify_column(s) == "categorical" and s.nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD:
            warnings.append({
                "feature": f,
                "type": "high_cardinality",
                "message": f"{f} has {s.nunique()} unique values"
            })
    
    # Compute correlations for numeric target
    if target and target in df.columns and classify_column(df[target]) == "numeric":
        for f in features:
            if f not in df.columns or f == target:
                continue
            if classify_column(df[f]) == "numeric":
                try:
                    corr = float(df[[f, target]].dropna().corr().iloc[0, 1])
                    if not np.isnan(corr):
                        correlations.append({"feature": f, "correlation": round(corr, 4)})
                except Exception:
                    pass
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    return {
        "warnings": warnings,
        "correlations": correlations[:15],
    }


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data quality report."""
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "missing_data": {},
        "duplicates": {
            "total": int(df.duplicated().sum()),
            "percentage": round(df.duplicated().sum() / len(df) * 100, 2)
        },
        "column_issues": [],
    }
    
    for col in df.columns:
        s = df[col]
        null_pct = s.isna().sum() / len(df) * 100
        report["missing_data"][col] = {
            "count": int(s.isna().sum()),
            "percentage": round(null_pct, 2)
        }
        
        ctype = classify_column(s)
        
        # Flag issues
        issues = []
        if null_pct > 30:
            issues.append(f"High missing rate ({null_pct:.1f}%)")
        
        if ctype == "numeric":
            if s.min() < 0 and s.max() > 1000:
                issues.append("Wide numeric range (potential outliers)")
        elif ctype == "categorical":
            if s.nunique() > 50:
                issues.append(f"High cardinality ({s.nunique()} unique values)")
        
        if issues:
            report["column_issues"].append({
                "column": col,
                "type": ctype,
                "issues": issues
            })
    
    return report


def detect_outliers(series: pd.Series, method: str = "iqr") -> np.ndarray:
    """Detect outliers using IQR or Z-score method."""
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    else:  # z-score
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > 3


def get_class_distribution(series: pd.Series) -> Dict[str, int]:
    """Get class distribution for classification target."""
    return series.value_counts().to_dict()
