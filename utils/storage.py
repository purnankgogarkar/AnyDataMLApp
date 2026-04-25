"""SQLite storage operations."""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DB_PATH

logger = logging.getLogger(__name__)


def init_db():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Datasets table
    c.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            columns TEXT NOT NULL,
            preview TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # Runs (training runs) table
    c.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            target TEXT NOT NULL,
            features TEXT NOT NULL,
            problem_type TEXT NOT NULL,
            model_name TEXT NOT NULL,
            business_outcome TEXT,
            metrics TEXT NOT NULL,
            feature_importance TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )
    """)
    
    conn.commit()
    conn.close()


def save_dataset(dataset_id: str, filename: str, rows: int, cols: int,
                 columns: List[Dict[str, Any]], preview: List[Dict[str, Any]]) -> None:
    """Save dataset metadata to database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        INSERT OR REPLACE INTO datasets 
        (id, filename, rows, cols, columns, preview, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        dataset_id,
        filename,
        rows,
        cols,
        json.dumps(columns),
        json.dumps(preview),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve dataset metadata from database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT id, filename, rows, cols, columns, preview, created_at FROM datasets WHERE id = ?",
              (dataset_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "id": row[0],
        "filename": row[1],
        "rows": row[2],
        "cols": row[3],
        "columns": json.loads(row[4]),
        "preview": json.loads(row[5]),
        "created_at": row[6],
    }


def save_run(run_id: str, dataset_id: str, target: str, features: List[str],
             problem_type: str, model_name: str, business_outcome: str,
             metrics: Dict[str, float], feature_importance: List[Dict[str, Any]]) -> None:
    """Save training run metadata to database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        INSERT OR REPLACE INTO runs
        (id, dataset_id, target, features, problem_type, model_name, business_outcome, metrics, feature_importance, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        dataset_id,
        target,
        json.dumps(features),
        problem_type,
        model_name,
        business_outcome,
        json.dumps(metrics),
        json.dumps(feature_importance),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve training run metadata from database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        SELECT id, dataset_id, target, features, problem_type, model_name, 
               business_outcome, metrics, feature_importance, created_at 
        FROM runs WHERE id = ?
    """, (run_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "run_id": row[0],
        "dataset_id": row[1],
        "target": row[2],
        "features": json.loads(row[3]),
        "problem_type": row[4],
        "model_name": row[5],
        "business_outcome": row[6],
        "metrics": json.loads(row[7]),
        "feature_importance": json.loads(row[8]),
        "created_at": row[9],
    }


def list_runs() -> List[Dict[str, Any]]:
    """List all training runs."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        SELECT id, dataset_id, target, model_name, problem_type, metrics, created_at 
        FROM runs ORDER BY created_at DESC
    """)
    rows = c.fetchall()
    conn.close()
    
    return [
        {
            "run_id": row[0],
            "dataset_id": row[1],
            "target": row[2],
            "model_name": row[3],
            "problem_type": row[4],
            "metrics": json.loads(row[5]),
            "created_at": row[6],
        }
        for row in rows
    ]


def list_datasets() -> List[Dict[str, Any]]:
    """List all datasets."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT id, filename, rows, cols, created_at FROM datasets ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "filename": row[1],
            "rows": row[2],
            "cols": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]
