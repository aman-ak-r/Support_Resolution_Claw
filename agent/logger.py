import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import config

logger = logging.getLogger(__name__)


def get_db_connection():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    Path(config.DB_PATH).parent.mkdir(exist_ok=True)
    conn = get_db_connection()
    conn.cursor().execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('resolved', 'escalated')),
            response TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            max_similarity REAL NOT NULL,
            severity TEXT CHECK(severity IN ('Low', 'Medium', 'High', NULL)),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_interaction(
    query: str,
    status: str,
    response: str,
    confidence_score: float,
    max_similarity: float,
    severity: str = None
) -> int:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO interactions (query, status, response, confidence_score, max_similarity, severity, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (query, status, response, confidence_score, max_similarity, severity, datetime.now().isoformat())
    )
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id


def get_all_logs() -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interactions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def export_summary_report() -> Dict[str, Any]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM interactions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM interactions WHERE status = 'resolved'")
    resolved = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM interactions WHERE status = 'escalated'")
    escalated = cursor.fetchone()[0]

    cursor.execute("SELECT severity, COUNT(*) FROM interactions WHERE status = 'escalated' GROUP BY severity")
    severity_breakdown = {"Low": 0, "Medium": 0, "High": 0}
    for row in cursor.fetchall():
        if row[0] in severity_breakdown:
            severity_breakdown[row[0]] = row[1]

    cursor.execute("SELECT AVG(confidence_score) FROM interactions")
    avg_confidence = cursor.fetchone()[0] or 0.0

    cursor.execute("SELECT AVG(max_similarity) FROM interactions")
    avg_similarity = cursor.fetchone()[0] or 0.0

    conn.close()

    return {
        "total_queries": total,
        "resolved_count": resolved,
        "escalated_count": escalated,
        "resolution_rate_percent": round((resolved / total * 100.0) if total > 0 else 0.0, 2),
        "severity_breakdown": severity_breakdown,
        "average_confidence": round(avg_confidence, 2),
        "average_similarity": round(avg_similarity, 2)
    }


def write_summary_report_to_file(file_path: str = None) -> str:
    if file_path is None:
        file_path = str(config.BASE_DIR / "logs" / "summary_report.json")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_summary_report(), f, indent=4)
        return file_path
    except Exception as e:
        logger.error(f"Failed to write summary report: {e}")
        return ""
