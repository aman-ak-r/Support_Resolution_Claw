import sqlite3
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import config
from agent.schemas import LoggingInput

logger = logging.getLogger(__name__)

# Global in-memory fallback queue for failed database logs
fallback_queue: List[Dict[str, Any]] = []


def get_db_connection():
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    Path(config.DB_PATH).parent.mkdir(exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Interactions table
    cursor.execute("""
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
    
    # Tickets lifecycle table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            severity TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('open', 'assigned', 'resolved', 'closed')),
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            escalation_note TEXT NOT NULL,
            state_history TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()


def flush_queued_logs() -> None:
    """Attempts to write any queued interactions/tickets back to SQLite."""
    global fallback_queue
    if not fallback_queue:
        return

    logger.info(f"Attempting to flush {len(fallback_queue)} logs from in-memory queue...")
    remaining_queue = []
    
    # Try connecting to check if SQLite is writable
    try:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()
        
        for item in fallback_queue:
            try:
                op_type = item.get("op")
                if op_type == "interaction":
                    cursor.execute(
                        """
                        INSERT INTO interactions (query, status, response, confidence_score, max_similarity, severity, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item["query"],
                            item["status"],
                            item["response"],
                            item["confidence_score"],
                            item["max_similarity"],
                            item["severity"],
                            item["timestamp"]
                        )
                    )
                elif op_type == "create_ticket":
                    cursor.execute(
                        """
                        INSERT INTO tickets (ticket_id, query, severity, state, created_at, updated_at, escalation_note, state_history)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item["ticket_id"],
                            item["query"],
                            item["severity"],
                            item["state"],
                            item["created_at"],
                            item["updated_at"],
                            item["escalation_note"],
                            item["state_history"]
                        )
                    )
                elif op_type == "update_ticket":
                    cursor.execute(
                        """
                        UPDATE tickets 
                        SET state = ?, updated_at = ?, state_history = ?
                        WHERE ticket_id = ?
                        """,
                        (
                            item["state"],
                            item["updated_at"],
                            item["state_history"],
                            item["ticket_id"]
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to process item during flush: {e}")
                remaining_queue.append(item)
                
        conn.commit()
        conn.close()
        fallback_queue = remaining_queue
        logger.info(f"Flush complete. {len(fallback_queue)} logs remaining in memory queue.")
    except Exception as e:
        logger.warning(f"Could not connect to database to flush queue: {e}")


def log_interaction(
    query: str,
    status: str,
    response: str,
    confidence_score: float,
    max_similarity: float,
    severity: str = None
) -> int:
    # Explicit Pydantic Schema Validation
    input_data = LoggingInput(
        query=query,
        status=status,
        response=response,
        confidence_score=confidence_score,
        max_similarity=max_similarity,
        severity=severity
    )
    
    # Try flushing queued items first
    flush_queued_logs()
    
    timestamp_str = datetime.now().isoformat()

    try:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO interactions (query, status, response, confidence_score, max_similarity, severity, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                input_data.query,
                input_data.status,
                input_data.response,
                input_data.confidence_score,
                input_data.max_similarity,
                input_data.severity,
                timestamp_str
            )
        )
        conn.commit()
        inserted_id = cursor.lastrowid
        conn.close()
        return inserted_id
    except Exception as e:
        logger.warning(f"SQLite log write failed: {e}. Falling back to in-memory queue.")
        # Queue the interaction details
        fallback_queue.append({
            "op": "interaction",
            "query": input_data.query,
            "status": input_data.status,
            "response": input_data.response,
            "confidence_score": input_data.confidence_score,
            "max_similarity": input_data.max_similarity,
            "severity": input_data.severity,
            "timestamp": timestamp_str
        })
        return -1


# --- Ticket State Machine Helper Functions ---

def create_ticket(query: str, severity: str, escalation_note: Dict[str, Any]) -> str:
    """Creates a new escalation ticket in the database with UUID and 'open' status."""
    ticket_id = str(uuid.uuid4())
    state = "open"
    timestamp_str = datetime.now().isoformat()
    
    history_entry = {
        "state": state,
        "timestamp": timestamp_str,
        "note": "Ticket generated by autonomous escalation node."
    }
    
    payload = {
        "ticket_id": ticket_id,
        "query": query,
        "severity": severity,
        "state": state,
        "created_at": timestamp_str,
        "updated_at": timestamp_str,
        "escalation_note": json.dumps(escalation_note),
        "state_history": json.dumps([history_entry])
    }

    flush_queued_logs()

    try:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO tickets (ticket_id, query, severity, state, created_at, updated_at, escalation_note, state_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["ticket_id"],
                payload["query"],
                payload["severity"],
                payload["state"],
                payload["created_at"],
                payload["updated_at"],
                payload["escalation_note"],
                payload["state_history"]
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to create SQLite ticket: {e}. Falling back to in-memory queue.")
        payload["op"] = "create_ticket"
        fallback_queue.append(payload)
        
    return ticket_id


def update_ticket_state(ticket_id: str, new_state: str, note: str = "") -> bool:
    """Transitions a ticket state and appends the action to the state history timeline."""
    if new_state not in ("open", "assigned", "resolved", "closed"):
        raise ValueError(f"Invalid state transition: {new_state}")

    flush_queued_logs()
    timestamp_str = datetime.now().isoformat()

    try:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Fetch existing history
        cursor.execute("SELECT state_history FROM tickets WHERE ticket_id = ?", (ticket_id,))
        row = cursor.fetchone()
        if not row:
            logger.error(f"Ticket {ticket_id} not found in database.")
            conn.close()
            return False
            
        history = json.loads(row["state_history"])
        history.append({
            "state": new_state,
            "timestamp": timestamp_str,
            "note": note or f"State transitioned to {new_state}."
        })
        
        cursor.execute(
            """
            UPDATE tickets 
            SET state = ?, updated_at = ?, state_history = ?
            WHERE ticket_id = ?
            """,
            (new_state, timestamp_str, json.dumps(history), ticket_id)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"Failed to update SQLite ticket state: {e}. Falling back to in-memory queue.")
        
        # We need to construct a fallback entry even if the db fetch fails
        fallback_queue.append({
            "op": "update_ticket",
            "ticket_id": ticket_id,
            "state": new_state,
            "updated_at": timestamp_str,
            "state_history": json.dumps([{
                "state": new_state,
                "timestamp": timestamp_str,
                "note": note or f"Fallback transition queue update to {new_state}."
            }])
        })
        return False


def assign_ticket(ticket_id: str, assignee: str = "Human Agent") -> bool:
    return update_ticket_state(ticket_id, "assigned", f"Ticket assigned to {assignee}.")


def resolve_ticket(ticket_id: str) -> bool:
    return update_ticket_state(ticket_id, "resolved", "Issue resolved by support agent.")


def close_ticket(ticket_id: str) -> bool:
    return update_ticket_state(ticket_id, "closed", "Ticket closed by support agent.")


def get_all_tickets() -> List[Dict[str, Any]]:
    flush_queued_logs()
    try:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tickets ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to fetch tickets: {e}")
        # Search the fallback queue for any in-flight create operations
        mock_tickets = []
        for item in fallback_queue:
            if item.get("op") == "create_ticket":
                mock_tickets.append({
                    "ticket_id": item["ticket_id"],
                    "query": item["query"],
                    "severity": item["severity"],
                    "state": item["state"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "escalation_note": item["escalation_note"],
                    "state_history": item["state_history"]
                })
        return mock_tickets


# --- Telemetry & Reporting ---

def get_all_logs() -> List[Dict[str, Any]]:
    flush_queued_logs()
    try:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM interactions ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to fetch interactions logs: {e}")
        return []


def export_summary_report() -> Dict[str, Any]:
    flush_queued_logs()
    try:
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
    except Exception as e:
        logger.error(f"Failed to export summary report: {e}")
        return {
            "total_queries": 0,
            "resolved_count": 0,
            "escalated_count": 0,
            "resolution_rate_percent": 0.0,
            "severity_breakdown": {"Low": 0, "Medium": 0, "High": 0},
            "average_confidence": 0.0,
            "average_similarity": 0.0
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
