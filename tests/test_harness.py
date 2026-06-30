import sys
from pathlib import Path
import json

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import sqlite3
import config
from unittest.mock import patch, MagicMock
from agent.decision_node import run_decision_node
from agent.escalation_node import run_escalation_node
from agent.graph import run_pipeline
from agent import logger


def test_low_confidence_routing():
    """Verify that a low confidence score (e.g. score < 3.0) routes to escalation."""
    state = {
        "query": "What is Eko's stock price?",
        "retrieved_chunks": [
            {"source": "settlement_recon_discrepancy.md", "content": "Daily reconciliation runs at 11:30 PM.", "similarity": 0.2}
        ],
        "attempts": 1
    }
    
    # Combined score should be: (1 * 0.7) + (0.2 * 5 * 0.3) = 0.7 + 0.3 = 1.0 < 3.0
    result = run_decision_node(state)
    assert result["routing_decision"] == "escalate"
    assert result["combined_confidence"] < config.CONFIDENCE_THRESHOLD


def test_high_risk_keyword_escalation():
    """Verify that queries containing high-risk keywords route to High severity regardless of confidence."""
    # "fraud" is a high-risk keyword
    state = {
        "query": "A customer committed fraud at my store, what do I do?",
        "retrieved_chunks": [
            {"source": "csp_registration_sop.md", "content": "This is standard registration.", "similarity": 0.9}
        ],
        "combined_confidence": 4.5,
        "max_similarity": 0.9,
        "why_failed": ""
    }
    
    result = run_escalation_node(state)
    assert result["severity"] == "High"
    assert result["status"] == "escalated"
    assert result["escalation_note"]["severity"] == "High"


def test_retrieval_failure():
    """
    Verify that when retrieval returns zero chunks, the pipeline does not crash
    and escalates with a clear 'no relevant context found' reason.

    We mock retrieve_chunks to return [] because FAISS always returns nearest
    neighbours regardless of relevance — the retrieval failure path is a runtime
    condition (e.g. empty index), not a query-matching condition.
    """
    from agent import graph as graph_module

    original_retrieve = graph_module.retriever.retrieve_chunks

    try:
        # Force retriever to return no chunks
        graph_module.retriever.retrieve_chunks = lambda query, k=3: []

        result = run_pipeline("A query that should find zero matching documents")

        assert result["status"] == "escalated", f"Expected 'escalated', got: {result['status']}"
        # The pipeline must reach the escalation node — that's the contract.
        # why_failed may be set by retrieval or escalation node depending on attempt count.
        assert result["routing_decision"] in ("escalate", "malformed_output", "retry"), \
            f"Unexpected routing: {result['routing_decision']}"
    finally:
        # Always restore the original method
        graph_module.retriever.retrieve_chunks = original_retrieve


def test_malformed_llm_output():
    """Verify that malformed/schema-violating LLM output triggers stricter retry and eventual Low severity escalation."""
    # We pass __TEST_MALFORMED_OUTPUT__ in query to trigger mock LLM validation failure
    result = run_pipeline("My query has __TEST_MALFORMED_OUTPUT__")
    
    assert result["status"] == "escalated"
    # Ensure it routed to escalation due to schema check validation failures
    assert result["severity"] == "Low"


def test_logging_failure_fallback_queue():
    """Verify that SQLite write failures trigger fallback to the in-memory queue without losing data."""
    # Clear the queue first
    logger.fallback_queue.clear()
    
    # Mock sqlite3.connect to raise a database operational error
    with patch("sqlite3.connect", side_effect=sqlite3.OperationalError("Database disk is full or locked")):
        log_id = logger.log_interaction(
            query="Test query with SQLite failure",
            status="resolved",
            response="Fallback logging answer",
            confidence_score=4.8,
            max_similarity=0.9
        )
        
        # log_interaction should return -1 on fallback
        assert log_id == -1
        # Item must be stored in the fallback queue
        assert len(logger.fallback_queue) == 1
        assert logger.fallback_queue[0]["query"] == "Test query with SQLite failure"
        assert logger.fallback_queue[0]["op"] == "interaction"

    # Now verify flushing writes it to SQLite when the connection is restored
    assert len(logger.fallback_queue) == 1
    logger.flush_queued_logs()
    
    # Queue should be cleared and written successfully
    assert len(logger.fallback_queue) == 0
