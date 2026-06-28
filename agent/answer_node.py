import logging
from typing import Dict, Any
from agent.logger import log_interaction

logger = logging.getLogger(__name__)


def run_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    draft_answer = state.get("draft_answer", "No answer was generated.")
    combined_confidence = state.get("combined_confidence", 0.0)
    max_similarity = state.get("max_similarity", 0.0)

    try:
        log_interaction(
            query=query,
            status="resolved",
            response=draft_answer,
            confidence_score=combined_confidence,
            max_similarity=max_similarity,
            severity=None
        )
    except Exception as e:
        logger.error(f"Failed to log resolved interaction: {e}")

    return {
        "final_response": draft_answer,
        "status": "resolved"
    }
