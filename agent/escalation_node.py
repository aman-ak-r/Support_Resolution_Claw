import json
import logging
from typing import Dict, Any
from agent.llm import call_escalation_node
from agent.logger import log_interaction

logger = logging.getLogger(__name__)

HIGH_KEYWORDS = [
    "fraud", "scam", "police", "legal", "notice", "lawyer",
    "hacked", "stolen", "hack", "blocked", "freeze", "chargeback",
    "dispute", "court", "compliance"
]

MEDIUM_KEYWORDS = [
    "kyc", "pan", "aadhaar", "verification", "reject",
    "otp", "pending", "failed transaction", "error 403", "payment failed"
]


def classify_severity_heuristics(query: str) -> str:
    q = query.lower()
    if any(k in q for k in HIGH_KEYWORDS):
        return "High"
    if any(k in q for k in MEDIUM_KEYWORDS):
        return "Medium"
    return "Low"


def run_escalation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    combined_confidence = state.get("combined_confidence", 0.0)

    if not retrieved_chunks:
        why_failed = "No matching documents found in the knowledge base."
    elif combined_confidence < 3.0:
        why_failed = f"Confidence score too low ({combined_confidence:.2f}) to generate a reliable answer."
    else:
        why_failed = "Escalated due to rule-based override."

    heuristic_severity = classify_severity_heuristics(query)

    formatted_chunks = [
        {
            "source": c["source"],
            "content_snippet": c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"]
        }
        for c in retrieved_chunks
    ]

    try:
        llm_severity, suggested_action = call_escalation_node(query, why_failed, json.dumps(formatted_chunks))
    except Exception as e:
        logger.error(f"LLM call failed in escalation node: {e}")
        llm_severity = "Medium"
        suggested_action = "Review merchant history and consult Eko support protocols."

    # Heuristic takes priority for High severity; otherwise we trust the LLM
    if heuristic_severity == "High":
        final_severity = "High"
    elif heuristic_severity == "Medium" and llm_severity != "High":
        final_severity = "Medium"
    else:
        final_severity = llm_severity

    escalation_ticket = {
        "query": query,
        "why_it_couldnt_be_resolved": why_failed,
        "severity": final_severity,
        "suggested_human_action": suggested_action,
        "retrieved_context": formatted_chunks
    }

    try:
        log_interaction(
            query=query,
            status="escalated",
            response=json.dumps(escalation_ticket, indent=2),
            confidence_score=combined_confidence,
            max_similarity=state.get("max_similarity", 0.0),
            severity=final_severity
        )
    except Exception as e:
        logger.error(f"Failed to log escalated interaction: {e}")

    escalation_markdown = f"""⚠️ **Your query has been escalated to support.**

An agent will contact you shortly.

- **Severity**: {final_severity}
- **Suggested Action**: {suggested_action}
"""

    return {
        "final_response": escalation_markdown,
        "escalation_note": escalation_ticket,
        "severity": final_severity,
        "status": "escalated"
    }
