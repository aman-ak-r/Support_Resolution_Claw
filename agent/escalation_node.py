import json
import logging
from typing import Dict, Any
from agent.llm import call_escalation_node
from agent.logger import log_interaction, create_ticket
from agent.schemas import EscalationInput, EscalationOutput

logger = logging.getLogger(__name__)

HIGH_KEYWORDS = [
    "fraud", "scam", "police", "legal", "notice", "lawyer",
    "hacked", "stolen", "hack", "blocked", "freeze", "chargeback",
    "dispute", "court", "compliance"
]

MEDIUM_KEYWORDS = [
    "kyc", "pan", "aadhaar", "verification", "reject",
    "otp", "pending", "failed transaction", "error 403", "payment failed",
    "biometric", "mismatch", "timeout", "partial debit", "duplicate",
    "csp", "deactivation", "reactivation"
]


def classify_severity_heuristics(query: str) -> str:
    q = query.lower()
    if any(k in q for k in HIGH_KEYWORDS):
        return "High"
    if any(k in q for k in MEDIUM_KEYWORDS):
        return "Medium"
    return "Low"


def run_escalation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Validate Input Contract
    inputs = EscalationInput(
        query=state.get("query", ""),
        retrieved_chunks=state.get("retrieved_chunks", []),
        combined_confidence=state.get("combined_confidence", 0.0),
        max_similarity=state.get("max_similarity", 0.0),
        why_failed=state.get("why_failed", "")
    )

    query = inputs.query
    retrieved_chunks = inputs.retrieved_chunks
    combined_confidence = inputs.combined_confidence
    max_similarity = inputs.max_similarity
    why_failed = inputs.why_failed

    if not why_failed:
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

    # 2. Call LLM for severity classification with stricter retry on malformed JSON
    llm_severity = "Medium"
    suggested_action = "Review merchant query details manually."
    malformed_exception = None

    try:
        llm_severity, suggested_action = call_escalation_node(query, why_failed, json.dumps(formatted_chunks), stricter=False)
        if llm_severity not in ("Low", "Medium", "High"):
            raise ValueError(f"Invalid severity value: {llm_severity}")
    except Exception as first_error:
        logger.warning(f"First escalation LLM call failed: {first_error}. Retrying with stricter instructions...")
        try:
            llm_severity, suggested_action = call_escalation_node(query, why_failed, json.dumps(formatted_chunks), stricter=True)
            if llm_severity not in ("Low", "Medium", "High"):
                raise ValueError(f"Invalid severity value: {llm_severity}")
        except Exception as retry_error:
            logger.error(f"Stricter retry failed as well: {retry_error}. Forcing Low severity malformed path.")
            malformed_exception = retry_error

    # 3. Handle Malformed Output Exception Path
    if malformed_exception is not None:
        final_severity = "Low"
        suggested_action = "Automated escalation failed verification. Review malformed ticket details manually."
        why_failed = f"Escalation LLM generation failed/malformed: {malformed_exception}"
    else:
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

    # 4. Create persistent ticket in SQLite (returns UUID)
    ticket_id = create_ticket(query, final_severity, escalation_ticket)

    # 5. Log interaction to SQLite database
    try:
        log_interaction(
            query=query,
            status="escalated",
            response=json.dumps(escalation_ticket, indent=2),
            confidence_score=combined_confidence,
            max_similarity=max_similarity,
            severity=final_severity
        )
    except Exception as e:
        logger.error(f"Failed to log escalated interaction to SQLite: {e}")

    escalation_markdown = f"""⚠️ **Your query has been escalated to support.**

An agent will contact you shortly.

- **Ticket ID**: `{ticket_id}`
- **Severity**: `{final_severity}`
- **Suggested Action**: {suggested_action}
"""

    # 6. Validate Output Contract
    outputs = EscalationOutput(
        final_response=escalation_markdown,
        escalation_note=escalation_ticket,
        severity=final_severity,
        status="escalated",
        ticket_id=ticket_id
    )

    return outputs.model_dump()
