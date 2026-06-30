import logging
from typing import Dict, Any
import config
from agent.llm import call_decision_node
from agent.schemas import DecisionInput, DecisionOutput

logger = logging.getLogger(__name__)


def run_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Validate Input Contract
    inputs = DecisionInput(
        query=state.get("query", ""),
        retrieved_chunks=state.get("retrieved_chunks", []),
        attempts=state.get("attempts", 1)
    )

    query = inputs.query
    retrieved_chunks = inputs.retrieved_chunks
    attempts = inputs.attempts

    if retrieved_chunks:
        context_text = "\n\n".join([
            f"--- SOURCE: {c['source']} ---\n{c['content']}"
            for c in retrieved_chunks
        ])
        max_similarity = max(c["similarity"] for c in retrieved_chunks)
    else:
        context_text = "No context retrieved."
        max_similarity = 0.0

    # 2. Call LLM with Pydantic Validation & Stricter Retry
    draft_answer = ""
    llm_confidence = 1
    malformed_exception = None

    try:
        draft_answer, llm_confidence = call_decision_node(query, context_text, stricter=False)
        # Validate values are realistic
        if not (1 <= llm_confidence <= 5):
            raise ValueError(f"Confidence score {llm_confidence} out of range [1, 5]")
    except Exception as first_error:
        logger.warning(f"First decision LLM attempt malformed/failed: {first_error}. Retrying with stricter prompt...")
        try:
            # Stricter prompt retry call
            draft_answer, llm_confidence = call_decision_node(query, context_text, stricter=True)
            if not (1 <= llm_confidence <= 5):
                raise ValueError(f"Confidence score {llm_confidence} out of range [1, 5]")
        except Exception as retry_error:
            logger.error(f"Stricter retry failed as well: {retry_error}. Routing to malformed_output exception path.")
            malformed_exception = retry_error

    # 3. Handle Malformed Output Exception Path
    if malformed_exception is not None:
        outputs = DecisionOutput(
            draft_answer="The automated decision system generated a malformed response.",
            llm_confidence=1.0,
            max_similarity=max_similarity,
            combined_confidence=1.0,
            routing_decision="malformed_output",
            attempts=attempts
        )
        return outputs.model_dump()

    # Calculate combined score
    combined_score = (llm_confidence * config.LLM_CONFIDENCE_WEIGHT) + \
                     ((max_similarity * 5.0) * config.SIMILARITY_WEIGHT)
    combined_score = max(1.0, min(5.0, combined_score))

    # Circuit breaker: on the 2nd attempt, if we still have low confidence, or we just want to route:
    routing_decision = "answer" if combined_score >= config.CONFIDENCE_THRESHOLD else "escalate"

    # 4. Validate Output Contract
    outputs = DecisionOutput(
        draft_answer=draft_answer,
        llm_confidence=float(llm_confidence),
        max_similarity=float(max_similarity),
        combined_confidence=float(combined_score),
        routing_decision=routing_decision,
        attempts=attempts + 1 if routing_decision == "escalate" else attempts
    )

    logger.info(f"Decision Node → LLM: {llm_confidence}, Score: {combined_score:.2f}, Route: {routing_decision}")
    return outputs.model_dump()
