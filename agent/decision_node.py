import logging
from typing import Dict, Any
import config
from agent.llm import call_decision_node

logger = logging.getLogger(__name__)


def run_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    if retrieved_chunks:
        context_text = "\n\n".join([
            f"--- SOURCE: {c['source']} ---\n{c['content']}"
            for c in retrieved_chunks
        ])
        max_similarity = max(c["similarity"] for c in retrieved_chunks)
    else:
        context_text = "No context retrieved."
        max_similarity = 0.0

    try:
        draft_answer, llm_confidence = call_decision_node(query, context_text)
    except Exception as e:
        logger.error(f"LLM call failed in decision node: {e}")
        draft_answer = "Could not generate a response."
        llm_confidence = 1

    # Weighted combination of LLM self-rating and FAISS similarity score
    combined_score = (llm_confidence * config.LLM_CONFIDENCE_WEIGHT) + \
                     ((max_similarity * 5.0) * config.SIMILARITY_WEIGHT)
    combined_score = max(1.0, min(5.0, combined_score))

    routing_decision = "answer" if combined_score >= config.CONFIDENCE_THRESHOLD else "escalate"

    logger.info(f"Decision → LLM: {llm_confidence}, Score: {combined_score:.2f}, Route: {routing_decision}")

    return {
        "draft_answer": draft_answer,
        "llm_confidence": float(llm_confidence),
        "max_similarity": float(max_similarity),
        "combined_confidence": float(combined_score),
        "routing_decision": routing_decision
    }
