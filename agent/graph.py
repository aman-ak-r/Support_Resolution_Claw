import logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from agent.retrieval import KnowledgeBaseRetriever
from agent.decision_node import run_decision_node
from agent.answer_node import run_answer_node
from agent.escalation_node import run_escalation_node
from agent.llm import call_verification_node
from agent.logger import log_interaction
from agent.schemas import RetrievalInput, RetrievalOutput, VerificationInput, VerificationOutput

logger = logging.getLogger(__name__)

retriever = KnowledgeBaseRetriever()


class AgentState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    draft_answer: str
    llm_confidence: float
    max_similarity: float
    combined_confidence: float
    routing_decision: str  # "answer", "escalate", "retry", "malformed_output", "resolved"
    final_response: str
    severity: str
    escalation_note: Dict[str, Any]
    status: str
    attempts: int
    why_failed: str


def run_retrieval_node(state: AgentState) -> Dict[str, Any]:
    # 1. Validate Input Contract
    inputs = RetrievalInput(query=state["query"])
    query = inputs.query
    attempts = state.get("attempts", 1)

    logger.info(f"Graph Entry (Attempt {attempts}): Retrieving FAQ chunks for query: '{query}'")
    chunks = retriever.retrieve_chunks(query)

    if not chunks:
        logger.warning(f"Retrieval returned 0 chunks on attempt {attempts}.")
        if attempts >= 2:
            # Hard cap hit — force escalation directly from retrieval node
            logger.error("Retrieval failed after max attempts. Forcing escalation.")
            return {
                "retrieved_chunks": [],
                "routing_decision": "escalate",
                "why_failed": "No matching documents found in the knowledge base after retries."
            }
        # First attempt: let the decision node handle low-confidence routing naturally
        logger.info("Empty retrieval on attempt 1. Decision node will score low and trigger retry.")

    # 2. Validate Output Contract
    outputs = RetrievalOutput(query=query, retrieved_chunks=chunks)
    return outputs.model_dump()


def run_verification_node(state: AgentState) -> Dict[str, Any]:
    # 1. Validate Input Contract
    inputs = VerificationInput(
        query=state.get("query", ""),
        retrieved_chunks=state.get("retrieved_chunks", []),
        draft_answer=state.get("draft_answer", "")
    )

    attempts = state.get("attempts", 1)
    
    logger.info(f"Self-Verification Node (Attempt {attempts}): Verifying answer validity...")
    
    # Format chunks text for verification context
    context_text = "\n\n".join([
        f"--- SOURCE: {c['source']} ---\n{c['content']}"
        for c in inputs.retrieved_chunks
    ]) if inputs.retrieved_chunks else "No context retrieved."

    verified, reason = call_verification_node(inputs.query, context_text, inputs.draft_answer)
    logger.info(f"Verification Result: {verified} | Reason: {reason}")

    # 2. Validate Output Contract
    outputs = VerificationOutput(verified=verified, reason=reason)
    
    if verified:
        # If verification passed, log resolved interaction to SQLite
        try:
            log_interaction(
                query=inputs.query,
                status="resolved",
                response=inputs.draft_answer,
                confidence_score=state.get("combined_confidence", 5.0),
                max_similarity=state.get("max_similarity", 1.0),
                severity=None
            )
        except Exception as e:
            logger.error(f"Failed to log resolved interaction: {e}")

        return {
            "routing_decision": "resolved",
            "final_response": inputs.draft_answer,
            "status": "resolved"
        }
    else:
        # Verification failed: check retry loop
        if attempts < 2:
            logger.warning(f"Self-verification failed on attempt {attempts}. Retrying entire pipeline workflow...")
            return {
                "routing_decision": "retry",
                "attempts": attempts + 1,
                "why_failed": f"Attempt {attempts} failed self-verification: {reason}"
            }
        else:
            logger.error("Self-verification failed after all retries. Routing to escalation.")
            return {
                "routing_decision": "escalate",
                "why_failed": f"Failed self-verification: {reason} after retries."
            }


def route_decision(state: AgentState) -> str:
    decision = state["routing_decision"]
    attempts = state.get("attempts", 1)

    # Malformed output exception path always goes to escalation
    if decision == "malformed_output":
        logger.error("Routing to escalation due to malformed output exception path.")
        return "escalate"

    # If retrieval node already set escalate directly (e.g. empty chunks after retries), pass through
    if decision == "escalate":
        if attempts < 2:
            logger.info(f"Decision node returned escalate on attempt {attempts}. Retrying pipeline...")
            return "retry"
        else:
            logger.warning("Hard cap reached (attempt 2). Routing to escalation.")
            return "escalate"

    return decision


def route_retrieve(state: AgentState) -> str:
    """Short-circuit to escalation if retrieval returned empty chunks and attempts are exhausted."""
    if state.get("routing_decision") == "escalate" and not state.get("retrieved_chunks"):
        return "escalate"
    return "decision"


def route_verification(state: AgentState) -> str:
    return state["routing_decision"]


# Construct Workflow
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", run_retrieval_node)
workflow.add_node("decision", run_decision_node)
workflow.add_node("answer", run_answer_node)
workflow.add_node("verify", run_verification_node)
workflow.add_node("escalation", run_escalation_node)

workflow.set_entry_point("retrieve")

# Retrieval can short-circuit to escalation when chunks are empty after max retries
workflow.add_conditional_edges(
    "retrieve",
    route_retrieve,
    {
        "decision": "decision",
        "escalate": "escalation"
    }
)

# Decision Routing
workflow.add_conditional_edges(
    "decision",
    route_decision,
    {
        "answer": "answer",
        "escalate": "escalation",
        "retry": "retrieve"  # Loop back to retrieve for retry
    }
)

workflow.add_edge("answer", "verify")

# Verification Routing
workflow.add_conditional_edges(
    "verify",
    route_verification,
    {
        "resolved": END,
        "escalate": "escalation",
        "retry": "retrieve"  # Loop back to retrieve for retry
    }
)

workflow.add_edge("escalation", END)

compiled_app = workflow.compile()


def run_pipeline(query: str) -> Dict[str, Any]:
    initial_state = {
        "query": query,
        "retrieved_chunks": [],
        "draft_answer": "",
        "llm_confidence": 0.0,
        "max_similarity": 0.0,
        "combined_confidence": 0.0,
        "routing_decision": "escalate",
        "final_response": "",
        "severity": "Low",
        "escalation_note": {},
        "status": "",
        "attempts": 1,
        "why_failed": ""
    }
    return compiled_app.invoke(initial_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_pipeline("What documents are required to register as Eko partner?")
    print(f"Status: {result['status']}")
    print(f"Attempts: {result['attempts']}")
    print(f"Response:\n{result['final_response']}")
