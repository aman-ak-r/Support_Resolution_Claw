import logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from agent.retrieval import KnowledgeBaseRetriever
from agent.decision_node import run_decision_node
from agent.answer_node import run_answer_node
from agent.escalation_node import run_escalation_node

logger = logging.getLogger(__name__)

retriever = KnowledgeBaseRetriever()


class AgentState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    draft_answer: str
    llm_confidence: float
    max_similarity: float
    combined_confidence: float
    routing_decision: str
    final_response: str
    severity: str
    escalation_note: Dict[str, Any]
    status: str


def run_retrieval_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    chunks = retriever.retrieve_chunks(query)
    return {"retrieved_chunks": chunks}


def route_decision(state: AgentState) -> str:
    return state["routing_decision"]


workflow = StateGraph(AgentState)

workflow.add_node("retrieve", run_retrieval_node)
workflow.add_node("decision", run_decision_node)
workflow.add_node("answer", run_answer_node)
workflow.add_node("escalation", run_escalation_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "decision")
workflow.add_conditional_edges(
    "decision",
    route_decision,
    {"answer": "answer", "escalate": "escalation"}
)
workflow.add_edge("answer", END)
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
        "status": ""
    }
    return compiled_app.invoke(initial_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_pipeline("What documents are required to register as Eko partner?")
    print(f"Status: {result['status']}")
    print(f"Response:\n{result['final_response']}")
