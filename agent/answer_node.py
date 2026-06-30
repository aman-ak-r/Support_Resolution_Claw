import logging
from typing import Dict, Any
from agent.schemas import AnswerInput, AnswerOutput

logger = logging.getLogger(__name__)


def run_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Validate Input Contract
    inputs = AnswerInput(
        query=state.get("query", ""),
        draft_answer=state.get("draft_answer", "No answer was generated."),
        combined_confidence=state.get("combined_confidence", 0.0),
        max_similarity=state.get("max_similarity", 0.0)
    )

    final_response = inputs.draft_answer

    # 2. Validate Output Contract
    outputs = AnswerOutput(
        final_response=final_response,
        status="resolved"
    )

    return outputs.model_dump()
