import sys
from pathlib import Path

# Add project root to sys.path so we can import modules correctly during test execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import config
from agent.decision_node import run_decision_node

def test_decision_node_high_confidence():
    """
    Test that the decision node correctly routes to 'answer' when LLM confidence 
    and FAISS similarity scores are high.
    
    Expected logic:
      - Query: standard FAQ
      - Retrieved chunks have high similarity
      - Combined score >= threshold (3.0) -> route to 'answer'
    """
    # Create a mock state matching AgentState shape
    mock_state = {
        "query": "What documents are required to register as Eko partner?",
        "retrieved_chunks": [
            {"source": "onboarding_checklist.md", "content": "You must submit PAN and Aadhaar.", "similarity": 0.9}
        ],
        "draft_answer": "",
        "llm_confidence": 0.0,
        "max_similarity": 0.0,
        "combined_confidence": 0.0,
        "routing_decision": "",
        "final_response": "",
        "severity": "",
        "escalation_note": {},
        "status": ""
    }

    # Execute decision node logic
    result = run_decision_node(mock_state)

    # Assertions
    assert result["routing_decision"] == "answer"
    assert result["combined_confidence"] >= config.CONFIDENCE_THRESHOLD
    assert result["max_similarity"] == 0.9
    assert result["llm_confidence"] >= 3.0

def test_decision_node_low_confidence():
    """
    Test that the decision node correctly routes to 'escalate' when similarity 
    and LLM confidence are low (e.g. out-of-scope query).
    
    Expected logic:
      - Query: out-of-scope or high risk
      - Retrieved chunks have low similarity (or empty context)
      - Combined score < threshold (3.0) -> route to 'escalate'
    """
    mock_state = {
        "query": "My account has been hacked and Rs 50000 stolen, arrest them!",
        "retrieved_chunks": [],  # Empty context
        "draft_answer": "",
        "llm_confidence": 0.0,
        "max_similarity": 0.0,
        "combined_confidence": 0.0,
        "routing_decision": "",
        "final_response": "",
        "severity": "",
        "escalation_note": {},
        "status": ""
    }

    result = run_decision_node(mock_state)

    assert result["routing_decision"] == "escalate"
    assert result["combined_confidence"] < config.CONFIDENCE_THRESHOLD
    assert result["max_similarity"] == 0.0
    assert result["llm_confidence"] == 1.0  # Empty context should rate low

def test_combined_confidence_formula():
    """
    Verify that the mathematical combination of LLM confidence and FAISS similarity 
    respects our config weight ratios.
    
    Formula: score = (llm_conf * 0.7) + ((similarity * 5.0) * 0.3)
    """
    # Case: LLM rates 4/5, FAISS similarity is 0.8.
    # Expected: (4 * 0.7) + (0.8 * 5 * 0.3) = 2.8 + 1.2 = 4.0
    mock_state = {
        "query": "What is the age requirement to join?",
        "retrieved_chunks": [
            {"source": "onboarding_age_limit.md", "content": "You must be 18.", "similarity": 0.8}
        ],
    }
    
    # We will override the config weights temporarily to guarantee mathematical values
    original_llm_w = config.LLM_CONFIDENCE_WEIGHT
    original_sim_w = config.SIMILARITY_WEIGHT
    
    config.LLM_CONFIDENCE_WEIGHT = 0.7
    config.SIMILARITY_WEIGHT = 0.3
    
    try:
        # Note run_decision_node will call the LLM/mock. For "age" query and 0.8 similarity, 
        # the mock returns confidence=5.
        # Combined score should be: (5 * 0.7) + (0.8 * 5 * 0.3) = 3.5 + 1.2 = 4.7
        result = run_decision_node(mock_state)
        
        assert result["llm_confidence"] == 5.0
        assert result["max_similarity"] == 0.8
        assert pytest.approx(result["combined_confidence"], 0.01) == 4.7
    finally:
        # Restore configuration values
        config.LLM_CONFIDENCE_WEIGHT = original_llm_w
        config.SIMILARITY_WEIGHT = original_sim_w
