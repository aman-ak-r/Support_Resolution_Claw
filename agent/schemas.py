from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class RetrievalInput(BaseModel):
    query: str


class RetrievalOutput(BaseModel):
    query: str
    retrieved_chunks: List[Dict[str, Any]]


class DecisionInput(BaseModel):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    attempts: int


class DecisionOutput(BaseModel):
    draft_answer: str
    llm_confidence: float
    max_similarity: float
    combined_confidence: float
    routing_decision: str
    attempts: Optional[int] = None


class AnswerInput(BaseModel):
    query: str
    draft_answer: str
    combined_confidence: float
    max_similarity: float


class AnswerOutput(BaseModel):
    final_response: str
    status: str


class VerificationInput(BaseModel):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    draft_answer: str


class VerificationOutput(BaseModel):
    verified: bool
    reason: str


class EscalationInput(BaseModel):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    combined_confidence: float
    max_similarity: float
    why_failed: str


class EscalationOutput(BaseModel):
    final_response: str
    escalation_note: Dict[str, Any]
    severity: str
    status: str
    ticket_id: str


class LoggingInput(BaseModel):
    query: str
    status: str
    response: str
    confidence_score: float
    max_similarity: float
    severity: Optional[str] = None


class LoggingOutput(BaseModel):
    log_id: int
