import json
import logging
import re
from typing import Dict, Any, Tuple
import config

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain LLM libraries not fully installed. Falling back to Mock LLM.")
    LANGCHAIN_AVAILABLE = False


def _get_langchain_model():
    if not LANGCHAIN_AVAILABLE:
        raise ValueError("LangChain libraries are not available.")

    provider = config.LLM_PROVIDER
    if provider == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(model=config.OPENAI_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.0)
    elif provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, google_api_key=config.GEMINI_API_KEY, temperature=0.0)
    elif provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        return ChatAnthropic(model=config.ANTHROPIC_MODEL, api_key=config.ANTHROPIC_API_KEY, temperature=0.0)
    else:
        raise ValueError(f"Unsupported API LLM provider: {provider}")


def _mock_llm_decision(query: str, retrieved_context: str) -> Dict[str, Any]:
    query_lower = query.lower()
    
    # Check if context is missing
    if not retrieved_context.strip() or "No context" in retrieved_context:
        return {
            "draft_answer": "I'm sorry, I couldn't find any relevant SOP or FAQ documents in our knowledge base to answer your question.",
            "confidence": 1
        }
        
    # High risk / legal / fraud
    if any(k in query_lower for k in ["fraud", "scam", "police", "legal", "stolen", "hack", "court", "arrest"]):
        return {
            "draft_answer": "This request involves high-risk activities, compliance concerns, or legal/security notices. Standard SOP guidelines do not cover autonomous resolution of these cases.",
            "confidence": 1
        }
        
    # Wallet blockage / hold
    if any(k in query_lower for k in ["block", "freeze", "hold", "suspicious"]):
        if "unblock" in query_lower:
            return {
                "draft_answer": "To unblock a trade wallet frozen due to suspicious activity, the partner must raise a ticket in the Support Cockpit, upload the latest 3 months bank ledger statement, and complete an in-person KYC video verification within 48 hours.",
                "confidence": 5
            }
        return {
            "draft_answer": "Your wallet has been placed on hold due to compliance alerts. Self-resolution is restricted; manual review is required.",
            "confidence": 2
        }

    # CSP Registration / Deactivation
    if "csp" in query_lower:
        if "deactivat" in query_lower:
            return {
                "draft_answer": "A Customer Service Point (CSP) outlet can be deactivated either due to 90 days of inactivity, compliance violations, or by merchant request. Reactivation requires submitting a formal request via the distributor portal.",
                "confidence": 5
            }
        if "reactivat" in query_lower:
            return {
                "draft_answer": "To reactivate a deactivated CSP outlet: 1. Pay the reactivation fee of Rs. 500, 2. Resubmit latest shop photos with GPS coordinates, 3. Complete biometric re-verification within 3 working days.",
                "confidence": 5
            }
        return {
            "draft_answer": "To register a new Customer Service Point (CSP), you must fill the CSP registration form, pay a security deposit of Rs. 1000, and upload your shop registry deed and a clean police clearance certificate.",
            "confidence": 5
        }

    # Settlement delays / cycles
    if any(k in query_lower for k in ["settlement", "t+1", "t+2", "delay", "payout delay"]):
        if "t+1" in query_lower or "t+2" in query_lower or "cycle" in query_lower:
            return {
                "draft_answer": "Eko settlements run on a standard T+1 cycle (within 24 hours of transaction completion). For remote banks or bank holidays, settlements fallback to T+2 cycle. Transaction fees of 0.5% apply to instant payout settlements.",
                "confidence": 5
            }
        return {
            "draft_answer": "Settlement delays are typically caused by downstream beneficiary bank server downtime. Please check the Settlement Recon status. If pending for over 24 hours, raise a ticket with transaction UTR.",
            "confidence": 5
        }

    # Transaction failures
    if any(k in query_lower for k in ["timeout", "debit", "duplicate", "fail"]):
        if "timeout" in query_lower:
            return {
                "draft_answer": "For transaction timeouts where status is pending: wait 2 hours for auto-reconciliation. If the gateway updates to FAILED, the trade wallet will be refunded. Do not re-initiate transfer immediately.",
                "confidence": 5
            }
        if "partial debit" in query_lower or "debited" in query_lower:
            return {
                "draft_answer": "If a partial debit occurs (money cut from customer bank but transaction failed at Eko), the NPCI rules dictate that the bank must reverse the amount within 3 to 5 business days automatically.",
                "confidence": 5
            }
        if "duplicate" in query_lower:
            return {
                "draft_answer": "Duplicate transaction guard: Eko blocks identical transactions (same amount, same beneficiary) initiated within 5 minutes. If a duplicate debit occurs, raise a chargeback ticket with both UTRs.",
                "confidence": 5
            }

    # KYC re-verification / biometric
    if "kyc" in query_lower or "biometric" in query_lower:
        if "biometric" in query_lower or "mismatch" in query_lower:
            return {
                "draft_answer": "Biometric verification mismatch happens due to dirty scanner plates or low ink/dry fingers. SOP dictates cleaning the sensor, using biometric enhancer gel, and matching Aadhaar details. 3 attempts are allowed per hour.",
                "confidence": 5
            }
        if "re-verif" in query_lower or "reverif" in query_lower:
            return {
                "draft_answer": "A regular KYC re-verification is triggered every 12 months for compliance. Upload updated PAN card and Aadhaar XML document. Failure to complete re-verification within 7 days results in wallet limits being restricted to Rs. 10000.",
                "confidence": 5
            }

    # Standard FAQ simulation
    if "age" in query_lower or "minor" in query_lower or "18" in query_lower:
        return {
            "draft_answer": "To become an Eko partner, you must be at least 18 years of age. Minors cannot register due to legal and financial compliance regulations in India. Using another person's details is a violation of Eko's terms of service.",
            "confidence": 5
        }
    
    if any(k in query_lower for k in ["document", "onboard", "checklist", "required", "join"]):
        return {
            "draft_answer": "To onboard as an Eko merchant, you must submit: 1. PAN Card, 2. Aadhaar Card, 3. Active Bank Account (passbook/cheque), 4. Shop photos (inside/outside with board). Ensure all uploads are under 5MB and legible.",
            "confidence": 5
        }

    return {
        "draft_answer": f"Based on the retrieved knowledge base document: {retrieved_context[:200]}...",
        "confidence": 3
    }


def _mock_llm_escalation(query: str, why_failed: str) -> Dict[str, Any]:
    query_lower = query.lower()

    if any(k in query_lower for k in ["fraud", "scam", "police", "legal", "notice", "lawyer", "stolen", "hack"]):
        severity = "High"
        suggested_action = "Freeze trade wallet instantly, flag compliance risk tag on merchant record, and notify the compliance legal desk."
    elif any(k in query_lower for k in ["timeout", "debit", "duplicate", "block", "freeze", "settlement"]):
        severity = "High" if "chargeback" in query_lower else "Medium"
        suggested_action = "Examine gateway settlement logs, trace UTR status with beneficiary bank, and release pending fund reconciliations."
    elif any(k in query_lower for k in ["kyc", "pan", "aadhaar", "verification", "biometric"]):
        severity = "Medium"
        suggested_action = "Cross-reference uploaded ID documentation manually. Trigger SMS for biometric re-verification."
    else:
        severity = "Low"
        suggested_action = "Review merchant user flow history and assist via ticket reply."

    return {
        "severity": severity,
        "suggested_action": suggested_action
    }


def call_decision_node(query: str, retrieved_context: str, stricter: bool = False) -> Tuple[str, int]:
    # Test harness check for malformed-output testing
    if "__TEST_MALFORMED_OUTPUT__" in query:
        # Return invalid output (missing keys or string) to trigger schema validation errors
        if stricter:
            # If stricter retry is requested, let's still fail to verify the full escalation path
            return "STRICT_RETRY_MALFORMED_JSON", 99
        return "MALFORMED_JSON_STRING_NO_DICT", -5

    provider = config.LLM_PROVIDER
    if provider == "mock" or not LANGCHAIN_AVAILABLE:
        res = _mock_llm_decision(query, retrieved_context)
        return res["draft_answer"], res["confidence"]

    try:
        model = _get_langchain_model()
        
        system_prompt = """You are the AI engine of the 'Support Resolution Claw' agent.
Your task is to draft a response to a support query based ONLY on the provided FAQ/SOP knowledge base chunks.
You must also rate your confidence on a scale of 1 to 5:
- 5: The retrieved context provides a complete, clear, and direct answer to the query.
- 4: The context covers the query, but minor details are missing or require slight deduction.
- 3: The context relates to the topic but is missing direct details, or only partially answers the query.
- 2: The context is highly incomplete or only vaguely related to the query.
- 1: The context does not answer the query at all, or the query is completely outside the knowledge base scope (e.g. fraud, legal threats, or off-topic requests).

You MUST output your response in strict JSON format with exactly two keys:
{
  "draft_answer": "your drafted response to the user based on the context",
  "confidence": 5
}
"""
        if stricter:
            system_prompt += "\nWARNING: Your previous response was malformed or failed schema check. You MUST return ONLY a valid JSON object. No markdown wrappers or extra characters."

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "User Query: {query}\n\nRetrieved Knowledge Base Chunks:\n{context}")
        ])

        chain = prompt | model
        response = chain.invoke({"query": query, "context": retrieved_context})
        
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            draft_answer = data["draft_answer"]
            confidence = int(data["confidence"])
            return draft_answer, confidence
        else:
            raise ValueError("No JSON object found in LLM response.")
            
    except Exception as e:
        logger.error(f"Error in decision node LLM call (stricter={stricter}): {e}")
        raise e


def call_escalation_node(query: str, why_failed: str, retrieved_context: str, stricter: bool = False) -> Tuple[str, str]:
    provider = config.LLM_PROVIDER
    if provider == "mock" or not LANGCHAIN_AVAILABLE:
        res = _mock_llm_escalation(query, why_failed)
        return res["severity"], res["suggested_action"]

    try:
        model = _get_langchain_model()
        
        system_prompt = """You are the support supervisor of the 'Support Resolution Claw' agent.
An incoming customer query could not be resolved autonomously because the knowledge base coverage was insufficient or the request involves high-risk events.

Your task is to classify this ticket's severity:
- High: mentions of money/payment failure, fraud, account blocked, legal/compliance threats, or hacking.
- Medium: KYC issues, repeated transaction failures, technical errors blocking partner operations.
- Low: general how-to questions, minor app navigation issues that lack direct KB coverage.

Also generate a specific, actionable suggested human action for Eko support agents.

You MUST output your response in strict JSON format with exactly two keys:
{
  "severity": "Low" | "Medium" | "High",
  "suggested_action": "what a human agent should do step-by-step"
}
"""
        if stricter:
            system_prompt += "\nWARNING: Your previous response was malformed. You MUST return ONLY valid JSON with severity and suggested_action keys."

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "User Query: {query}\nReason why unresolved: {why_failed}\nRetrieved context (for reference):\n{context}")
        ])

        chain = prompt | model
        response = chain.invoke({"query": query, "why_failed": why_failed, "context": retrieved_context})
        
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            severity = data["severity"]
            suggested_action = data["suggested_action"]
            return severity, suggested_action
        else:
            raise ValueError("No JSON object found in escalation LLM response.")
            
    except Exception as e:
        logger.error(f"Error in escalation node LLM call (stricter={stricter}): {e}")
        raise e


def call_verification_node(query: str, retrieved_context: str, draft_answer: str) -> Tuple[bool, str]:
    """
    Self-Verification step: Runs a fresh LLM call to verify if the generated answer is fully
    supported by the context and answers the query. No conversation history.
    """
    # Test case overrides
    if "__TEST_VERIFICATION_FAIL__" in query:
        return False, "Failed self-verification check mock target."

    provider = config.LLM_PROVIDER
    if provider == "mock" or not LANGCHAIN_AVAILABLE:
        # Mock verification logic: verify that draft answer doesn't contain fallback error messages
        if "couldn't find" in draft_answer or "Could not generate" in draft_answer or not retrieved_context.strip():
            return False, "No relevant context available to substantiate the answer."
        return True, "Answer is supported by the retrieved SOP documents."

    try:
        model = _get_langchain_model()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a rigorous quality assurance auditor verifying customer support answers.
Evaluate the draft answer using only the provided context and the customer query.

You must determine:
1. Does this answer actually address the customer's query?
2. Is the answer based ONLY on the provided retrieved context?
3. Does the answer contain any hallucinations or unsupported assertions?

You MUST output your response in strict JSON format with exactly two keys:
{
  "verified": true | false,
  "reason": "a brief explanation of your verification decision"
}
Do not include any other text.
"""),
            ("user", "Query: {query}\n\nRetrieved Context:\n{context}\n\nDraft Answer:\n{answer}")
        ])

        chain = prompt | model
        response = chain.invoke({"query": query, "context": retrieved_context, "answer": draft_answer})
        
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            verified = bool(data.get("verified", False))
            reason = data.get("reason", "Verification completed.")
            return verified, reason
        else:
            return False, "Malformed verification JSON output from LLM."
            
    except Exception as e:
        logger.error(f"Verification node LLM call failed: {e}")
        return False, f"LLM execution error: {e}"
