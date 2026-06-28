import json
import logging
import re
from typing import Dict, Any, Tuple
import config

logger = logging.getLogger(__name__)

# Try importing LangChain models. If they are missing, we fall back to mock.
try:
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain LLM libraries not fully installed. Falling back to Mock LLM.")
    LANGCHAIN_AVAILABLE = False


def _get_langchain_model():
    """
    Instantiates and returns the configured LangChain chat model based on config.py.
    """
    if not LANGCHAIN_AVAILABLE:
        raise ValueError("LangChain libraries are not available.")

    provider = config.LLM_PROVIDER
    if provider == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment.")
        return ChatOpenAI(model=config.OPENAI_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.0)
    elif provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment.")
        return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, google_api_key=config.GEMINI_API_KEY, temperature=0.0)
    elif provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment.")
        return ChatAnthropic(model=config.ANTHROPIC_MODEL, api_key=config.ANTHROPIC_API_KEY, temperature=0.0)
    else:
        raise ValueError(f"Unsupported API LLM provider: {provider}")


def _mock_llm_decision(query: str, retrieved_context: str) -> Dict[str, Any]:
    """
    Simulates the decision node logic for Mock provider.
    Analyzes the query and retrieved context to output a draft answer and self-confidence score.
    """
    query_lower = query.lower()
    
    # Check if we have context at all
    if not retrieved_context.strip():
        return {
            "draft_answer": "I'm sorry, I couldn't find any relevant SOP or FAQ documents in our knowledge base to answer your question.",
            "confidence": 1
        }
        
    # Check for keywords that represent high risk / escalation queries
    # High risk / legal / fraud
    if any(k in query_lower for k in ["fraud", "scam", "police", "legal", "stolen", "hack", "court", "arrest"]):
        return {
            "draft_answer": "This request involves high-risk activities, compliance concerns, or legal/security notices. Standard SOP guidelines do not cover autonomous resolution of these cases.",
            "confidence": 1
        }
        
    # Significant payment failure / wallet blockage
    if any(k in query_lower for k in ["blocked", "wallet freeze", "chargeback dispute", "recovery"]):
        # Chargebacks are handled via custom processes
        if "chargeback" in query_lower or "dispute" in query_lower:
            return {
                "draft_answer": "A chargeback dispute requires merchant verification files. Standard support rules demand freezing the amount and requesting documentation within 48 hours.",
                "confidence": 2
            }
        return {
            "draft_answer": "Your account/wallet appears to have a security freeze or block. Standard FAQs suggest resetting details, but immediate escalation is required.",
            "confidence": 2
        }

    # Standard FAQ simulation based on query content matching context
    # Let's extract the first markdown heading or clean it up for draft answers
    # Age Limit
    if "age" in query_lower or "minor" in query_lower or "18" in query_lower:
        return {
            "draft_answer": "To become an Eko partner, you must be at least 18 years of age. Minors cannot register due to legal and financial compliance regulations in India. Using another person's details is a violation of Eko's terms of service.",
            "confidence": 5
        }
    
    # Onboarding Checklist
    if any(k in query_lower for k in ["document", "onboard", "checklist", "required", "join"]):
        return {
            "draft_answer": "To onboard as an Eko merchant, you must submit: 1. PAN Card, 2. Aadhaar Card, 3. Active Bank Account (passbook/cheque), 4. Shop photos (inside/outside with board). Ensure all uploads are under 5MB and legible.",
            "confidence": 5
        }

    # Payout / Commission Slabs
    if any(k in query_lower for k in ["commission", "slab", "rate", "earn", "tds"]):
        if "tds" in query_lower or "tax" in query_lower:
            return {
                "draft_answer": "Eko deducts 5% TDS under Section 194H of the Income Tax Act on commissions paid to PAN-verified partners. If PAN is not provided, TDS is deducted at 20%. Quarterly TDS certificates (Form 16A) are available on the portal.",
                "confidence": 5
            }
        return {
            "draft_answer": "Eko commissions vary by service. For DMT, earn up to 0.50% (capped at Rs. 15). For AePS cash withdrawals, earn flat commission based on slabs (e.g. Rs. 9 for Rs. 3001-10000 transactions). Micro ATM payouts follow the same slabs.",
            "confidence": 5
        }

    # Transaction Failures (DMT / AePS)
    if any(k in query_lower for k in ["dmt", "money transfer", "aeps", "aadhaar withdrawal", "refund"]):
        if "aeps" in query_lower:
            return {
                "draft_answer": "If an AePS transaction fails but the customer's bank account was debited, the NPCI guidelines dictate that the bank must reverse the amount automatically within 3 to 5 business days. Eko does not hold these funds.",
                "confidence": 5
            }
        if "dmt" in query_lower or "transfer" in query_lower:
            return {
                "draft_answer": "If a Domestic Money Transfer (DMT) fails: 1. Status FAILED: refunded to Trade Wallet instantly. 2. Status PENDING: wait 2 hours for auto-resolution to SUCCESS/FAILED. 3. Status SUCCESS: share the UTR with the beneficiary.",
                "confidence": 5
            }
        return {
            "draft_answer": "Failed transaction refunds: IMPS is refunded within 10 mins. NEFT is refunded within 24 hours. Pending gateway disputes can take up to 72 hours to resolve and initiate credit.",
            "confidence": 4
        }

    # Device binding block
    if any(k in query_lower for k in ["device blocked", "device registration", "device limit"]):
        return {
            "draft_answer": "Eko allows login on only one device at a time. If you switched phones or reinstalled the app, you must reset your device binding. Go to Profile -> Reset Device in the web portal or contact your distributor to receive an OTP and unlock.",
            "confidence": 5
        }

    # Password / MPIN reset
    if any(k in query_lower for k in ["password", "pin", "mpin", "reset"]):
        return {
            "draft_answer": "To reset password or MPIN: Click 'Forgot Password' or 'Forgot MPIN' on the login screen, enter your registered number, input the OTP received, and configure your new login/transaction pin. Do not share credentials.",
            "confidence": 5
        }

    # If we have some context, let's make a generic draft answer and give confidence 3
    return {
        "draft_answer": f"Based on the retrieved knowledge base document: {retrieved_context[:200]}...",
        "confidence": 3
    }


def _mock_llm_escalation(query: str, why_failed: str) -> Dict[str, Any]:
    """
    Simulates the escalation severity classification and suggested human actions for Mock provider.
    """
    query_lower = query.lower()

    # Rule-based classification combined with LLM logic simulator
    if any(k in query_lower for k in ["fraud", "scam", "police", "legal", "notice", "lawyer", "stolen", "hack", "cyber"]):
        severity = "High"
        suggested_action = "Immediately freeze merchant's Eko Trade Wallet, flag the account for compliance review, and prepare transaction logs for cyber-cell investigation."
    elif any(k in query_lower for k in ["money", "payout", "payment failure", "blocked", "wallet freeze", "chargeback", "dispute"]):
        severity = "High" if "chargeback" in query_lower or "fraud" in query_lower else "Medium"
        if severity == "High":
            suggested_action = "Initiate chargeback dispute protocol: hold disputed funds in trade wallet and request merchant receipt/ledger registry proof within 48 hours."
        else:
            suggested_action = "Check transaction status in backend gateway. Cross-reference partner bank reports and initiate manual refund if transaction failed but was not auto-reversed."
    elif any(k in query_lower for k in ["kyc", "pan", "aadhaar", "verification", "reject", "otp"]):
        severity = "Medium"
        suggested_action = "Review uploaded documents manually in admin panel. Check UIDAI or NSDL verification response codes. Advise merchant on name spellings if mismatch found."
    else:
        severity = "Low"
        suggested_action = "Follow up with standard training resources. Guide merchant on app navigation or general registration procedures."

    return {
        "severity": severity,
        "suggested_action": suggested_action
    }


def call_decision_node(query: str, retrieved_context: str) -> Tuple[str, int]:
    """
    Asks the LLM to draft an answer and self-rate its confidence (1-5 scale) 
    based on the provided retrieved context.
    
    Returns:
        (draft_answer, confidence_score)
    """
    provider = config.LLM_PROVIDER
    if provider == "mock" or not LANGCHAIN_AVAILABLE:
        res = _mock_llm_decision(query, retrieved_context)
        return res["draft_answer"], res["confidence"]

    try:
        model = _get_langchain_model()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the AI engine of the 'Support Resolution Claw' agent.
Your task is to draft a response to a support query based ONLY on the provided FAQ/SOP knowledge base chunks.
You must also rate your confidence on a scale of 1 to 5:
- 5: The retrieved context provides a complete, clear, and direct answer to the query.
- 4: The context covers the query, but minor details are missing or require slight deduction.
- 3: The context relates to the topic but is missing direct details, or only partially answers the query.
- 2: The context is highly incomplete or only vaguely related to the query.
- 1: The context does not answer the query at all, or the query is completely outside the knowledge base scope (e.g. fraud, legal threats, or off-topic requests).

You MUST output your response in strict JSON format with exactly two keys:
{{
  "draft_answer": "your drafted response to the user based on the context",
  "confidence": 5
}}
Do not include any other text, markdown formatting (other than JSON itself), or explanations.
"""),
            ("user", "User Query: {query}\n\nRetrieved Knowledge Base Chunks:\n{context}")
        ])

        # Execute LLM chain
        chain = prompt | model
        response = chain.invoke({"query": query, "context": retrieved_context})
        
        # Parse the JSON response
        # Using a regex to extract JSON if LLM returns markdown blocks
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            draft_answer = data.get("draft_answer", "No draft answer generated.")
            confidence = int(data.get("confidence", 1))
            return draft_answer, confidence
        else:
            logger.error("LLM did not return a valid JSON format. Falling back to score 1.")
            return response.content, 1
            
    except Exception as e:
        logger.error(f"Error invoking LLM in decision node: {e}. Falling back to Mock.")
        res = _mock_llm_decision(query, retrieved_context)
        return res["draft_answer"], res["confidence"]


def call_escalation_node(query: str, why_failed: str, retrieved_context: str) -> Tuple[str, str]:
    """
    Asks the LLM to classify severity (Low/Medium/High) and suggest human actions 
    for unresolved support queries.
    
    Returns:
        (severity, suggested_action)
    """
    provider = config.LLM_PROVIDER
    if provider == "mock" or not LANGCHAIN_AVAILABLE:
        res = _mock_llm_escalation(query, why_failed)
        return res["severity"], res["suggested_action"]

    try:
        model = _get_langchain_model()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the support supervisor of the 'Support Resolution Claw' agent.
An incoming customer query could not be resolved autonomously because the knowledge base coverage was insufficient or the request involves high-risk events.

Your task is to classify this ticket's severity:
- High: mentions of money/payment failure, fraud, account blocked, legal/compliance threats, or hacking.
- Medium: KYC issues, repeated transaction failures, technical errors blocking partner operations.
- Low: general how-to questions, minor app navigation issues that lack direct KB coverage.

Also generate a specific, actionable suggested human action for Eko support agents.

You MUST output your response in strict JSON format with exactly two keys:
{{
  "severity": "Low" | "Medium" | "High",
  "suggested_action": "what a human agent should do step-by-step"
}}
Do not include any other text.
"""),
            ("user", "User Query: {query}\nReason why unresolved: {why_failed}\nRetrieved context (for reference):\n{context}")
        ])

        chain = prompt | model
        response = chain.invoke({"query": query, "why_failed": why_failed, "context": retrieved_context})
        
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            severity = data.get("severity", "Medium")
            suggested_action = data.get("suggested_action", "Investigate account and contact merchant.")
            return severity, suggested_action
        else:
            logger.error("LLM did not return a valid JSON in escalation node.")
            return "Medium", "Manually review the merchant's query and resolve."
            
    except Exception as e:
        logger.error(f"Error invoking LLM in escalation node: {e}. Falling back to Mock.")
        res = _mock_llm_escalation(query, why_failed)
        return res["severity"], res["suggested_action"]


if __name__ == "__main__":
    # Test script for LLM node calls
    print("Testing decision node:")
    ans, conf = call_decision_node("What is the age requirement to join?", "You must be at least 18 years old to join Eko.")
    print(f"Confidence: {conf} | Answer: {ans}")
    
    print("\nTesting escalation node:")
    sev, act = call_escalation_node("My account has been hacked and Rs 50000 stolen", "Low confidence score and mentions theft", "None")
    print(f"Severity: {sev} | Action: {act}")
