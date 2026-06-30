# Support Resolution Claw - Core Report

## What This Agent Does (end-to-end)
The **Support Resolution Claw** is an autonomous support agent designed for micro-entrepreneurs and merchant partners of a fictional distribution platform (similar to Eko). The agent automates standard customer service inquiries using semantic search over a local knowledge base of standard operating procedures (SOPs) and FAQs. 

If a query falls within the documentation scope, the agent drafts a direct answer. However, if the query is outside the documentation scope, lacks sufficient context, or involves high-risk events (such as financial fraud, account blockages, or compliance issues), the agent autonomously routes the query to an escalation pipeline. It classifies the ticket severity (Low, Medium, High), drafts a recommended action plan for human agents, builds a structured JSON ticket, and logs the entire transaction to a relational SQLite database.

---

## Architecture Diagram

```
                 [ User Query ]
                       │
                       ▼
            ┌─────────────────────┐
            │   Retrieval Layer   │ ◄── [ Knowledge Base ] (FAQ/SOP Markdown)
            │ (FAISS Vector Store)│
            └──────────┬──────────┘
                       │ (Retrieve Top-k chunks)
                       ▼
            ┌─────────────────────┐
            │    Decision Node    │ ◄── [ LLM Confidence Self-Rating (1-5) ]
            │ (LangGraph Node 1)  │
            └──────────┬──────────┘
                       │ (Calculate Combined Confidence Score)
                       ▼
            /─────────────────────\
           <   Confidence >= 3.0?  >
            \─────────────────────/
                 /           \
         (Yes)  /             \  (No)
               ▼               ▼
      ┌────────────────┐ ┌─────────────────────┐
      │  Answer Node   │ │   Escalation Node   │
      │(LangGraph Node)│ │ (LangGraph Node 3)  │
      └────────┬───────┘ └──────────┬──────────┘
               │                    │ (Severity Classification: Low/Med/High)
               │                    │ (Generate Structured Escalation Ticket)
               ▼                    ▼
      ┌────────────────────────────────────────┐
      │          Structured Logging            │
      │     (Persist to SQLite Database)       │
      └──────────────────┬─────────────────────┘
                         │
                         ▼
             [ Output Response / Ticket ]
```

---

## Input & Output Examples

### Example 1: Resolved
*   **Query**: `"What documents do I need to register as an Eko merchant?"`
*   **Status**: `resolved`
*   **Combined Confidence**: `4.76`
*   **Output / Final Response**:
    > To onboard as an Eko merchant or micro-entrepreneur, the following documents must be submitted via the app:
    > 1. PAN Card (Individual or business)
    > 2. Aadhaar Card (For identity and address verification)
    > 3. Active Bank Account (Passbook or cancelled cheque with candidate name visible)
    > 4. Shop Photo (Inside and outside, showing business board)
    > 
    > All files must be clear and under 5MB in JPEG, PNG, or PDF formats. Incomplete submissions will result in temporary rejection and delay the verification process.

---

### Example 2: Escalated - Medium Severity
*   **Query**: `"My uploaded PAN card was rejected but the spelling is correct. Help me verify."`
*   **Status**: `escalated`
*   **Severity**: `Medium`
*   **Combined Confidence**: `2.30`
*   **Output JSON Escalation Note**:
```json
{
  "query": "My uploaded PAN card was rejected but the spelling is correct. Help me verify.",
  "why_it_couldnt_be_resolved": "Low combined confidence score (2.30 < 3.0 threshold). LLM context did not fully support a complete answer.",
  "severity": "Medium",
  "suggested_human_action": "Review uploaded documents manually in admin panel. Check UIDAI or NSDL verification response codes. Advise merchant on name spellings if mismatch found.",
  "retrieved_context": [
    {
      "source": "kyc_pan_verification.md",
      "content_snippet": "# SOP: Troubleshooting Rejected PAN Card\n\nPAN card rejection usually happens due to:\n1. Name Mismatch: The name on the PAN card must exactly match the name entered in the registration form and the Aadhaar..."
    }
  ]
}
```

---

### Example 3: Escalated - High Severity
*   **Query**: `"A customer is complaining that Rs. 10000 was debited but transaction failed. I need to register a chargeback dispute."`
*   **Status**: `escalated`
*   **Severity**: `High`
*   **Combined Confidence**: `2.15`
*   **Output JSON Escalation Note**:
```json
{
  "query": "A customer is complaining that Rs. 10000 was debited but transaction failed. I need to register a chargeback dispute.",
  "why_it_couldnt_be_resolved": "Low combined confidence score (2.15 < 3.0 threshold). LLM context did not fully support a complete answer.",
  "severity": "High",
  "suggested_human_action": "Initiate chargeback dispute protocol: hold disputed funds in trade wallet and request merchant receipt/ledger registry proof within 48 hours.",
  "retrieved_context": [
    {
      "source": "tx_chargeback.md",
      "content_snippet": "# SOP: Handling Chargebacks and Disputes\n\nA chargeback occurs when a cardholder disputes a transaction made at your shop through their issuing bank.\n1. When Eko receives a chargeback dispute..."
    }
  ]
}
```

---

## Tools, APIs, Models, and Databases Used
*   **Orchestration Engine**: `LangChain` and `LangGraph` to define states, design nodes, set conditional edges, and run the state graph machine.
*   **Embeddings Model**: `Sentence-Transformers` (`all-MiniLM-L6-v2` loading locally via `HuggingFaceEmbeddings`), creating dense 384-dimensional vector representations.
*   **Vector Search Database**: `FAISS` (Facebook AI Similarity Search) used locally to perform L2 distance computations.
*   **LLM Providers Supported**: `OpenAI` (`gpt-4o-mini`), `Google Gemini` (`gemini-1.5-flash`), `Anthropic` (`claude-3-5-sonnet`), and a built-in rule-based `Mock LLM` to allow immediate operation without API keys.
*   **Relational Logs Database**: `SQLite3` (relational SQL file database initialized at runtime at `logs/support_claw.db`).
*   **User Interface**: `Streamlit` with reactive tab states, styled metrics widgets, live database tables, and interactive card containers.

---

## Exception Handling & Escalation Logic

### 1. Confidence Evaluation Formula
In the `decision` node, the system combines statistical similarity and LLM semantic evaluation:
$$\text{Combined Confidence} = (\text{LLM Confidence} \times 0.7) + (\text{FAISS Similarity} \times 5.0 \times 0.3)$$
*   **LLM Confidence**: A 1 to 5 integer rating generated by the LLM self-evaluating how completely the context answers the user query.
*   **FAISS Similarity**: Derived from FAISS L2 distance: $\text{Similarity} = 1.0 - \frac{d^2}{2.0}$, scaled by $5.0$ to match the 1-5 range.
*   **Confidence Threshold**: If $\text{Combined Confidence} \ge 3.0$, the query routes to `answer`. If $< 3.0$, it routes to `escalate`.

### 2. Severity Classification Rules
The `escalation` node uses a fallback hybrid classifier (Rule-based keywords + LLM analysis):
*   **High Severity**: Triggered if query contains keywords like `fraud`, `scam`, `police`, `legal`, `blocked`, `freeze`, `chargeback`, `stolen`.
*   **Medium Severity**: Triggered if query contains keywords like `kyc`, `pan`, `aadhaar`, `verification`, `otp`, `pending`.
*   **Low Severity**: Standard operational or navigation questions that lacked document coverage.

---

## What the Current Version Can Do Autonomously
1.  **Retrieve Context**: Scan the FAISS index to find the top-3 closest matching SOP sections.
2.  **Evaluate Completeness**: Auto-grade how well the documents support the answer and route the execution path accordingly.
3.  **Perform Safe Resolution**: Render the drafted answer if confidence is high.
4.  **Auto-Escalate**: Classify severity, generate an action plan, and compile a JSON card if confidence is low.
5.  **Telemetry Reporting**: Track real-time statistics (resolved rate, severity metrics) and display interactive logs in a database viewer.

---

## What the Next Version Would Improve
1.  **Real Ticket System Integration**: Push the escalation JSON cards directly to ticketing platforms like Zendesk or Freshdesk via webhooks.
2.  **Feedback Loop & Threshold Optimization**: Add a reinforcement mechanism where human corrections of unresolved tickets are fed back to fine-tune the confidence threshold.
3.  **Multi-turn Clarification**: Allow the agent to ask clarifying questions in chat rather than immediately escalating when context is partially missing.
4.  **Multi-language Support**: Support native regional queries (e.g., Hindi, Hinglish, Tamil) using multi-lingual sentence embeddings.

---

## Harness Architecture

The v2 pipeline is structured across three distinct engineering layers:

### Layer 1 — Prompt Engineering
Each LangGraph node uses a dedicated, purpose-scoped prompt. The decision node prompt instructs the LLM to rate its own confidence (1–5). The escalation node prompt forces JSON-structured severity classification. The verification node uses a separate, completely independent prompt with no conversation history, checking only: *"Does this answer address the query using only the given context?"* Prompts are never shared across nodes to prevent context bleed.

### Layer 2 — Context Engineering
State is passed explicitly between nodes via the typed `AgentState` TypedDict. Retrieved chunks (source file, content, similarity score) are formatted and injected as structured context strings. The verification node receives the original query, the retrieved context, and the generated answer as three separate inputs — preventing circular self-justification by the same generation call.

### Layer 3 — Harness Engineering
This is the layer that makes the pipeline trustworthy beyond prompt quality:

- **Explicit Tool Contracts (Pydantic)**: Every node validates its inputs and outputs using `agent/schemas.py`. `DecisionInput`, `DecisionOutput`, `EscalationInput`, `EscalationOutput`, `VerificationInput`, `VerificationOutput`, `LoggingInput`, and `LoggingOutput` are enforced before any node returns data to the next.
- **Malformed Output Exception Path**: If the LLM returns schema-violating output, the node retries once with a stricter prompt. On a second failure, it routes to a `malformed_output` exception path that escalates with `Low` severity rather than crashing.
- **External Self-Verification**: After the Answer node generates a response, a structurally separate `call_verification_node()` function in `agent/llm.py` makes a fresh LLM call — no history, no draft answer context — to independently verify the answer is grounded in the retrieved documents. If verification fails, the answer is rejected and the query escalates instead.
- **Loop / Retry Circuit Breaker**: The `attempts` counter in `AgentState` tracks retries. If retrieval returns zero chunks or self-verification fails, the pipeline loops back to retrieval for one retry. On the second attempt, the circuit breaker hard-caps the loop and forces escalation, logging the reason and attempt count to SQLite.
- **Persistent Ticket State Machine**: Every escalation creates a UUID-keyed ticket in the `tickets` SQLite table with state: `open → assigned → resolved → closed`. State transitions are append-only (state_history is a JSON list of timestamped events, never overwritten). Human agents transition states interactively in the Streamlit dashboard.
- **SQLite Failure Fallback Queue**: If any write to SQLite fails (e.g. disk full, lock), the interaction or ticket is queued to an in-memory list (`fallback_queue`). On the next operation, `flush_queued_logs()` attempts to drain the queue back to disk. The agent never crashes due to a logging failure.

> This implementation follows harness-engineering principles (explicit tool contracts, external verification separate from generation, persistent state with lifecycle management, and loop/retry circuit breakers) rather than relying on a specific agent runtime. The same architecture could be deployed on top of OpenClaw, NemoClaw, NanoClaw, or Hermes Agent by wrapping each node as a tool/skill exposed through that runtime's tool-calling interface; the underlying contracts, state machine, and verification logic would not need to change.

### Known Limitations

- **No real-time multi-session handoff**: Each query is a single stateless session. The agent has no persistent memory across unrelated queries from the same user.
- **No production deployment hardening**: The system currently lacks rate limiting, authentication, or request throttling. It is designed as a demonstration prototype.

