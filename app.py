import streamlit as st
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file before importing agent code
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)

import config
from agent.graph import run_pipeline, retriever
from agent.logger import get_all_logs, export_summary_report, init_db, write_summary_report_to_file

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database schema
init_db()

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(
    page_title="Support Resolution Claw",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for UI
st.markdown("""
<style>
    /* Premium visual overrides */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #21262d;
        color: #ecf2f8;
        border: 1px solid #30363d;
        border-radius: 6px;
        transition: 0.2s ease-in-out;
    }
    .stButton>button:hover {
        border-color: #58a6ff;
        color: #58a6ff;
        background-color: #30363d;
    }
    /* Ticket Card styling */
    .ticket-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid;
        background-color: #161b22;
        border-top: 1px solid #30363d;
        border-right: 1px solid #30363d;
        border-bottom: 1px solid #30363d;
    }
    .severity-high {
        border-left-color: #da3637;
    }
    .severity-medium {
        border-left-color: #d29922;
    }
    .severity-low {
        border-left-color: #3080f0;
    }
    .metric-container {
        background-color: #161b22;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #30363d;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 12px;
        color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.title("🦅 Support Resolution Claw")
st.subheader("Autonomous Support Agent & LangGraph Escalation Pipeline")

# --- SIDEBAR: LIVE STATS & UTILITIES ---
st.sidebar.title("🦅 System Cockpit")
st.sidebar.markdown("Real-time telemetry and management controls.")

# Fetch report stats
try:
    stats = export_summary_report()
except Exception as e:
    logger.error(f"Error fetching stats: {e}")
    stats = {
        "total_queries": 0,
        "resolved_count": 0,
        "escalated_count": 0,
        "resolution_rate_percent": 0.0,
        "severity_breakdown": {"Low": 0, "Medium": 0, "High": 0},
        "average_confidence": 0.0,
        "average_similarity": 0.0
    }

# Display Stats Section in Sidebar
st.sidebar.subheader("Live Operational Stats")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown(f'<div class="metric-container"><div class="metric-value">{stats["total_queries"]}</div><div class="metric-label">Total Queries</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-container"><div class="metric-value">{stats["resolution_rate_percent"]}%</div><div class="metric-label">Resolve Rate</div></div>', unsafe_allow_html=True)

st.sidebar.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)

col3, col4 = st.sidebar.columns(2)
with col3:
    st.markdown(f'<div class="metric-container"><div class="metric-value">{stats["resolved_count"]}</div><div class="metric-label">Resolved</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-container"><div class="metric-value">{stats["escalated_count"]}</div><div class="metric-label">Escalated</div></div>', unsafe_allow_html=True)

st.sidebar.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)

# Severity Breakdown in Sidebar
st.sidebar.subheader("Escalation Severity Breakdown")
sev_high = stats["severity_breakdown"]["High"]
sev_med = stats["severity_breakdown"]["Medium"]
sev_low = stats["severity_breakdown"]["Low"]
st.sidebar.markdown(f"🔴 **High Risk**: `{sev_high}` ticket(s)")
st.sidebar.markdown(f"🟡 **Medium Risk**: `{sev_med}` ticket(s)")
st.sidebar.markdown(f"🔵 **Low Risk**: `{sev_low}` ticket(s)")

st.sidebar.markdown("<hr style='border-color: #30363d;'/>", unsafe_allow_html=True)

# Management Actions in Sidebar
st.sidebar.subheader("Management Controls")

# Rebuild FAISS Vector Store Button
if st.sidebar.button("⚙️ Rebuild Vector Index"):
    with st.spinner("Parsing documents and rebuilding FAISS Index..."):
        try:
            retriever.build_and_save_index()
            retriever.load_index()
            st.sidebar.success("FAISS Index rebuilt and loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Rebuild failed: {e}")

# Export Logs Button
if st.sidebar.button("💾 Export Summary Report"):
    try:
        report_path = write_summary_report_to_file()
        if report_path:
            st.sidebar.success(f"Report exported to:\n{report_path}")
        else:
            st.sidebar.error("Failed to export report.")
    except Exception as e:
        st.sidebar.error(f"Export failed: {e}")

# Display LLM Settings summary
st.sidebar.markdown("<hr style='border-color: #30363d;'/>", unsafe_allow_html=True)
st.sidebar.subheader("Active Configuration")
st.sidebar.code(
    f"Provider: {config.LLM_PROVIDER.upper()}\n"
    f"Threshold: {config.CONFIDENCE_THRESHOLD}\n"
    f"Weights:\n"
    f" - LLM: {config.LLM_CONFIDENCE_WEIGHT}\n"
    f" - FAISS: {config.SIMILARITY_WEIGHT}",
    language="yaml"
)

# --- MAIN INTERFACE: TABS ---
tab_chat, tab_tickets, tab_logs = st.tabs([
    "💬 Merchant Chat Interface", 
    "🎫 Support Escalation Tickets", 
    "📂 System Audit Logs"
])

# ================= TAB 1: MERCHANT CHAT INTERFACE =================
with tab_chat:
    st.write("Submit a question below. The agent will retrieve guidelines and either answer or escalate automatically.")
    
    # Text input for query
    user_query = st.text_input("Enter your support query:", placeholder="e.g., How do I reset my MPIN?", key="chat_input")
    
    if st.button("Submit Query", key="submit_btn"):
        if not user_query.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Claw AI Agent running LangGraph pipeline..."):
                try:
                    # Run the LangGraph execution flow
                    final_state = run_pipeline(user_query)
                    
                    # Display Final Answer/Response
                    st.markdown("### Agent Response")
                    st.markdown(final_state["final_response"])
                    
                    # Display Agent Telemetry Metrics in expander
                    with st.expander("🔍 Trace Pipeline Telemetry"):
                        st.markdown(f"**Step Routing Decision**: `{final_state['routing_decision'].upper()}`")
                        st.markdown(f"**Max FAISS Similarity**: `{final_state['max_similarity']:.4f}`")
                        st.markdown(f"**LLM Confidence Self-Rating**: `{final_state['llm_confidence']}/5.0`")
                        st.markdown(f"**Combined Confidence Score**: `{final_state['combined_confidence']:.2f}` (Threshold: `{config.CONFIDENCE_THRESHOLD}`)")
                        
                        st.markdown("---")
                        st.markdown("**Retrieved KB Context Chunks (Top-K):**")
                        if final_state["retrieved_chunks"]:
                            for idx, chunk in enumerate(final_state["retrieved_chunks"]):
                                st.markdown(f"**[{idx+1}] File: {chunk['source']} (Similarity: {chunk['similarity']:.4f})**")
                                st.code(chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else ""), language="markdown")
                        else:
                            st.info("No matching chunks retrieved from FAISS.")
                            
                    # Trigger rerun to refresh sidebar telemetry
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process query: {e}")
                    logger.exception("Error running support pipeline")

# ================= TAB 2: SUPPORT ESCALATION TICKETS =================
with tab_tickets:
    st.write("Dashboard for human agents to review and act on escalated tickets.")
    
    try:
        logs = get_all_logs()
        escalated_logs = [log for log in logs if log["status"] == "escalated"]
    except Exception as e:
        st.error(f"Error loading escalated tickets: {e}")
        escalated_logs = []

    if not escalated_logs:
        st.info("🎉 No pending escalated tickets found. All queries are resolved!")
    else:
        for ticket in escalated_logs:
            try:
                ticket_data = json.loads(ticket["response"])
            except Exception:
                ticket_data = {
                    "query": ticket["query"],
                    "why_it_couldnt_be_resolved": "Could not parse JSON ticket details.",
                    "severity": ticket["severity"] or "Medium",
                    "suggested_human_action": "Review manually.",
                    "retrieved_context": []
                }
            
            severity = ticket_data.get("severity", "Medium")
            sev_class = f"severity-{severity.lower()}"
            
            # Draw ticket card using custom HTML for premium card look
            st.markdown(
                f"""
                <div class="ticket-card {sev_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h4 style="margin: 0; color: #58a6ff;">Ticket ID: #{ticket['id']}</h4>
                        <span style="font-weight: bold; padding: 2px 8px; border-radius: 4px; font-size: 12px; 
                                     background-color: {'#4a1516' if severity=='High' else '#3c2c0e' if severity=='Medium' else '#162e4a'};
                                     color: {'#ff7b72' if severity=='High' else '#d29922' if severity=='Medium' else '#58a6ff'};">
                            {severity.upper()} SEVERITY
                        </span>
                    </div>
                    <p><strong>Merchant Query:</strong> "{ticket_data.get('query')}"</p>
                    <p><strong>Escalation Reason:</strong> <em>{ticket_data.get('why_it_couldnt_be_resolved')}</em></p>
                    <p style="color: #aff5b4;"><strong>Suggested Human Action:</strong> {ticket_data.get('suggested_human_action')}</p>
                    <p style="font-size: 12px; color: #8b949e; margin-bottom: 0;">Logged at: {ticket['timestamp']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Show retrieved context references inside an expander inside the card
            with st.expander(f"Review Retrieved Context Reference for Ticket #{ticket['id']}"):
                context_list = ticket_data.get("retrieved_context", [])
                if not context_list:
                    st.write("No matching knowledge base documents were retrieved.")
                else:
                    for c_idx, c in enumerate(context_list):
                        st.markdown(f"**Reference {c_idx+1}: {c.get('source', 'unknown')}**")
                        st.write(c.get("content_snippet", ""))

# ================= TAB 3: SYSTEM AUDIT LOGS =================
with tab_logs:
    st.write("Full read-only relational transaction database logs.")
    
    try:
        all_logs = get_all_logs()
    except Exception as e:
        st.error(f"Error loading database logs: {e}")
        all_logs = []

    if not all_logs:
        st.info("No query transaction logs found in database.")
    else:
        # Display logs in clean pandas DataFrame table
        import pandas as pd
        df = pd.DataFrame(all_logs)
        # Reorder and format columns for legibility
        df = df[["id", "timestamp", "query", "status", "severity", "confidence_score", "max_similarity"]]
        st.dataframe(df, use_container_width=True)
