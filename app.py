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
from agent.logger import (
    get_all_logs,
    export_summary_report,
    init_db,
    write_summary_report_to_file,
    get_all_tickets,
    assign_ticket,
    resolve_ticket,
    close_ticket
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database schema
init_db()

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(
    page_title="Support Resolution Claw v2",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for UI
st.markdown("""
<style>
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
    .timeline-log {
        background-color: #0d1117;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px dashed #30363d;
        font-size: 13px;
        color: #8b949e;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.title("🦅 Support Resolution Claw v2")
st.subheader("Autonomous Support Agent, LangGraph Circuit Breaker & Ticket Lifecycle State Machine")

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
    "🎫 Support Tickets State Cockpit", 
    "📂 System Audit Logs"
])

# ================= TAB 1: MERCHANT CHAT INTERFACE =================
with tab_chat:
    st.write("Submit a question below. The agent will retrieve Eko guidelines and either answer or escalate automatically.")
    
    # Text input for query
    user_query = st.text_input("Enter your support query:", placeholder="e.g., How do I unblock my trade wallet?", key="chat_input")
    
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
                        st.markdown(f"**Execution Steps/Attempts**: `{final_state.get('attempts', 1)}` attempt(s)")
                        st.markdown(f"**Last Route Transition**: `{final_state['routing_decision'].upper()}`")
                        st.markdown(f"**Max FAISS Similarity**: `{final_state['max_similarity']:.4f}`")
                        st.markdown(f"**LLM Confidence Self-Rating**: `{final_state['llm_confidence']}/5.0`")
                        st.markdown(f"**Combined Confidence Score**: `{final_state['combined_confidence']:.2f}` (Threshold: `{config.CONFIDENCE_THRESHOLD}`)")
                        if final_state.get("why_failed"):
                            st.markdown(f"**Pipeline Error Details**: *{final_state['why_failed']}*")
                        
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
    st.write("Review, assign, and transition escalated merchant support tickets state dynamically.")
    
    try:
        tickets = get_all_tickets()
    except Exception as e:
        st.error(f"Error loading tickets: {e}")
        tickets = []

    if not tickets:
        st.info("🎉 No pending escalated tickets found. All queries are resolved!")
    else:
        # Filter buttons by state
        state_filter = st.radio("Filter by State:", ["All", "open", "assigned", "resolved", "closed"], horizontal=True)

        for ticket in tickets:
            ticket_id = ticket["ticket_id"]
            current_state = ticket["state"]
            
            if state_filter != "All" and current_state != state_filter:
                continue

            try:
                ticket_data = json.loads(ticket["escalation_note"])
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
            
            # Show ticket card with current state
            st.markdown(
                f"""
                <div class="ticket-card {sev_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h4 style="margin: 0; color: #58a6ff;">Ticket ID: {ticket_id}</h4>
                        <div>
                            <span style="font-weight: bold; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 5px;
                                         background-color: {'#4a1516' if severity=='High' else '#3c2c0e' if severity=='Medium' else '#162e4a'};
                                         color: {'#ff7b72' if severity=='High' else '#d29922' if severity=='Medium' else '#58a6ff'};">
                                {severity.upper()} SEVERITY
                            </span>
                            <span style="font-weight: bold; padding: 2px 8px; border-radius: 4px; font-size: 11px;
                                         background-color: #21262d; border: 1px solid #30363d; color: #aff5b4;">
                                STATE: {current_state.upper()}
                            </span>
                        </div>
                    </div>
                    <p><strong>Merchant Query:</strong> "{ticket_data.get('query')}"</p>
                    <p><strong>Escalation Reason:</strong> <em>{ticket_data.get('why_it_couldnt_be_resolved')}</em></p>
                    <p style="color: #aff5b4; margin-bottom: 5px;"><strong>Suggested Action:</strong> {ticket_data.get('suggested_human_action')}</p>
                    <p style="font-size: 12px; color: #8b949e; margin-bottom: 5px;">Created: {ticket['created_at']} | Last Updated: {ticket['updated_at']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Action controls to transition state
            c1, c2, c3, _ = st.columns([1, 1, 1, 5])
            with c1:
                if current_state == "open" and st.button("Assign Ticket", key=f"assign_{ticket_id}"):
                    assign_ticket(ticket_id)
                    st.success(f"Ticket {ticket_id[:8]}... assigned!")
                    st.rerun()
            with c2:
                if current_state in ("open", "assigned") and st.button("Resolve Ticket", key=f"resolve_{ticket_id}"):
                    resolve_ticket(ticket_id)
                    st.success(f"Ticket {ticket_id[:8]}... resolved!")
                    st.rerun()
            with c3:
                if current_state == "resolved" and st.button("Close Ticket", key=f"close_{ticket_id}"):
                    close_ticket(ticket_id)
                    st.success(f"Ticket {ticket_id[:8]}... closed!")
                    st.rerun()
            
            # Transition Timeline logs
            try:
                history_list = json.loads(ticket["state_history"])
            except Exception:
                history_list = []
            
            with st.expander(f"📜 View Lifecycle History & References ({len(history_list)} events)"):
                st.markdown("**Timeline History Logs:**")
                for entry in history_list:
                    st.markdown(
                        f"""<div class="timeline-log">
                            📅 <strong>{entry.get('timestamp')}</strong><br/>
                            🔄 State: <code>{entry.get('state').upper()}</code><br/>
                            📝 Note: {entry.get('note')}
                        </div>""",
                        unsafe_allow_html=True
                    )
                
                st.markdown("---")
                st.markdown("**Retrieved Context Reference:**")
                context_list = ticket_data.get("retrieved_context", [])
                if not context_list:
                    st.write("No matching documents references.")
                else:
                    for c_idx, c in enumerate(context_list):
                        st.markdown(f"**Reference {c_idx+1}: {c.get('source', 'unknown')}**")
                        st.write(c.get("content_snippet", ""))

# ================= TAB 3: SYSTEM AUDIT LOGS =================
with tab_logs:
    st.write("Read-only relational transaction database log view.")
    
    try:
        all_logs = get_all_logs()
    except Exception as e:
        st.error(f"Error loading database logs: {e}")
        all_logs = []

    if not all_logs:
        st.info("No query transaction logs found in database.")
    else:
        import pandas as pd
        df = pd.DataFrame(all_logs)
        df = df[["id", "timestamp", "query", "status", "severity", "confidence_score", "max_similarity"]]
        st.dataframe(df, use_container_width=True)
