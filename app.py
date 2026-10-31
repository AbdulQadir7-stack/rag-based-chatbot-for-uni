# app.py - Streamlit frontend for AJKU AI Assistant (Light Theme)

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import json
from datetime import datetime

# ─── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AJKU AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (Light Theme: White background, black text) ───────────────────
st.markdown("""
<style>
    /* Main theme */
    .stApp { background-color: #ffffff; }
    
    /* Chat messages */
    .user-msg {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    .bot-msg {
        background: #f0f2f6;
        color: #000000;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        border-left: 3px solid #1a73e8;
        word-wrap: break-word;
    }
    .msg-meta {
        font-size: 11px;
        color: #666;
        margin-top: 4px;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    
    /* Stats cards */
    .stat-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Input styling */
    .stTextInput input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ccc !important;
    }
    
    /* Sidebar */
    .css-1d391kg { background-color: #f8f9fa; }
    
    /* Badge */
    .route-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        background: #e3f2fd;
        color: #0d47a1;
        margin-top: 4px;
    }
    
    /* Text colors for main content */
    .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, .stText, .stNumberInput, .stSelectbox {
        color: #000000 !important;
    }
    
    /* Metric cards */
    .stMetric label, .stMetric .stMetric-value, .stMetric .stMetric-delta {
        color: #000000 !important;
    }
    
    /* Sidebar text */
    .sidebar .sidebar-content, .css-1d391kg p, .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg label {
        color: #000000 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #000000 !important;
        background-color: #f8f9fa !important;
    }
    
    /* Dataframe */
    .stDataFrame, .dataframe {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1a73e8;
        color: white;
    }
    .stButton button:hover {
        background-color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "logged_in": False,
        "token": "",
        "user_id": 0,
        "role": "",
        "username": "",
        "messages": [],
        "login_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Helpers ───────────────────────────────────────────────────────────────────
def api_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def backend_healthy() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_quick_questions() -> list[str]:
    fallback = [
        "Admission fee kitni hai?",
        "Scholarship kaise milti hai?",
        "Exam date sheet kab aati hai?",
    ]
    try:
        r = requests.get(f"{API_BASE}/quick-questions", headers=api_headers(), timeout=5)
        if r.status_code == 200:
            questions = r.json().get("questions", [])
            return questions[:3] or fallback
    except Exception:
        pass
    return fallback


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""

    lines = ["**Sources**"]
    for source in sources[:3]:
        label = source.get("label", "S")
        title = source.get("title", "Retrieved FAQ")
        doc_id = source.get("doc_id", "")
        lines.append(f"- [{label}] {title} `{doc_id}`")
    return "\n".join(lines)


def logout():
    for key in ["logged_in", "token", "user_id", "role", "username", "messages"]:
        st.session_state[key] = "" if isinstance(st.session_state[key], str) else (
            False if isinstance(st.session_state[key], bool) else
            [] if isinstance(st.session_state[key], list) else 0
        )
    st.rerun()


# ─── Login page ────────────────────────────────────────────────────────────────
def show_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="main-header" style="text-align:center">
            <h1>🎓 AJKU AI Assistant</h1>
            <p style="opacity:0.85">University of Azad Jammu & Kashmir</p>
        </div>
        """, unsafe_allow_html=True)

        if not backend_healthy():
            st.error("❌ Backend not running. Start with: `python api.py`")
            st.code("python api.py", language="bash")
            return

        with st.form("login_form", clear_on_submit=False):
            st.subheader("Sign In")
            username = st.text_input("Username", placeholder="admin or student")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Login →", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("Please enter username and password.")
                else:
                    with st.spinner("Logging in..."):
                        try:
                            resp = requests.post(
                                f"{API_BASE}/login",
                                json={"username": username, "password": password},
                                timeout=5,
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                st.session_state.token = data["token"]
                                st.session_state.user_id = data["user_id"]
                                st.session_state.role = data["role"]
                                st.session_state.username = data["username"]
                                st.session_state.logged_in = True
                                st.rerun()
                            else:
                                detail = resp.json().get("detail", "Invalid credentials.")
                                st.error(f"❌ {detail}")
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Cannot connect to backend. Make sure `python api.py` is running.")
                        except Exception as e:
                            st.error(f"Error: {e}")

        st.markdown("""
        <div style="text-align:center; color:#666; font-size:13px; margin-top:16px">
            Default: admin/admin123 &nbsp;|&nbsp; student/student123
        </div>
        """, unsafe_allow_html=True)


# ─── Chat interface ────────────────────────────────────────────────────────────
def chat_interface():
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0">💬 AJKU AI Assistant</h2>
        <p style="margin:4px 0 0; opacity:0.85">Admission • Exams • Scholarships • Programs • Complaints</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; color:#555; padding:40px 0">
                <div style="font-size:48px">🎓</div>
                <h3 style="color:#333">Welcome to AJKU Assistant!</h3>
                <p>Ask me about admissions, fees, exams, scholarships, or programs.</p>
                <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:16px">
                    <span style="background:#e8f0fe;padding:6px 12px;border-radius:16px;font-size:13px;color:#000">📋 Admission process</span>
                    <span style="background:#e8f0fe;padding:6px 12px;border-radius:16px;font-size:13px;color:#000">💰 Fee structure</span>
                    <span style="background:#e8f0fe;padding:6px 12px;border-radius:16px;font-size:13px;color:#000">🏆 Scholarships</span>
                    <span style="background:#e8f0fe;padding:6px 12px;border-radius:16px;font-size:13px;color:#000">📚 Programs</span>
                    <span style="background:#e8f0fe;padding:6px 12px;border-radius:16px;font-size:13px;color:#000">📝 Exams</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="text-align:right">
                        <div class="user-msg">{msg["content"]}</div>
                        <div class="msg-meta">{msg.get("time", "")}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    route = msg.get("route", "")
                    confidence = msg.get("confidence_score")
                    confidence_text = f"Confidence: {confidence}%" if confidence is not None else ""
                    badge = f'<span class="route-badge">🏷️ {route}</span>' if route and route != "blocked" else ""
                    st.markdown(f"""
                    <div>
                        <div class="bot-msg">{msg["content"]}</div>
                        {badge}
                        <div class="msg-meta">{msg.get("time", "")} {confidence_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if msg.get("was_corrected"):
                        st.caption(f"Corrected query: {msg.get('corrected_query', '')}")
                    if msg.get("sources"):
                        st.markdown(format_sources(msg["sources"]))
                    if msg.get("debug_info") and st.session_state.role == "admin":
                        with st.expander("Admin debug", expanded=False):
                            st.json(msg["debug_info"])
                    if msg.get("follow_up_suggestions"):
                        cols = st.columns(len(msg["follow_up_suggestions"]))
                        for idx, question in enumerate(msg["follow_up_suggestions"]):
                            with cols[idx]:
                                if st.button(question, key=f"follow_{msg.get('query_id')}_{idx}", use_container_width=True):
                                    handle_user_message(question)
                                    st.rerun()

                    # Feedback buttons
                    if msg.get("query_id") and not msg.get("feedback_given"):
                        col_a, col_b, col_c = st.columns([1, 1, 8])
                        with col_a:
                            if st.button("👍", key=f"up_{msg['query_id']}"):
                                send_feedback(msg["query_id"], True, msg)
                        with col_b:
                            if st.button("👎", key=f"dn_{msg['query_id']}"):
                                send_feedback(msg["query_id"], False, msg)

    # Chat input
    st.divider()
    
    # Quick suggestion buttons
    suggestions = get_quick_questions()
    
    with st.expander("💡 Quick Questions", expanded=False):
        cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            with cols[i]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    handle_user_message(sug)
                    st.rerun()

    user_input = st.chat_input("Kuch bhi poochhein AJKU ke baare mein...")
    if user_input:
        handle_user_message(user_input)
        st.rerun()


def send_feedback(query_id: str, thumbs_up: bool, msg: dict):
    try:
        requests.post(
            f"{API_BASE}/feedback",
            json={"query_id": query_id, "thumbs_up": thumbs_up},
            headers=api_headers(),
            timeout=5,
        )
        msg["feedback_given"] = True
        st.toast("✅ Feedback submitted!")
    except Exception:
        st.toast("❌ Could not submit feedback.")


def handle_user_message(user_input: str):
    now = datetime.now().strftime("%H:%M")

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "time": now,
    })

    # Build history string (last 4 exchanges)
    history = ""
    recent = st.session_state.messages[-9:-1]
    for m in recent:
        role_label = "User" if m["role"] == "user" else "Assistant"
        history += f"{role_label}: {m['content']}\n"

    # Call API
    try:
        resp = requests.post(
            f"{API_BASE}/chat/stream",
            json={"question": user_input, "history": history},
            headers=api_headers(),
            timeout=60,
            stream=True,
        )
        if resp.status_code == 200:
            answer = ""
            metadata = {}
            stream_box = st.empty()

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                event = json.loads(raw_line)
                if event.get("type") == "meta":
                    metadata.update(event)
                elif event.get("type") == "token":
                    answer += event.get("text", "")
                    stream_box.markdown(f'<div class="bot-msg">{answer}</div>', unsafe_allow_html=True)
                elif event.get("type") == "final":
                    metadata.update(event)
                    answer = event.get("answer", answer)

            # Clear the live streaming placeholder — the message will be
            # rendered properly from session_state on the next rerun.
            stream_box.empty()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "route": metadata.get("route", ""),
                "query_id": metadata.get("query_id", f"{st.session_state.user_id}_{int(time.time())}"),
                "chunks_found": metadata.get("chunks_found", 0),
                "confidence_score": metadata.get("confidence_score", 0),
                "corrected_query": metadata.get("corrected_query", user_input),
                "was_corrected": metadata.get("was_corrected", False),
                "sources": metadata.get("sources", []),
                "follow_up_suggestions": metadata.get("follow_up_suggestions", []),
                "debug_info": metadata.get("debug_info"),
                "time": datetime.now().strftime("%H:%M"),
                "feedback_given": False,
            })
        elif resp.status_code == 401:
            st.error("Session expired. Please login again.")
            logout()
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Server error ({resp.status_code}). Please try again.",
                "time": datetime.now().strftime("%H:%M"),
            })
    except requests.exceptions.Timeout:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⏱️ Request timed out. Server busy hai, please retry.",
            "time": datetime.now().strftime("%H:%M"),
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Connection error: {e}",
            "time": datetime.now().strftime("%H:%M"),
        })


# ─── Admin dashboard ───────────────────────────────────────────────────────────
def admin_dashboard():
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0">📊 Admin Dashboard</h2>
        <p style="margin:4px 0 0; opacity:0.85">Query Analytics & System Stats</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        resp = requests.get(f"{API_BASE}/admin/stats", headers=api_headers(), timeout=10)
        if resp.status_code == 403:
            st.error("Admin access required.")
            return
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"Cannot fetch stats: {e}")
        return

    if st.button("Reload retrieval cache", type="secondary"):
        try:
            reload_resp = requests.post(
                f"{API_BASE}/admin/reload-data",
                headers=api_headers(),
                timeout=10,
            )
            if reload_resp.status_code == 200:
                st.success(reload_resp.json().get("message", "Cache cleared."))
            else:
                st.error(reload_resp.json().get("detail", "Could not clear cache."))
        except Exception as e:
            st.error(f"Could not clear cache: {e}")

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📨 Total Queries", data["total_queries"])
    with col2:
        st.metric("✅ Allowed", data.get("allowed_queries", data["total_queries"] - data["blocked_queries"]))
    with col3:
        st.metric("🚫 Blocked", data["blocked_queries"])
    with col4:
        st.metric("Avg Chunks", data.get("avg_chunks_found", 0))
    with col5:
        st.metric("👍 Feedback Positive", f"{data['feedback_positive_rate']}%")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Query Distribution by Topic")
        if data.get("route_distribution"):
            df = pd.DataFrame(
                list(data["route_distribution"].items()),
                columns=["Topic", "Count"]
            )
            fig = px.pie(
                df, names="Topic", values="Count",
                color_discrete_sequence=px.colors.sequential.Blues_r,
                hole=0.4,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="black",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    with col_right:
        st.subheader("Collection Coverage")
        if data.get("collection_distribution"):
            df_collections = pd.DataFrame(
                list(data["collection_distribution"].items()),
                columns=["Collection", "Count"]
            )
            fig2 = px.bar(df_collections, x="Collection", y="Count", color="Collection")
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="black",
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No collection data yet.")

    col_trend, col_users = st.columns(2)

    with col_trend:
        st.subheader("Daily Query Trend")
        if data.get("daily_queries"):
            df_daily = pd.DataFrame(data["daily_queries"])
            fig3 = px.line(df_daily, x="date", y=["total", "blocked"], markers=True)
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="black",
                xaxis_title="Date",
                yaxis_title="Queries",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No trend data yet.")

    with col_users:
        st.subheader("Most Active Users")
        if data.get("active_users"):
            df_users = pd.DataFrame(data["active_users"])
            fig4 = px.bar(df_users, x="count", y="username", orientation="h", color="count")
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="black",
                yaxis_title="",
                xaxis_title="Queries",
                showlegend=False,
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No user activity yet.")

    col_top, col_rerank = st.columns(2)

    with col_top:
        st.subheader("🔝 Top 10 Questions")
        if data.get("top_questions"):
            st.dataframe(pd.DataFrame(data["top_questions"]), use_container_width=True, height=260)
        else:
            st.info("No questions logged yet.")

    with col_rerank:
        st.subheader("Reranker Usage")
        if data.get("rerank_methods"):
            df_rerank = pd.DataFrame(
                list(data["rerank_methods"].items()),
                columns=["Method", "Count"]
            )
            fig5 = px.pie(df_rerank, names="Method", values="Count", hole=0.45)
            fig5.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="black",
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No reranker data yet.")

    st.info(f"No-context answers: {data.get('no_context_queries', 0)}")

    # Recent logs
    st.divider()
    st.subheader("📜 Recent Query Logs")
    try:
        logs_resp = requests.get(f"{API_BASE}/admin/logs?limit=50", headers=api_headers(), timeout=10)
        if logs_resp.status_code == 200:
            logs = logs_resp.json()
            if logs:
                df_logs = pd.DataFrame(logs)
                df_logs["allowed"] = df_logs["allowed"].map({1: "✅", 0: "🚫"})
                columns = [
                    "timestamp", "username", "query", "route", "collection",
                    "chunks_found", "top_rerank_score", "rerank_method", "allowed"
                ]
                visible_columns = [col for col in columns if col in df_logs.columns]
                st.dataframe(df_logs[visible_columns], use_container_width=True, height=300)
            else:
                st.info("No logs yet.")
    except Exception:
        st.warning("Could not load logs.")

    # User management
    st.divider()
    st.subheader("👥 User Management")
    col_u1, col_u2 = st.columns(2)
    
    with col_u1:
        try:
            users_resp = requests.get(f"{API_BASE}/admin/users", headers=api_headers(), timeout=5)
            if users_resp.status_code == 200:
                users = users_resp.json()
                st.dataframe(pd.DataFrame(users), use_container_width=True)
        except Exception:
            pass

    with col_u2:
        st.write("**Add New User**")
        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["student", "admin"])
            if st.form_submit_button("Create User", type="primary"):
                try:
                    r = requests.post(
                        f"{API_BASE}/register",
                        json={"username": new_username, "password": new_password, "role": new_role},
                        headers=api_headers(),
                        timeout=5,
                    )
                    if r.status_code == 200:
                        st.success(r.json()["message"])
                    else:
                        st.error(r.json().get("detail", "Error creating user."))
                except Exception as e:
                    st.error(f"Error: {e}")


# ─── Sidebar ───────────────────────────────────────────────────────────────────
def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:16px">
            <div style="font-size:40px">🎓</div>
            <h3 style="margin:8px 0 4px; color:#000">AJKU Assistant</h3>
            <p style="color:#666; font-size:13px">University of AJ&K</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # User info
        role_icon = "👑" if st.session_state.role == "admin" else "👤"
        st.markdown(f"**{role_icon} {st.session_state.username}**")
        st.caption(f"Role: {st.session_state.role.title()}")
        
        st.divider()

        # Navigation
        if st.session_state.role == "admin":
            page = st.radio("Navigation", ["💬 Chat", "📊 Dashboard"], label_visibility="collapsed")
        else:
            page = "💬 Chat"

        st.divider()

        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Logout
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            logout()

        st.divider()
        st.caption("Powered by Gemini AI + ChromaDB")

    return page


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        show_login()
        return

    page = show_sidebar()

    if "Chat" in page:
        chat_interface()
    elif "Dashboard" in page:
        admin_dashboard()


main()
