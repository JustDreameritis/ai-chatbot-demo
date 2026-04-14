"""
app.py — Streamlit web app for AI Document Chatbot

Features:
  - Document upload (PDF, TXT, MD, DOCX)
  - Chat interface with streaming responses
  - Side-by-side retrieved context display
  - Session management (multiple conversations)
  - Professional dark-themed UI with custom CSS
"""

import os
import uuid
from typing import List

import streamlit as st

# ── Page config must be the first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Global ──────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #e0e0e0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #58a6ff;
}

/* ── Chat messages ───────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
}

/* ── Input box ───────────────────────────────────── */
[data-testid="stChatInputContainer"] {
    background-color: #161b22;
    border-top: 1px solid #30363d;
}
[data-testid="stChatInputContainer"] textarea {
    background-color: #21262d !important;
    color: #e0e0e0 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
    background-color: #238636;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    transition: background 0.2s;
}
.stButton > button:hover {
    background-color: #2ea043;
}

/* ── Metric cards ────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
}
[data-testid="stMetricValue"] { color: #58a6ff; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.8rem; }

/* ── Expanders ───────────────────────────────────── */
.streamlit-expanderHeader {
    background-color: #161b22 !important;
    color: #58a6ff !important;
    border: 1px solid #30363d !important;
    border-radius: 6px;
}
.streamlit-expanderContent {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-top: none !important;
}

/* ── Source badge ────────────────────────────────── */
.source-badge {
    display: inline-block;
    background: #1f6feb;
    color: #fff;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* ── Relevance bar ───────────────────────────────── */
.rel-bar-wrap { display: flex; align-items: center; gap: 8px; margin: 4px 0 10px; }
.rel-bar-bg { flex: 1; height: 6px; background: #30363d; border-radius: 3px; }
.rel-bar-fill { height: 6px; border-radius: 3px; background: linear-gradient(90deg, #238636, #58a6ff); }
.rel-label { font-size: 0.75rem; color: #8b949e; white-space: nowrap; }

/* ── File uploader ───────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: #161b22;
    border: 1px dashed #30363d;
    border-radius: 8px;
    padding: 8px;
}

/* ── Session tabs ────────────────────────────────── */
.stTabs [data-baseweb="tab"] {
    background-color: #161b22;
    color: #8b949e;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    background-color: #21262d;
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff;
}

/* ── Hero header ─────────────────────────────────── */
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff 0%, #a5f3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
}
.hero-sub {
    color: #8b949e;
    font-size: 0.95rem;
    margin-top: 4px;
    margin-bottom: 20px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Lazy-load heavy dependencies after page config ─────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def _load_resources():
    """Load config, RAG engine, and Claude client (cached across reruns)."""
    from config import get_config
    from rag_engine import RAGEngine
    from claude_client import ClaudeClient

    try:
        cfg = get_config()
    except ValueError as e:
        return None, None, None, str(e)

    os.makedirs(cfg.uploads_dir, exist_ok=True)
    os.makedirs(cfg.chroma_db_path, exist_ok=True)

    rag = RAGEngine(cfg)
    claude = ClaudeClient(cfg)
    return cfg, rag, claude, None


# ── Session state helpers ──────────────────────────────────────────────────────

def _init_state() -> None:
    """Initialise all session-state keys on first run."""
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}      # session_id → ConversationSession
    if "active_session" not in st.session_state:
        st.session_state.active_session = None
    if "upload_log" not in st.session_state:
        st.session_state.upload_log = []    # list of {"name", "chunks"}
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "last_chunks" not in st.session_state:
        st.session_state.last_chunks = []   # retrieved chunks from last query


def _new_session(claude) -> str:
    """Create a new ConversationSession and return its ID."""
    from claude_client import ConversationSession

    sid = str(uuid.uuid4())[:8]
    st.session_state.sessions[sid] = ConversationSession(session_id=sid)
    st.session_state.active_session = sid
    return sid


def _active_session(claude):
    """Return the current ConversationSession, creating one if needed."""
    if (
        st.session_state.active_session is None
        or st.session_state.active_session not in st.session_state.sessions
    ):
        _new_session(claude)
    return st.session_state.sessions[st.session_state.active_session]


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(cfg, rag, claude) -> None:
    """Render the full left-hand sidebar."""
    with st.sidebar:
        st.markdown('<div class="hero-title">🤖 DocChat</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-sub">RAG-powered Q&A · Claude API</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Document upload ────────────────────────────────────────────────────
        st.subheader("📄 Upload Documents")
        uploaded = st.file_uploader(
            "PDF, TXT, MD, or DOCX",
            type=["pdf", "txt", "md", "markdown", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded:
            _process_uploads(uploaded, rag)

        # ── Knowledge-base stats ───────────────────────────────────────────────
        st.divider()
        st.subheader("📚 Knowledge Base")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", rag.document_count())
        with col2:
            st.metric("Chunks", rag.chunk_count())

        docs = rag.list_documents()
        if docs:
            with st.expander("View documents", expanded=False):
                for doc in docs:
                    col_name, col_del = st.columns([4, 1])
                    with col_name:
                        st.markdown(f"📄 `{doc}`")
                    with col_del:
                        if st.button("✕", key=f"del_{doc}", help=f"Remove {doc}"):
                            n = rag.delete_document(doc)
                            st.success(f"Removed {n} chunks from '{doc}'")
                            st.rerun()

            if st.button("🗑 Clear all documents", use_container_width=True):
                rag.clear_all()
                st.session_state.upload_log.clear()
                st.success("Knowledge base cleared.")
                st.rerun()

        # ── Session management ─────────────────────────────────────────────────
        st.divider()
        st.subheader("💬 Sessions")

        if st.button("➕ New conversation", use_container_width=True):
            _new_session(claude)
            st.rerun()

        sessions = list(st.session_state.sessions.keys())
        if len(sessions) > 1:
            current = st.session_state.active_session
            options = sessions
            idx = options.index(current) if current in options else 0
            selected = st.selectbox(
                "Switch session",
                options,
                index=idx,
                format_func=lambda s: f"Session {s}",
            )
            if selected != st.session_state.active_session:
                st.session_state.active_session = selected
                st.rerun()

            if st.button("🗑 Delete this session", use_container_width=True):
                del st.session_state.sessions[current]
                st.session_state.active_session = None
                st.rerun()

        # ── RAG settings ───────────────────────────────────────────────────────
        st.divider()
        st.subheader("⚙️ Settings")
        with st.expander("RAG parameters", expanded=False):
            new_k = st.slider(
                "Top-K results",
                min_value=1,
                max_value=10,
                value=cfg.top_k_results,
                help="Number of document chunks to retrieve per query.",
            )
            cfg.top_k_results = new_k

            new_temp = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=cfg.temperature,
                step=0.05,
                help="Lower = more factual, higher = more creative.",
            )
            cfg.temperature = new_temp

        # ── Cost tracker ──────────────────────────────────────────────────────
        st.divider()
        sess = st.session_state.sessions.get(st.session_state.active_session)
        if sess:
            total_cost = sum(
                s.total_usage.cost_usd
                for s in st.session_state.sessions.values()
            )
            st.metric("Session cost (est.)", f"${total_cost:.4f}")

        # ── Footer ─────────────────────────────────────────────────────────────
        st.divider()
        st.markdown(
            "<small style='color:#8b949e'>Built by "
            "<a href='https://www.upwork.com/freelancers/~JustDreameritis' "
            "style='color:#58a6ff'>JustDreameritis</a> · "
            "<a href='https://github.com/JustDreameritis/ai-chatbot-demo' "
            "style='color:#58a6ff'>GitHub</a></small>",
            unsafe_allow_html=True,
        )


def _process_uploads(uploaded_files, rag) -> None:
    """Ingest newly uploaded files into the RAG engine."""
    from document_loader import load_from_bytes

    already_loaded = {entry["name"] for entry in st.session_state.upload_log}
    new_files = [f for f in uploaded_files if f.name not in already_loaded]

    if not new_files:
        return

    progress = st.progress(0, text="Ingesting documents…")
    for i, uf in enumerate(new_files):
        progress.progress((i + 1) / len(new_files), text=f"Processing {uf.name}…")
        try:
            pages = load_from_bytes(uf.read(), uf.name)
            n_chunks = rag.ingest(pages)
            st.session_state.upload_log.append({"name": uf.name, "chunks": n_chunks})
            st.success(f"✅ {uf.name} → {n_chunks} chunks added")
        except Exception as e:
            st.error(f"❌ {uf.name}: {e}")

    progress.empty()


# ── Main chat area ─────────────────────────────────────────────────────────────

def render_chat(cfg, rag, claude) -> None:
    """Render the main chat interface with message history."""
    session = _active_session(claude)

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([6, 1])
    with col_title:
        st.markdown(
            f"<h2 style='color:#58a6ff;margin-bottom:0'>Session <code>{session.session_id}</code></h2>",
            unsafe_allow_html=True,
        )
    with col_badge:
        st.markdown(
            f"<span style='background:#238636;color:#fff;padding:3px 10px;"
            f"border-radius:12px;font-size:0.78rem;font-weight:700'>"
            f"{cfg.claude_model.split('-')[1].title()}</span>",
            unsafe_allow_html=True,
        )

    if rag.document_count() == 0:
        st.info(
            "📭 No documents loaded yet. Upload PDFs, TXT, MD, or DOCX files "
            "in the sidebar to enable document Q&A.",
            icon="ℹ️",
        )

    # ── Message history ───────────────────────────────────────────────────────
    for msg in session.history:
        with st.chat_message(msg.role, avatar="🧑" if msg.role == "user" else "🤖"):
            st.markdown(msg.content)

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your documents…"):
        _handle_user_message(prompt, session, cfg, rag, claude)


def _handle_user_message(
    prompt: str, session, cfg, rag, claude
) -> None:
    """Process a user message: retrieve context, stream response, show sources."""

    # Show user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Retrieve relevant context
    chunks = rag.query(prompt, top_k=cfg.top_k_results)
    st.session_state.last_chunks = chunks
    context = rag.build_context(chunks)

    # Two-column layout: response left, sources right
    col_chat, col_sources = st.columns([3, 2])

    with col_chat:
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for token in claude.stream(session, prompt, context):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            except Exception as e:
                response_placeholder.error(f"Error calling Claude API: {e}")

    with col_sources:
        _render_sources(chunks)


def _render_sources(chunks: list) -> None:
    """Render retrieved context chunks in the right column."""
    if not chunks:
        st.markdown(
            "<div style='color:#8b949e;padding:16px;font-size:0.85rem'>"
            "No document context retrieved for this query.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"<div style='color:#8b949e;font-size:0.8rem;margin-bottom:8px'>"
        f"🔍 {len(chunks)} source chunk(s) retrieved</div>",
        unsafe_allow_html=True,
    )

    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "?")
        score_pct = int(chunk.score * 100)
        bar_width = max(4, score_pct)

        badge_html = (
            f'<span class="source-badge">Source {i}</span>'
            f'<span style="color:#8b949e;font-size:0.78rem">{source} · p.{page}</span>'
        )
        bar_html = (
            f'<div class="rel-bar-wrap">'
            f'  <div class="rel-bar-bg"><div class="rel-bar-fill" style="width:{bar_width}%"></div></div>'
            f'  <span class="rel-label">{score_pct}% match</span>'
            f'</div>'
        )

        with st.expander(f"Source {i} — {source}", expanded=(i == 1)):
            st.markdown(badge_html + bar_html, unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:0.83rem;color:#cdd9e5;line-height:1.55;"
                f"background:#0d1117;padding:10px 12px;border-radius:6px;"
                f"border:1px solid #30363d'>{chunk.text}</div>",
                unsafe_allow_html=True,
            )


# ── Error screen ───────────────────────────────────────────────────────────────

def render_error(message: str) -> None:
    """Render a full-page error with setup instructions."""
    st.error("⚠️ Configuration Error", icon="🔑")
    st.markdown(f"**{message}**")
    st.markdown(
        """
### Quick Setup

1. Copy the example env file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

3. Get an API key at [console.anthropic.com](https://console.anthropic.com)

4. Restart the app:
   ```bash
   streamlit run app.py
   ```
"""
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Main application entry point."""
    _init_state()

    cfg, rag, claude, error = _load_resources()

    if error:
        render_error(error)
        return

    render_sidebar(cfg, rag, claude)
    render_chat(cfg, rag, claude)


if __name__ == "__main__":
    main()
