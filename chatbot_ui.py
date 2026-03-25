import ollama
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AgentDev Chat", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #1a1a1a; }
    #MainMenu, footer, header { visibility: hidden; }

    .user-bubble {
        background-color: #2f2f2f;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 4px 0 4px 20%;
        color: #ececec;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .assistant-bubble {
        background-color: transparent;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 4px 20% 4px 0;
        color: #ececec;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .role-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        color: #888;
    }
    .mode-badge {
        display: inline-block;
        font-size: 0.65rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin-left: 8px;
        vertical-align: middle;
    }
    .mode-badge.stream { background-color: #2d4a2d; color: #6fcf6f; }
    .mode-badge.invoke { background-color: #4a3d2d; color: #cfaa6f; }

    [data-testid="stSidebar"] { background-color: #111; border-right: 1px solid #2a2a2a; }
    [data-testid="stSidebar"] * { color: #ccc !important; }
    [data-testid="stSidebar"] textarea {
        background-color: #1e1e1e !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        color: #ececec !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        flex-direction: row !important;
        gap: 1rem;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.85rem !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #2f2f2f !important;
        border: 1px solid #444 !important;
        border-radius: 12px !important;
        color: #ececec !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "last_system_prompt" not in st.session_state:
    st.session_state.last_system_prompt = (
        "you are expert AI engineer, act as if you're teaching me."
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.last_system_prompt,  # ← always show the SAVED prompt
        height=160,
    )

    # Detect if the text_area differs from what's actually applied
    prompt_changed = system_prompt != st.session_state.last_system_prompt

    if prompt_changed:
        st.warning("⚠️ System prompt changed. Apply to start a new conversation.")

    col1, col2 = st.columns(2)

    with col1:
        apply_clicked = st.button(
            "✅ Apply prompt",
            use_container_width=True,
            disabled=not prompt_changed,
        )

    with col2:
        clear_clicked = st.button(
            "🗑️ Clear chat",
            use_container_width=True,
        )

    st.markdown("---")

    try:
        available_models = [m.model for m in ollama.list().models]
    except Exception:
        available_models = []

    if not available_models:
        st.warning("No Ollama models found. Is Ollama running?")

    model_name = st.selectbox(
        "Model",
        options=available_models,
        disabled=not available_models,
    )

    st.markdown("---")

    response_mode = st.radio(
        "Response mode",
        options=["⚡ Stream", "📦 Invoke"],
        index=0,
        help=(
            "**Stream** — tokens appear one-by-one as they're generated.\n\n"
            "**Invoke** — waits for the full response, then displays it all at once."
        ),
    )
    is_streaming = response_mode == "⚡ Stream"

    if is_streaming:
        st.caption("🟢 Tokens will appear in real-time")
    else:
        st.caption("🟡 Full response will appear after generation")

    st.markdown("---")
    st.caption("Running locally via Ollama")

# ── Handle button actions ──────────────────────────────────────────────────────
if apply_clicked:
    st.session_state.history = []
    st.session_state.last_system_prompt = system_prompt
    st.rerun()

if clear_clicked:
    st.session_state.history = []
    st.rerun()

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model(name: str):
    return init_chat_model(
        name, model_provider="ollama", base_url="http://localhost:11434"
    )

model = get_model(model_name)

# ── Chat history display ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if not st.session_state.history:
    st.markdown(
        "<div style='text-align:center; color:#555; margin-top: 15vh; font-size:1.1rem;'>"
        "How can I help you today?"
        "</div>",
        unsafe_allow_html=True,
    )

for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(
            f"<div class='role-label' style='text-align:right;'>You</div>"
            f"<div class='user-bubble'>{msg.content}</div>",
            unsafe_allow_html=True,
        )
    elif isinstance(msg, AIMessage):
        mode = msg.response_metadata.get("mode", "")
        badge = ""
        if mode == "stream":
            badge = "<span class='mode-badge stream'>streamed</span>"
        elif mode == "invoke":
            badge = "<span class='mode-badge invoke'>invoked</span>"

        st.markdown(
            f"<div class='role-label'>Assistant {badge}</div>"
            f"<div class='assistant-bubble'>{msg.content}</div>",
            unsafe_allow_html=True,
        )

# ── Chat input ─────────────────────────────────────────────────────────────────
# Use the SAVED prompt (not the text_area which might be mid-edit)
active_prompt = st.session_state.last_system_prompt

if user_input := st.chat_input("Message AgentDev..."):

    st.session_state.history.append(HumanMessage(content=user_input))
    st.markdown(
        f"<div class='role-label' style='text-align:right;'>You</div>"
        f"<div class='user-bubble'>{user_input}</div>",
        unsafe_allow_html=True,
    )

    messages = [SystemMessage(content=active_prompt)] + st.session_state.history

    # ── STREAM MODE ────────────────────────────────────────────────────────
    if is_streaming:
        badge = "<span class='mode-badge stream'>streamed</span>"
        st.markdown(
            f"<div class='role-label'>Assistant {badge}</div>",
            unsafe_allow_html=True,
        )
        response_placeholder = st.empty()

        full_response = ""
        for chunk in model.stream(messages):
            full_response += chunk.content
            response_placeholder.markdown(
                f"<div class='assistant-bubble'>{full_response}▌</div>",
                unsafe_allow_html=True,
            )

        response_placeholder.markdown(
            f"<div class='assistant-bubble'>{full_response}</div>",
            unsafe_allow_html=True,
        )

        st.session_state.history.append(
            AIMessage(
                content=full_response,
                response_metadata={"mode": "stream"},
            )
        )

    # ── INVOKE MODE ────────────────────────────────────────────────────────
    else:
        with st.spinner("Generating full response..."):
            response = model.invoke(messages)

        badge = "<span class='mode-badge invoke'>invoked</span>"
        st.markdown(
            f"<div class='role-label'>Assistant {badge}</div>"
            f"<div class='assistant-bubble'>{response.content}</div>",
            unsafe_allow_html=True,
        )

        st.session_state.history.append(
            AIMessage(
                content=response.content,
                response_metadata={"mode": "invoke"},
            )
        )

    st.rerun()