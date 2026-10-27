import streamlit as st
import numpy as np
import time, os

# FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv
from key import llm_call

load_dotenv()

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="MindCare AI Chatbot",
    layout="centered"
)

# ================== CSS ==================
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #0B1120;
}

/* Header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 20px;
}

/* Welcome Card */
.welcome-box {
    background: linear-gradient(145deg, #1F2937, #111827);
    color: #FFFFFF;
    padding: 30px 25px;
    border-radius: 16px;
    width: 70%;
    margin: 40px auto;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    text-align: center;
}

.welcome-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 10px;
}

.welcome-sub {
    font-size: 14px;
    color: #9CA3AF;
    line-height: 1.6;
}

/* Chat */
.msg_user {
    background: #2563EB;
    color: #FFFFFF;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 6px 0;
    max-width: 75%;
    margin-left: auto;
}

.msg_bot {
    background: #1F2937;
    color: #FFFFFF;
    padding: 12px 14px;
    border-radius: 12px;
    margin: 6px 0;
    max-width: 80%;
    margin-right: auto;
    border: 1px solid #374151;
}

.msg_bot b {
    color: #9CA3AF;
}

.typing {
    color: #9CA3AF;
}

</style>
""", unsafe_allow_html=True)

# ================== LOAD MODELS ==================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

@st.cache_resource
def load_vector_db():
    if not FAISS_AVAILABLE:
        return None, None
    try:
        index = faiss.read_index("vector.index")
        docs = np.load("stored_docs.npy", allow_pickle=True)
        return index, docs
    except:
        return None, None

embedder = load_embedder()
emo_pipe = load_emotion_model()
faiss_index, stored_docs = load_vector_db()

# ================== FUNCTIONS ==================
def retrieve(q, k=3):
    if faiss_index is None or stored_docs is None:
        return ""
    try:
        q_vec = embedder.encode([q])
        D, I = faiss_index.search(q_vec, k)
        return "\n".join([stored_docs[i] for i in I[0] if i < len(stored_docs)])
    except:
        return ""

def detect_emotion(text):
    try:
        return emo_pipe(text)[0]["label"].upper()
    except:
        return "NEUTRAL"

def is_crisis(q):
    keywords = ["suicide", "kill myself", "end my life", "want to die", "die"]
    return any(word in q.lower() for word in keywords)

def ask_llm(q, emotion):

    if is_crisis(q):
        return (
            "I'm really sorry you're feeling this way. "
            "Please consider reaching out to AASRA: 9820466726. "
            "You are not alone."
        )

    # Build conversation history from session state
    history_lines = []
    for msg in st.session_state.chat[:-1]:  # exclude the current user message just appended
        if msg[0] == "user":
            history_lines.append(f"User: {msg[1]}")
        else:
            history_lines.append(f"You: {msg[1]}")
    history_text = "\n".join(history_lines[-8:])  

    # RAG
    rag_context = retrieve(q)
    rag_section = f"\nRelevant background knowledge:\n{rag_context}\n" if rag_context else ""

    prompt = f"""You are a warm, human-like mental health companion. Your goal is to make the user feel genuinely heard and supported.

Conversation so far:
{history_text}
{rag_section}
Current user emotion detected: {emotion}

Rules:
- NEVER start your reply with "I'm so sorry to hear" or any fixed opener — vary your tone naturally
- Do NOT repeat anything already said in previous turns
- Be concise, warm, and genuinely curious about the user
- Ask only ONE follow-up question per reply
- Do NOT mention any external sources, datasets, or knowledge bases
- Do NOT assume anything the user hasn't explicitly shared
- Respond only to what the user actually said in their latest message

User: {q}
You:"""

    return llm_call(prompt)

# ================== TITLE ==================
st.markdown("<div class='main-title'>MindCare AI Chatbot</div>", unsafe_allow_html=True)

# ================== CHAT STATE ==================
if "chat" not in st.session_state:
    st.session_state.chat = []

# ================== WELCOME ==================
if len(st.session_state.chat) == 0:
    st.markdown("""
    <div class='welcome-box'>
        <div class='welcome-title'>MindCare AI</div>
        <div class='welcome-sub'>
            A supportive assistant designed to help you express your thoughts,
            understand your emotions, and feel heard without judgment.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================== QUICK BUTTONS ==================
col1, col2, col3, col4 = st.columns(4)

def handle_quick(text):
    st.session_state.chat.append(("user", text))
    emotion = detect_emotion(text)
    reply = ask_llm(text, emotion)
    st.session_state.chat.append(("bot", reply, emotion))

if col1.button("I'm stressed"):
    handle_quick("I'm stressed")

if col2.button("I feel depressed"):
    handle_quick("I feel depressed")

if col3.button("I feel lonely"):
    handle_quick("I feel lonely")

if col4.button("I can't focus"):
    handle_quick("I can't focus")

# ================== DISPLAY CHAT ==================
for msg in st.session_state.chat:
    if msg[0] == "user":
        st.markdown(f"<div class='msg_user'>{msg[1]}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='msg_bot'><b>Emotion:</b> {msg[2]}<br><br>{msg[1]}</div>",
            unsafe_allow_html=True
        )

# ================== INPUT ==================
user_text = st.chat_input("Share what you're feeling...")

if user_text:
    st.session_state.chat.append(("user", user_text))
    emotion = detect_emotion(user_text)
    reply = ask_llm(user_text, emotion)
    st.session_state.chat.append(("bot", reply, emotion))
    st.rerun()