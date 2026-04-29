import os
import time
import streamlit as st
import numpy as np
import faiss
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai.errors import ServerError

# -------------------- SETUP --------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Gemini RAG Chatbot")

# -------------------- THEME --------------------
theme = st.get_option("theme.base")
is_dark = theme == "dark"

user_bg = "#2E7D32" if is_dark else "#DCF8C6"
bot_bg = "#2A2A2A" if is_dark else "#F1F0F0"
text_color = "white" if is_dark else "black"

st.markdown(f"""
<style>
.user {{
    background: {user_bg};
    color: {text_color};
    padding: 10px;
    border-radius: 12px;
    margin: 6px;
    text-align: right;
    max-width: 80%;
    margin-left: auto;
}}

.bot {{
    background: {bot_bg};
    color: {text_color};
    padding: 10px;
    border-radius: 12px;
    margin: 6px;
    max-width: 80%;
}}
</style>
""", unsafe_allow_html=True)

# -------------------- EMBEDDING MODEL --------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    return embed_model.encode(texts)

# -------------------- FILE READER --------------------
def read_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    if file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        return "\n".join([p.extract_text() or "" for p in pdf.pages])

    return ""

def chunk_text(text, size=500, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("📂 Upload Documents")
    files = st.file_uploader("PDF / TXT", type=["pdf", "txt"], accept_multiple_files=True)
    process = st.button("Build Knowledge Base")

# -------------------- SESSION STATE INIT --------------------
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

if "files_hash" not in st.session_state:
    st.session_state.files_hash = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------- BUILD KB (ONLY ONCE) --------------------
if process and files:

    current_hash = hash(tuple(f.name for f in files))

    if st.session_state.files_hash != current_hash:

        st.session_state.files_hash = current_hash
        st.session_state.kb_ready = False

        all_chunks = []

        with st.spinner("Processing files..."):
            for f in files:
                text = read_file(f)
                all_chunks.extend(chunk_text(text))

        with st.spinner("Creating embeddings..."):
            vectors = embed(all_chunks)
            dim = vectors.shape[1]

            index = faiss.IndexFlatL2(dim)
            index.add(np.array(vectors))

            st.session_state.index = index
            st.session_state.chunks = all_chunks
            st.session_state.kb_ready = True

        # ⚡ Upload and Embedding toast
        st.toast("File uploaded and embedded successfully ⚡", icon="📁")

        st.session_state.chat = []

# -------------------- RETRIEVAL --------------------
def retrieve(query, k=3):
    q_vec = embed([query])
    distances, idx = st.session_state.index.search(q_vec, k)
    return [(st.session_state.chunks[i], float(d)) for i, d in zip(idx[0], distances[0])]

# -------------------- GEMINI --------------------
def ask_llm(query, docs):
    context = ""

    for i, (chunk, score) in enumerate(docs, 1):
        context += f"\n[Chunk {i} | score {score:.4f}]\n{chunk}\n"

    prompt = f"""
You are a helpful RAG assistant.

Use ONLY the context below.

CONTEXT:
{context}

QUESTION:
{query}
"""

    models = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-pro-latest"
    ]

    for model in models:
        for _ in range(3):
            try:
                res = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                return res.text
            except ServerError:
                time.sleep(2)
            except:
                break

    return "⚠️ Model temporarily unavailable."

# -------------------- CHAT INPUT --------------------
user_input = st.chat_input("Ask something...")

if user_input:
    if "index" not in st.session_state:
        st.warning("Please upload and embed documents first.")
    else:
        st.session_state.chat.append(("user", user_input))
        st.session_state.chat.append(("bot", "⏳ Thinking..."))
        st.rerun()

# -------------------- PROCESS THINKING --------------------
if st.session_state.chat and st.session_state.chat[-1][1] == "⏳ Thinking...":

    last_user = None
    for msg in reversed(st.session_state.chat):
        if msg[0] == "user":
            last_user = msg[1]
            break

    if last_user:
        docs = retrieve(last_user, k=3)
        answer = ask_llm(last_user, docs)

        st.session_state.chat[-1] = ("bot", answer)
        st.rerun()

# -------------------- CHAT UI --------------------
for msg in st.session_state.chat:
    if msg[0] == "user":
        st.markdown(f"<div class='user'>🧑 {msg[1]}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>🤖 {msg[1]}</div>", unsafe_allow_html=True)