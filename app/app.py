"""
AT82.03 Machine Learning — Assignment 6
Streamlit Chatbot: Chapter 11 QA with Contextual Retrieval

Models:
  - Retriever  : all-MiniLM-L6-v2 (sentence-transformers, local)
  - Generator  : llama-3.1-8b-instant (Groq API)
  - Enrichment : llama-3.1-8b-instant (Groq API)

Run:
    streamlit run app.py
"""

import os
import re
import time
import numpy as np
import faiss
import PyPDF2
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# ─── Configuration ────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
GEN_MODEL     = "llama-3.1-8b-instant"
CHAPTER_TITLE = "Chapter 11: Information Retrieval and Retrieval-Augmented Generation"
PDF_PATH      = "11.pdf"   # Place the chapter PDF in the same folder as app.py

# ─── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chapter 11 QA — Contextual Retrieval",
    page_icon="📖",
    layout="wide"
)

st.title("📖 Chapter 11 QA Chatbot")
st.caption("Contextual Retrieval · AT82.03 Assignment 6 · Jurafsky & Martin (2026)")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Free key at https://console.groq.com/keys"
    )
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=2, max_value=6, value=3)
    st.divider()
    st.markdown("**Models**")
    st.markdown(f"- Retriever : `{EMBED_MODEL}` (local)")
    st.markdown(f"- Generator : `{GEN_MODEL}` (Groq)")
    st.divider()
    st.markdown("**Source**")
    st.markdown("Jurafsky & Martin (2026), *Speech and Language Processing*, Ch. 11")

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start.", icon="🔑")
    st.stop()

groq_client = Groq(api_key=api_key)

# ─── Groq LLM call with retry ─────────────────────────────────────────────────

def groq_chat(prompt: str, max_tokens: int = 300, retries: int = 5) -> str:
    """
    Send a prompt to the Groq LLM and return the response.
    Automatically retries on rate-limit (429) errors, parsing the
    wait time from the error message when available.
    """
    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=GEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                match = re.search(r"try again in (\d+)m([\d.]+)s", err)
                wait  = int(match.group(1)) * 60 + float(match.group(2)) + 5 if match else 60 * (attempt + 1)
                st.warning(f"Rate limit reached. Retrying in {wait:.0f}s… (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed after maximum retries due to rate limiting.")

# ─── Document helpers ─────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    pages = []
    with open(path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return "\n".join(pages)


def clean_text(raw: str) -> str:
    text = re.sub(r"-(\n)\s*", "", raw)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(CHAPTER \d+\s*[•]\s*[A-Z\s]+)", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def enrich_chunk(chunk: str, document: str, title: str) -> str:
    """
    Prepend a 1-sentence LLM-generated context prefix to the chunk.
    Inputs are trimmed to conserve Groq tokens.
    """
    prompt = (
        f"Title: {title}\n"
        f"{document[:1500]}\n\n"
        f"Chunk:\n{' '.join(chunk.split()[:200])}\n\n"
        f'Provide 1 sentence of context. '
        f'Format: "This chunk discusses [explanation]."'
    )
    context = groq_chat(prompt, max_tokens=60)
    return f"{context}\n\n{chunk}"

# ─── Embedding & FAISS ────────────────────────────────────────────────────────

def get_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True).astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# ─── Full pipeline (cached — runs once per session) ───────────────────────────

@st.cache_resource(show_spinner="Loading Chapter 11 and building index — runs once per session…")
def load_pipeline(_api_key: str):
    """
    Extract, clean, chunk, enrich, embed, and index the chapter.
    Cached so the expensive enrichment step does not re-run on every interaction.
    The _api_key parameter is prefixed with _ so Streamlit does not hash it.
    """
    # 1. Load and clean the chapter text
    raw_text     = extract_text_from_pdf(PDF_PATH)
    chapter_text = clean_text(raw_text)
    chunks       = chunk_text(chapter_text)

    # 2. Load local embedding model
    embedder = SentenceTransformer(EMBED_MODEL)

    # 3. Contextual enrichment via Groq with progress bar
    enriched = []
    prog     = st.progress(0, text="Enriching chunks with context…")
    for i, chunk in enumerate(chunks):
        enriched.append(enrich_chunk(chunk, chapter_text, CHAPTER_TITLE))
        prog.progress((i + 1) / len(chunks), text=f"Enriching chunk {i+1}/{len(chunks)}…")
    prog.empty()

    # 4. Embed enriched chunks and build FAISS index
    embeddings = embedder.encode(
        enriched, batch_size=64,
        convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)
    index = build_faiss_index(embeddings)

    return enriched, index, embedder, chapter_text


try:
    enriched_chunks, faiss_index, embedder, chapter_text = load_pipeline(api_key)
except FileNotFoundError:
    st.error(
        f"❌ `{PDF_PATH}` not found. "
        "Place the Chapter 11 PDF in the same folder as `app.py` and restart."
    )
    st.stop()

# ─── Retrieval & generation ───────────────────────────────────────────────────

def retrieve(query: str, k: int) -> List[Tuple[str, float]]:
    """Return top-k enriched chunks with their cosine similarity scores."""
    q_vec = get_embedding(query, embedder).reshape(1, -1)
    faiss.normalize_L2(q_vec)
    scores, indices = faiss_index.search(q_vec, k)
    return [
        (enriched_chunks[i], float(scores[0][rank]))
        for rank, i in enumerate(indices[0])
        if i < len(enriched_chunks)
    ]


def generate_answer(query: str, context_items: List[Tuple[str, float]]) -> str:
    """Generate a grounded answer from retrieved enriched chunks."""
    trimmed = [" ".join(chunk.split()[:300]) for chunk, _ in context_items]
    context = "\n\n".join(trimmed)
    prompt  = (
        f"{context}\n\n"
        f"Based on the passages above from {CHAPTER_TITLE}, answer the following "
        f"question accurately and concisely. "
        f"If the answer is not found in the passages, say so.\n\n"
        f"Question: {query}"
    )
    return groq_chat(prompt, max_tokens=300)

# ─── Chat UI ──────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Source chunks used", expanded=False):
                for idx, (chunk, score) in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {idx}** — similarity: `{score:.4f}`")
                    st.caption(chunk[:500] + ("…" if len(chunk) > 500 else ""))
                    if idx < len(msg["sources"]):
                        st.divider()

# Accept new user input
if prompt := st.chat_input("Ask a question about Chapter 11…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving passages and generating answer…"):
            retrieved = retrieve(prompt, top_k)
            answer    = generate_answer(prompt, retrieved)

        st.markdown(answer)

        with st.expander("📄 Source chunks used", expanded=False):
            for idx, (chunk, score) in enumerate(retrieved, 1):
                st.markdown(f"**Chunk {idx}** — similarity: `{score:.4f}`")
                st.caption(chunk[:500] + ("…" if len(chunk) > 500 else ""))
                if idx < len(retrieved):
                    st.divider()

    st.session_state.messages.append({
        "role"   : "assistant",
        "content": answer,
        "sources": retrieved
    })

# ─── Footer CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 350px;
            right: 0;
            background-color: var(--background-color);
            text-align: center;
            font-size: 12px;
            color: gray;
            padding: 6px 0 8px 0;
            z-index: 999;
            border-top: 0.5px solid rgba(128,128,128,0.2);
        }
    </style>
    <div class="footer">
        <b>AT82.03 Machine Learning — Assignment 6</b> &nbsp;|&nbsp;
        Naive RAG vs. Contextual Retrieval · Chapter 11 &nbsp;|&nbsp;
        <b>Student:</b> Michael Roque Lacar &nbsp;|&nbsp;
        <b>ID:</b> st126161 &nbsp;|&nbsp;
        <b>Embedder:</b> all-MiniLM-L6-v2 &nbsp;|&nbsp;
        <b>Generator:</b> llama-3.1-8b-instant (Groq)
    </div>
    """,
    unsafe_allow_html=True
)
