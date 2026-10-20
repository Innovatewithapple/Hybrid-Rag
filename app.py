"""HybridRag minimalist single-page UI.
Run: streamlit run app.py

Connect your existing services in build_index_service() and
answer_question_service() below.
"""
from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda
import redis
from clean_pdf import Load_PDF,PDF_Chunking
from src.reranking import Reranking
from src.embeddings import NomicEmbeddings
from src.models import load_models
import streamlit as st
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from main import Final_Answer
from src.document_summary import Generate_Document_Summary,Incoming_quesiton_Embeddings,Overview_quesiton_Embeddings,overview_questions

#-------Load Models and Redis once per Streamlit process------#
@st.cache_resource
def get_shared_resources():
    client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    tokenizer, encoder, reranker = load_models()
    embedding_adapter = NomicEmbeddings(encoder_model=encoder)
    return client, tokenizer, encoder, reranker, embedding_adapter

redis_client, encoder_tokenizer, encoder_model, reranker_model, embeddings = get_shared_resources()


st.set_page_config(page_title="HYBRIDRAG", page_icon="◼", layout="wide")

st.markdown("""
<style>
:root {
    color-scheme: dark;
}

/* Base */
html, body, [data-testid="stAppViewContainer"] {
    background: #080808 !important;
    color: #f3f3f3 !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 50% -20%, #1a1a1a 0%, #080808 48%)
        !important;
}

/* Streamlit header: prevent overlap with page content */
[data-testid="stHeader"] {
    height: 0 !important;
    background: transparent !important;
}

/* Main layout */
.block-container {
    max-width: 1120px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Typography */
h1, h2, h3, p, label, span, div {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                 "Segoe UI", sans-serif;
}

h1, h2, h3 {
    color: #ffffff !important;
    letter-spacing: -0.04em;
}

/* Top navigation */
.topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    border-bottom: 1px solid #292929;
    padding: 0 0 18px;
    margin-bottom: 44px;
    overflow: visible;
}

.brand {
    font-size: 13px;
    font-weight: 750;
    letter-spacing: 0.22em;
    color: #ffffff;
}

.meta, .kicker, .section {
    color: #929292;
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.meta {
    text-align: right;
}

.kicker {
    margin-bottom: 13px;
}

/* Hero */
.hero {
    font-size: clamp(38px, 5.5vw, 62px);
    line-height: 1.08;
    letter-spacing: -0.065em;
    font-weight: 560;
    color: #ffffff;
    margin: 0 0 14px;
    overflow: visible;
}

.copy {
    color: #a0a0a0;
    font-size: 14px;
    line-height: 1.7;
    max-width: 610px;
}

.section {
    margin: 4px 0 13px;
    font-weight: 650;
}

/* Panels */
.surface {
    border: 1px solid #292929;
    background: rgba(17, 17, 17, 0.86);
    border-radius: 12px;
    padding: 20px;
}

/* Buttons */
.stButton > button {
    min-height: 42px;
    border-radius: 7px !important;
    font-size: 12px !important;
    font-weight: 650 !important;
    transition: opacity 0.15s ease, border-color 0.15s ease;
}

.stButton > button[kind="primary"] {
    background: #f2f2f2 !important;
    color: #080808 !important;
    border: 1px solid #f2f2f2 !important;
}

.stButton > button[kind="secondary"] {
    background: #111111 !important;
    color: #eeeeee !important;
    border: 1px solid #333333 !important;
}

.stButton > button:hover {
    opacity: 0.85;
    border-color: #777777 !important;
}

.stButton > button:disabled {
    background: #202020 !important;
    color: #666666 !important;
    border-color: #292929 !important;
}

/* PDF uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #3a3a3a;
    border-radius: 9px;
    background: #0b0b0b;
    padding: 8px;
}

[data-testid="stFileUploader"] button {
    background: #e8e8e8 !important;
    color: #080808 !important;
    border: 0 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #b5b5b5 !important;
}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stChatInput"] textarea {
    background: #101010 !important;
    color: #f4f4f4 !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stChatInput"] textarea:focus {
    border-color: #777777 !important;
    box-shadow: none !important;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: #101010 !important;
    border: 1px solid #242424;
    border-radius: 10px;
    margin-bottom: 10px;
}

/* Progress */
[data-testid="stProgressBar"] > div > div {
    background: #e8e8e8 !important;
}

[data-testid="stProgressBar"] > div {
    background: #282828 !important;
}

/* Alerts and separators */
[data-testid="stAlert"] {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    color: #eeeeee !important;
}

hr {
    border-color: #272727 !important;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Small helper text */
.small {
    color: #777777;
    font-size: 10px;
    line-height: 1.6;
}

/* Responsive layout */
@media (max-width: 700px) {
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .topline {
        margin-bottom: 30px;
    }

    .meta {
        text-align: left;
    }

    .hero {
        font-size: clamp(36px, 9vw, 48px);
        line-height: 1.1;
    }

    .surface {
        padding: 16px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------- SERVICE ADAPTERS ----------
def build_index_service(pdf_path: str, progress_callback: Callable[[float, str], None]) -> dict:
    progress_callback(0.10, "Loading PDF...")
    doc = Load_PDF(pdf_path)
    semantic_chunks = PDF_Chunking(docs=doc, encoder_model=encoder_model)

    progress_callback(0.20, "Creating Vector Embeddings...")
    vector_store = FAISS.from_documents(semantic_chunks, embedding=embeddings)
    vector_retrieval = vector_store.as_retriever(search_kwargs={"k": 10})

    overview_embeddings = np.asarray(Overview_quesiton_Embeddings(embeddings=embeddings))
    print("Overview embeddings shape:", overview_embeddings.shape)

    progress_callback(0.40, "Best Matching...")
    bm25_retrieval = BM25Retriever.from_documents(documents=doc)
    bm25_retrieval.k = 10

    progress_callback(0.60, "Ensembling...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retrieval, bm25_retrieval], weights=[0.5, 0.5]
    )

    progress_callback(0.80, "Generating Summary (This might take some time! Grab your ☕️)...")
    reranker = RunnableLambda(Reranking(model=reranker_model, top_k=5))
    document_summary = Generate_Document_Summary(chunks=semantic_chunks)

    progress_callback(1.0, "Index Ready...")
    # Return all document-specific objects. The main Streamlit thread will
    # place them in session_state when it receives the worker's done event.
    return {
        "chunks": len(semantic_chunks),
        "summary": document_summary,
        "overview": overview_embeddings,
        "vector_store": vector_store,
        "ensemble_retrieval": ensemble_retriever,
        "active_chunks": semantic_chunks,
        "reranker": reranker,
    }


def answer_question_service(question: str) -> str:
    state = st.session_state
    required = ("overview", "summary", "ensemble_retrieval", "reranker")
    missing = [key for key in required if state.get(key) is None]
    if missing:
        raise RuntimeError(f"Index state is missing: {', '.join(missing)}. Please rebuild the index.")

    return Final_Answer(
        Quesiton=question,
        embeddings=embeddings,
        overveiw=state.overview,
        document_summary=state.summary,
        ensamble_retrieval=state.ensemble_retrieval,
        reranker=state.reranker,
        redis_client=redis_client,
    )


# ---------- Session state and background indexing ----------
def init_state():
    defaults = {
        "active_pdf_name": None, "index_ready": False, "is_processing": False,
        "job_id": None, "job_progress": 0.0, "job_message": "Waiting for a document",
        "job_error": None, "job_result": None, "chat_history": [],
        "overview": None, "summary": None, "vector_store": None,
        "ensemble_retrieval": None, "active_chunks": None, "reranker": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def worker(pdf_bytes: bytes, suffix: str, job_id: str, events: queue.Queue):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(pdf_bytes)
            temp_path = f.name

        def report(progress: float, message: str):
            events.put((job_id, "progress", max(0.0, min(1.0, float(progress))), str(message)))

        result = build_index_service(temp_path, report)
        events.put((job_id, "done", result or {}))
    except Exception as exc:
        events.put((job_id, "error", str(exc)))
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def start_indexing():
    uploaded = st.session_state.get("pdf_uploader")
    if uploaded is None:
        return
    data = uploaded.getvalue()
    if not data:
        st.session_state.job_error = "The selected file is empty."
        return
    job_id = str(uuid.uuid4())
    st.session_state.active_pdf_name = uploaded.name
    st.session_state.index_ready = False
    st.session_state.is_processing = True
    st.session_state.job_id = job_id
    st.session_state.job_progress = 0.0
    st.session_state.job_message = "Preparing document…"
    st.session_state.job_error = None
    st.session_state.job_result = None
    st.session_state.chat_history = []
    st.session_state.events = queue.Queue()
    threading.Thread(
        target=worker,
        args=(data, Path(uploaded.name).suffix or ".pdf", job_id, st.session_state.events),
        daemon=True,
    ).start()


def drain_events():
    events = st.session_state.get("events")
    if events is None:
        return
    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            break
        if event[0] != st.session_state.job_id:
            continue
        if event[1] == "progress":
            _, _, progress, message = event
            st.session_state.job_progress = progress
            st.session_state.job_message = message
        elif event[1] == "done":
            result = event[2]
            st.session_state.job_result = result
            # Persist every document-specific artifact across Streamlit reruns.
            st.session_state.overview = result["overview"]
            st.session_state.summary = result["summary"]
            st.session_state.vector_store = result["vector_store"]
            st.session_state.ensemble_retrieval = result["ensemble_retrieval"]
            st.session_state.active_chunks = result["active_chunks"]
            st.session_state.reranker = result["reranker"]
            st.session_state.job_progress = 1.0
            st.session_state.job_message = "Index ready"
            st.session_state.is_processing = False
            st.session_state.index_ready = True
        elif event[1] == "error":
            st.session_state.job_error = event[2]
            st.session_state.job_message = "Indexing failed"
            st.session_state.is_processing = False
            st.session_state.index_ready = False


init_state()
drain_events()

st.markdown("""
<div class="topline"><div class="brand">◼ &nbsp; HYBRIDRAG</div>
<div class="meta">DOCUMENT INTELLIGENCE &nbsp; / &nbsp; WORKSPACE</div></div>
<div class="kicker">A focused workspace for document intelligence</div>
<div class="hero">Your documents.<br>Clear answers.</div>
<div class="copy">Upload a PDF, build its index, then ask questions grounded in its content.
One document at a time. No clutter.</div>
""", unsafe_allow_html=True)
st.write("")

left, right = st.columns([0.82, 1.18], gap="large")

with left:
    st.markdown('<div class="section">01 / SOURCE DOCUMENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="surface"><b style="color:#f5f5f5">Add a PDF</b><p style="font-size:12px;color:#929292">Choose the document to index. Questions stay disabled until indexing completes.</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader",
                                label_visibility="collapsed", disabled=st.session_state.is_processing)
    if uploaded is not None:
        st.markdown(
            f'<div style="border:1px solid #303030;background:#0b0b0b;border-radius:8px;padding:12px;margin:12px 0;color:#eee;font-size:12px">▤ &nbsp; {uploaded.name}<span style="color:#888"> &nbsp; · &nbsp; {len(uploaded.getvalue())/1024:.1f} KB</span></div>',
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns([1.2, .85], gap="small")
        with b1:
            st.button("Build index  ↗", type="primary", use_container_width=True,
                      disabled=st.session_state.is_processing, on_click=start_indexing)
        with b2:
            if st.button("Remove file", use_container_width=True,
                         disabled=st.session_state.is_processing):
                st.session_state.active_pdf_name = None
                st.session_state.index_ready = False
                st.session_state.chat_history = []
                st.session_state.job_error = None
                st.session_state.job_progress = 0.0
                st.session_state.job_message = "Waiting for a document"
                st.session_state.job_result = None
                st.session_state.overview = None
                st.session_state.summary = None
                st.session_state.vector_store = None
                st.session_state.ensemble_retrieval = None
                st.session_state.active_chunks = None
                st.session_state.reranker = None
                st.session_state.pdf_uploader = None
                st.rerun()
    else:
        st.markdown('<div class="small">PDF only · Connect your existing services below.</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section">02 / INDEX STATUS</div>', unsafe_allow_html=True)
    st.markdown('<div class="surface">', unsafe_allow_html=True)
    if st.session_state.is_processing:
        st.markdown(f'<div style="font-size:11px;color:#ccc">● &nbsp; {st.session_state.job_message}</div>', unsafe_allow_html=True)
        st.progress(st.session_state.job_progress)
        st.caption(f"{round(st.session_state.job_progress * 100)}% complete")
        st.caption("Question answering is paused while the index is being built.")
    elif st.session_state.job_error:
        st.error(f"Indexing failed: {st.session_state.job_error}")
        st.caption("Connect build_index_service() to your existing indexing code.")
    elif st.session_state.index_ready:
        st.markdown('<div style="font-size:11px;color:#ddd">● &nbsp; INDEX READY</div>', unsafe_allow_html=True)
        result = st.session_state.job_result or {}
        if result.get("chunks") is not None:
            st.caption(f"{result['chunks']} chunks indexed")
        else:
            st.caption("Your document is ready for questions.")
    else:
        st.markdown('<div style="font-size:11px;color:#999">○ &nbsp; WAITING FOR A DOCUMENT</div>', unsafe_allow_html=True)
        st.caption("Upload a PDF and build the index to begin.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 03 / DOCUMENT OVERVIEW
    if st.session_state.index_ready:
        summary = (st.session_state.job_result or {}).get("summary", "")

        if summary:
            st.write("")
            st.markdown(
                '<div class="section">03 / DOCUMENT OVERVIEW</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="surface">', unsafe_allow_html=True)
            st.markdown(summary)
            st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section">03 / ASK YOUR DOCUMENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="surface"><b style="color:#f5f5f5">Question and answer</b><p style="font-size:12px;color:#929292">Ask about the indexed PDF. Responses come from your existing RAG service.</p></div>', unsafe_allow_html=True)
    disabled = not st.session_state.index_ready or st.session_state.is_processing
    if disabled:
        st.markdown('<div class="small" style="margin:14px 0">Questions unlock when indexing is complete.</div>', unsafe_allow_html=True)
    chat = st.container(height=360, border=False)
    with chat:
        if not st.session_state.chat_history:
            st.markdown('<div style="padding:38px 8px;text-align:center;color:#777"><div style="font-size:24px;margin-bottom:10px">⌁</div><div style="font-size:13px;color:#bbb">Start with a question.</div><div style="font-size:11px;margin-top:7px">Your conversation will appear here.</div></div>', unsafe_allow_html=True)
        else:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    question = st.chat_input("Ask something about your document…", disabled=disabled)
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        try:
            with st.spinner("Thinking…"):
                answer = answer_question_service(question)
            st.session_state.chat_history.append({"role": "assistant", "content": str(answer)})
        except NotImplementedError as exc:
            st.session_state.chat_history.append({"role": "assistant", "content": f"**Connect your service first.**\n\n{exc}"})
        except Exception as exc:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Something went wrong: `{exc}`"})
        st.rerun()

if st.session_state.is_processing:
    time.sleep(0.6)
    st.rerun()

st.markdown("---")
st.markdown('<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#666;font-size:10px;letter-spacing:.1em"><span>HYBRIDRAG / PRIVATE WORKSPACE</span><span>RETRIEVE · RERANK · RESPOND</span></div>', unsafe_allow_html=True)
