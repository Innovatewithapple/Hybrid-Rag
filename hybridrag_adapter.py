"""Adapter for the user's HybridRag project.

Copy this file to the HybridRag project root as hybridrag_eval_adapter.py.
It imports main.py's already-configured pipeline objects, but deliberately
bypasses Final_Answer/Redis semantic-cache lookup during evaluation.
"""
from __future__ import annotations
import hashlib
import time

import main as pipeline
from src.models import Call_LLM, load_models
from src.reranking import Reranking

# Reuse the cached CrossEncoder model from load_models(); do not load a second copy.
_cross_encoder = load_models()[2]
_doc_reranker = Reranking(model=_cross_encoder, top_k=10)


def _doc_id(doc) -> str:
    """Stable ID for a chunk, using metadata when available, otherwise content hash."""
    for key in ("chunk_id", "id", "source_id"):
        value = doc.metadata.get(key) if getattr(doc, "metadata", None) else None
        if value is not None and str(value).strip():
            return str(value)
    content = doc.page_content
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]


def run_rag(question: str) -> dict:
    """Run real retrieval, reranking, and generation without Redis cache hits."""
    # Retrieval
    start = time.perf_counter()
    retrieved_docs = pipeline.ensamble_retrieval.invoke(question)
    retrieval_ms = (time.perf_counter() - start) * 1000

    # Reranking returns individual Documents so retrieval metrics can inspect IDs.
    start = time.perf_counter()
    reranked_docs = _doc_reranker.rerank_documents(question, retrieved_docs)
    reranking_ms = (time.perf_counter() - start) * 1000

    # Generate answer using the same project prompt and normal generation client.
    context = "\n\n".join(doc.page_content for doc in reranked_docs)
    start = time.perf_counter()
    prompt_value = pipeline.prompt.invoke({"question": question, "context": context})
    answer = Call_LLM(prompt_value.to_messages())
    generation_ms = (time.perf_counter() - start) * 1000

    return {
        "question": question,
        "answer": answer,
        "contexts": [doc.page_content for doc in reranked_docs],
        "retrieved_ids": [_doc_id(doc) for doc in reranked_docs],
        "stage_latency_ms": {
            "retrieval_ms": round(retrieval_ms, 2),
            "reranking_ms": round(reranking_ms, 2),
            "generation_ms": round(generation_ms, 2),
        },
        "metadata": {"cache_bypassed": True, "retrieved_count": len(reranked_docs)},
    }
