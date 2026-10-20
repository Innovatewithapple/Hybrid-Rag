"""Evaluation adapter for the HybridRag project.

Place this file in the HybridRag project root, beside main.py.
It runs retrieval, reranking, and generation directly, bypassing the Redis
semantic-cache lookup in Final_Answer.
"""

import hashlib
import time

import main as pipeline
from src.models import load_models


def _document_id(doc):
    """Return a stable ID based on metadata or document content."""
    metadata = getattr(doc, "metadata", {}) or {}

    for key in ("chunk_id", "id", "source_id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)

    digest = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:20]
    return f"sha256:{digest}"


def run_rag(question: str) -> dict:
    """Run one question through HybridRag and return evaluation fields."""

    # Retrieve directly, avoiding Final_Answer and its Redis cache lookup.
    start = time.perf_counter()
    retrieved_docs = pipeline.ensamble_retrieval.invoke(question)
    retrieval_ms = (time.perf_counter() - start) * 1000

    # Rerank individual documents with the project's existing CrossEncoder.
    start = time.perf_counter()
    cross_encoder = load_models()[2]
    pairs = [(question, doc.page_content) for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs) if pairs else []
    ranked_pairs = sorted(
        zip(retrieved_docs, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    reranked_docs = [doc for doc, _score in ranked_pairs[:10]]
    reranking_ms = (time.perf_counter() - start) * 1000

    # Generate with the existing prompt and LLM function.
    context = "\n\n".join(doc.page_content for doc in reranked_docs)
    start = time.perf_counter()
    prompt_value = pipeline.prompt.invoke({
        "question": question,
        "context": context,
    })
    answer = pipeline.Call_LLM(prompt_value.to_messages())
    generation_ms = (time.perf_counter() - start) * 1000

    return {
        "question": question,
        "answer": answer,
        "contexts": [doc.page_content for doc in reranked_docs],
        "retrieved_ids": [_document_id(doc) for doc in reranked_docs],
        "stage_latency_ms": {
            "retrieval_ms": round(retrieval_ms, 2),
            "reranking_ms": round(reranking_ms, 2),
            "generation_ms": round(generation_ms, 2),
        },
        "metadata": {
            "cache_bypassed": True,
            "retrieved_count": len(reranked_docs),
        },
    }
