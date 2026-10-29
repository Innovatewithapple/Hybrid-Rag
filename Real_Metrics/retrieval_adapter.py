"""Standalone HybridRag adapter for retrieval evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Project layout:
# HybridRag/
# ├── src/
# ├── data/
# └── Real_Metrics/
#     └── retrieval_adapter.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make the HybridRag root importable.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from src.embeddings import NomicEmbeddings
from src.models import load_models

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "semantic_chunks_with_ids.json"
)

class HybridRagRetrievalAdapter:

    def __init__(
        self,
        chunks_path: str | Path = CHUNKS_PATH,
        retrieval_k: int = 10,
    ):
        if retrieval_k < 1:
            raise ValueError("retrieval_k must be >= 1")

        chunks_path = Path(chunks_path)

        if not chunks_path.is_absolute():
            chunks_path = PROJECT_ROOT / chunks_path

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {chunks_path}\n"
                "Create data/semantic_chunks_with_ids.json first."
            )

        with chunks_path.open("r", encoding="utf-8") as file:
            items = json.load(file)

        self.documents = []
        seen_ids = set()

        for index, item in enumerate(items):
            metadata = dict(item.get("metadata") or {})
            chunk_id = item.get("chunk_id") or metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError(
                    f"Chunk at index {index} has no "
                    "metadata['chunk_id']. Use the ID-enriched JSON."
                )

            chunk_id = str(chunk_id)

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk_id found: {chunk_id}"
                )

            seen_ids.add(chunk_id)
            metadata["chunk_id"] = chunk_id

            self.documents.append(
                Document(
                    page_content=item["page_content"],
                    metadata=metadata,
                )
            )

        if not self.documents:
            raise ValueError(
                f"No chunks found in {chunks_path}"
            )

        # Load the project's existing embedding and reranking models.
        _tokenizer, encoder_model, cross_encoder = load_models()

        embeddings = NomicEmbeddings(
            encoder_model=encoder_model
        )

        # BM25 index for evaluation only.
        self.bm25 = BM25Retriever.from_documents(
            self.documents
        )
        self.bm25.k = retrieval_k

        # FAISS index for evaluation only.
        vector_store = FAISS.from_documents(
            self.documents,
            embedding=embeddings,
        )

        self.faiss = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_k},
        )

        # Hybrid retrieval.
        self.ensemble = EnsembleRetriever(
            retrievers=[
                self.bm25,
                self.faiss,
            ],
            weights=[0.5, 0.5],
        )

        self.cross_encoder = cross_encoder
        self.retrieval_k = retrieval_k
        self.valid_chunk_ids = seen_ids

    @staticmethod
    def _chunk_id(doc: Document) -> str:
        chunk_id = (doc.metadata or {}).get("chunk_id")

        if not chunk_id:
            raise ValueError(
                "A retrieved document lost its chunk_id metadata."
            )

        return str(chunk_id)

    def retrieve_ids(self, question: str) -> list[str]:
        """Return chunk IDs ordered by hybrid retrieval and reranking."""

        docs = self.ensemble.invoke(question)

        if not docs:
            return []

        pairs = [
            (question, doc.page_content)
            for doc in docs
        ]

        scores = self.cross_encoder.predict(pairs)

        ranked_docs = [
            doc
            for doc, _score in sorted(
                zip(docs, scores),
                key=lambda pair: float(pair[1]),
                reverse=True,
            )
        ][:self.retrieval_k]

        return [
            self._chunk_id(doc)
            for doc in ranked_docs
        ]