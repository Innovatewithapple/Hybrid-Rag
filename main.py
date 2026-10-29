"""Hybrid retrieval setup for parent-child chunking.

This module only builds the retrieval components. It does not run a hard-coded
query, generate an answer, invoke the semantic cache, or route to a summary.
That keeps retrieval evaluation deterministic and separate from answer quality.
"""

from pathlib import Path
from typing import Any
import hashlib
from collections import defaultdict
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from clean_pdf import Load_PDF, parent_child_chunk
from src.embeddings import NomicEmbeddings
from src.models import load_models
from src.reranking import ParentChildReranking


PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / (
    "regulation eu 2016679 of the european parliament and-DOC_1.pdf"
)

# Retrieval configuration. Keep these values fixed between evaluation runs.
BM25_K = 15
FAISS_K = 15
RERANK_TOP_K = 10
ENSEMBLE_WEIGHTS = [0.5, 0.5]



def load_parent_child_documents(pdf_path: Path = PDF_PATH) -> list[Document]:
    """Load parent-child chunks and assign stable IDs from their text."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"GDPR PDF not found at: {pdf_path}")

    source_docs = Load_PDF(str(pdf_path))
    raw_chunks = parent_child_chunk(docs=source_docs)

    if not raw_chunks:
        raise ValueError("parent_child_chunk() returned no chunks.")

    def text_hash(text: str) -> str:
        normalized = " ".join(text.split())
        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()[:16]

    child_counts = defaultdict(int)
    child_documents = []

    for item in raw_chunks:
        child_text = item["child_text"]
        parent_text = item["parent_text"]

        if not child_text.strip() or not parent_text.strip():
            raise ValueError("Encountered an empty child or parent chunk.")

        parent_id = f"parent_{text_hash(parent_text)}"
        child_index = child_counts[parent_id]
        child_counts[parent_id] += 1

        child_id = f"{parent_id}_child_{child_index:04d}"

        metadata = dict(item.get("metadata") or {})
        metadata.update({
            "child_id": child_id,
            "child_text":child_text,
            "parent_id": parent_id,
            "parent_text": parent_text,
        })

        article = metadata.get("article", "")
        child_documents.append(
            Document(
                page_content=f"{article}: {child_text}" if article else child_text,
                metadata=metadata,
            )
        )

    child_ids = [doc.metadata["child_id"] for doc in child_documents]

    if len(child_ids) != len(set(child_ids)):
        raise ValueError("Generated child IDs are not unique.")

    print(f"Loaded source pages/documents: {len(source_docs)}")
    print(f"Parent-child child chunks: {len(child_documents)}")
    print(
        f"Unique parents: "
        f"{len({doc.metadata['parent_id'] for doc in child_documents})}"
    )

    return child_documents



def build_retrieval_components():
    """Build hybrid retrieval and parent-child reranking once per run."""
    _encoder_tokenizer, encoder_model, reranker_model = load_models()
    embeddings = NomicEmbeddings(encoder_model=encoder_model)
    chunks = load_parent_child_documents()

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = BM25_K

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": FAISS_K},
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=ENSEMBLE_WEIGHTS,
    )

    parent_child_reranker = ParentChildReranking(
        model=reranker_model,
        top_k=RERANK_TOP_K,
    )

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "faiss_retriever":faiss_retriever,
        "BM25_retriever":bm25_retriever,
        "hybrid_retriever": hybrid_retriever,
        "parent_child_reranker": parent_child_reranker,
        "vector_store": vector_store,
    }


def retrieve_ranked_documents(
    question: str,
    hybrid_retriever: Any,
    parent_child_reranker: ParentChildReranking,
) -> list[Document]:
    """Return the ranked, unique-parent Document list used for evaluation."""
    candidates = hybrid_retriever.invoke(question)
    reranked = parent_child_reranker(
        {"question": question, "context": candidates}
    )
    return reranked["context"]


def build_rag_chain(
    hybrid_retriever: Any,
    parent_child_reranker: ParentChildReranking,
    prompt: Any,
    call_llm: Any,
):
    """Build the answer chain; evaluation can use retrieval without calling this."""
    def format_parent_context(result: dict) -> dict:
        context = "\n\n".join(
            doc.metadata.get("parent_text", doc.page_content)
            for doc in result["context"]
        )
        return {"question": result["question"], "context": context}

    return (
        {
            "context": hybrid_retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(parent_child_reranker)
        | RunnableLambda(format_parent_context)
        | prompt
        | RunnableLambda(lambda prompt_value: call_llm(prompt_value.to_messages()))
    )

# def RunTest(query):
#     components = build_retrieval_components()
#     faiss_retriever = components["faiss_retriever"]
#     reranker = components["parent_child_reranker"]
#     bm25_retreiver = components["BM25_retriever"]
#     hybrid_retriever = components["hybrid_retriever"]


#     print(f"Query: {query}")
#     print("\nFaiss Retriever Begins....")
#     faiss_chunks = faiss_retriever.invoke(query)
#     bm25_chunks = bm25_retreiver.invoke(query)
#     hybrid_chunks = hybrid_retriever.invoke(query)
#     relevant_ids=["parent_0cd58fe49427d0ea_child_0004", "parent_0b9a31bfcc7d8ff9_child_0002", "parent_0b9a31bfcc7d8ff9_child_0003", "parent_b9f46675b9052e90_child_0002"]
#     for index,faiss_chunk in enumerate(faiss_chunks):
#         print(f"\nIndex: {index+1}")
#         print(f"child_id: {faiss_chunk.metadata['child_id']}")
#         print(f"child_text: {faiss_chunk.metadata['child_text']}")
#         # print(f"\nrelevant_ids: {relevant_ids}")
#         print(f"\nparent_id: {faiss_chunk.metadata['parent_id']} | faiss")
#         print(f"parent_text: {faiss_chunk.metadata['parent_text']}")
#         print("="*50)

#     print(f"\n\n BM25 Begins....")
#     for index,bm25_chunk in enumerate(bm25_chunks):
#             print(f"\nIndex: {index+1}")
#             print(f"child_id: {bm25_chunk.metadata['child_id']}")
#             print(f"child_text: {bm25_chunk.metadata['child_text']}")
#             # print(f"\nrelevant_ids: {relevant_ids}")
#             print(f"\nparent_id: {bm25_chunk.metadata['parent_id']} | BM25")
#             print(f"parent_text: {bm25_chunk.metadata['parent_text']}")
#             print("="*50)

#     print(f"\n\n Hybrid Begins....")
#     for index,hybrid_chunk in enumerate(hybrid_chunks):
#             print(f"\nIndex: {index+1}")
#             print(f"child_id: {hybrid_chunk.metadata['child_id']}")
#             print(f"child_text: {hybrid_chunk.metadata['child_text']}")
#             # print(f"\nrelevant_ids: {relevant_ids}")
#             print(f"\nparent_id: {hybrid_chunk.metadata['parent_id']} | Hybrid")
#             print(f"parent_text: {hybrid_chunk.metadata['parent_text']}")
#             print("="*50)

#     print(f"\n\n Reranker Begins....")
#     reranked = reranker({"question":query,"context":hybrid_chunks})
#     rerank_data = reranked["context"]
#     for i, doc in enumerate(rerank_data, 1):
#             print(f"\nRank: {i}")
#             print("child_id:", doc.metadata["child_id"])
#             print("parent_id:", doc.metadata["parent_id"])
#             print("parent_text:", doc.metadata["parent_text"])
#             print("Article: ", doc.metadata['article'])
#             print("=" * 50)
             

# RunTest(query='If a person believes their data protection rights were violated, what legal remedies do they have against the supervisory authority and against the controller or processor')
if __name__ == "__main__":
    # Intentionally no hard-coded test question or answer generation here.
    # Use Real_Metrics/run_parent_child_evaluation.py for Precision/Recall/MRR/nDCG.
    print("main.py defines the parent-child retrieval setup; no query was executed.")
    print("Run the evaluation script in Real_Metrics to measure retrieval metrics.")
