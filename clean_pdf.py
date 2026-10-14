from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from src.embeddings import NomicEmbeddings

def Load_PDF(path):
    loader = PyMuPDFLoader(file_path=path)
    documents = loader.load()

    for doc in documents:
        print(f'{doc.metadata.get("page")} \n{doc.page_content[:500]}')
        break
    return documents

def PDF_Chunking(docs,encoder_model):
    embeddings = NomicEmbeddings(encoder_model=encoder_model)
    text_splitter = SemanticChunker(embeddings)
    print("Total documents:", len(docs))
    semantic_chunks = text_splitter.create_documents(texts=[text.page_content for text in docs],metadatas=[text.metadata for text in docs])
    
    return semantic_chunks


import json
from pathlib import Path

from langchain_core.documents import Document


CHUNKS_PATH = Path("data/semantic_chunks.json")


def Save_Chunks(chunks, path=CHUNKS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"Saved {len(chunks)} chunks to {path}")


def Load_Chunks(path=CHUNKS_PATH):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
        for item in data
    ]

    print(f"Loaded {len(chunks)} cached chunks from {path}")
    return chunks
