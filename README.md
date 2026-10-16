# HybridRag

A hybrid Retrieval-Augmented Generation (RAG) system for querying legal documents, combining semantic and keyword retrieval, neural reranking, and semantic response caching with Redis.

## Overview

HybridRag is designed to answer questions about legal documents such as the GDPR (General Data Protection Regulation). It combines document-level summarization with retrieval-based question answering to handle both broad questions and detailed, provision-specific queries.

The project also explores semantic response caching to reduce redundant LLM inference when users ask questions that are sufficiently similar to previously answered questions.

## Architecture

### 1. Document Processing
- Load legal PDF documents.
- Split document content into semantic chunks.
- Generate and reuse cached document summaries.
- Preserve chunk content for retrieval and answer generation.

### 2. Hybrid Retrieval
- **Dense retrieval:** Nomic embeddings with FAISS.
- **Keyword retrieval:** BM25.
- **Hybrid search:** Combine dense and lexical retrieval results.
- **Reranking:** Use `BAAI/bge-reranker-large` to improve the relevance of retrieved passages.

### 3. Overview Routing
- Embed incoming questions using `nomic-ai/nomic-embed-text-v1.5`.
- Compare questions against predefined overview-question embeddings.
- Route broad questions to the document summary.
- Route detailed legal questions to the retrieval pipeline.

### 4. Semantic Response Caching
Redis stores previously generated answers alongside their original questions and embeddings.

The intended workflow is:

`Question → Semantic Cache → Cache Hit or Miss`

- **Cache hit:** Return a sufficiently similar cached answer without another LLM call.
- **Cache miss:** Continue through overview routing or hybrid retrieval, generate an answer, and cache the result.
- **Similarity matching:** Compare normalized embeddings using cosine similarity.
- **Cache metadata:** Store the source document version to help manage stale answers.

Semantic similarity alone does not guarantee that two legal questions have the same intent. Cache thresholds and answer validity must be evaluated carefully.

### 5. Answer Generation
- Use retrieved and reranked context for detailed questions.
- Generate responses using an OpenAI-compatible chat completion endpoint.
- Use Qwen3 30B A3B for local inference on a GPU-backed server.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Embeddings | Nomic Embed v1.5 |
| Dense retrieval | FAISS |
| Keyword retrieval | BM25 |
| Hybrid retrieval | LangChain EnsembleRetriever |
| Reranking | BGE Reranker Large |
| Response cache | Redis |
| Containerization | Docker |
| LLM | Qwen3 30B A3B |
| Document processing | PyMuPDF / LangChain |
| Chunking | Semantic chunking |

## Project Structure

```text
HybridRag/
├── data/
│   ├── semantic_chunks.json
│   └── document_summary.json
├── src/
│   ├── models.py
│   ├── embeddings.py
│   └── document_summary.py
├── main.py
├── .env
├── .env.example
└── README.md
```

*The structure above represents the main components; adapt it to match the actual files in your repository.*

## Getting Started

### 1. Create and activate a virtual environment

```bash
uv venv --python 3.11
source .venv/bin/activate
```

Install the dependencies defined by your project:

```bash
uv sync
```

### 2. Start Redis using Docker

```bash
docker run -d \
  --name hybridrag-redis \
  -p 127.0.0.1:6379:6379 \
  redis:7-alpine
```

Verify that the container is running:

```bash
docker ps
```

Test the Redis connection:

```bash
docker exec -it hybridrag-redis redis-cli ping
```

Expected output:

```text
PONG
```

### 3. Configure environment variables

Create a `.env` file for the configuration required by your project, including the LLM server URL.

Keep credentials and private configuration out of version control. Commit `.env.example` with placeholder values instead of real secrets.

### 4. Run the application

```bash
python main.py
```

Run the command from the project root with the virtual environment activated.

## Redis Management

**Check Redis status**

```bash
docker ps
```

**Inspect Redis keys**

```bash
docker exec -it hybridrag-redis redis-cli
```

Then, inside the Redis CLI:

```redis
SCAN 0 MATCH semantic_cache:* COUNT 100
```

**Clear semantic response cache**

```bash
docker exec hybridrag-redis redis-cli --scan --pattern 'semantic_cache:*' | while read key; do
  docker exec hybridrag-redis redis-cli DEL "$key"
done
```

**Stop Redis**

```bash
docker stop hybridrag-redis
```

**Start the existing container again**

```bash
docker start hybridrag-redis
```

**Remove the container**

```bash
docker stop hybridrag-redis
docker rm hybridrag-redis
```

**Remove the Docker image**

```bash
docker rmi redis:7-alpine
```

The current Redis container does not have an explicitly configured persistent volume, so removing it can permanently discard its stored cache data.

## Current Development Goals

- Improve semantic routing between document overview and detailed retrieval.
- Evaluate semantic cache thresholds against paraphrased and unrelated legal questions.
- Prevent stale cached answers when the source document changes.
- Measure cache hit rate, retrieval quality, and LLM latency.
- Explore persistent vector indexing and more scalable cache lookup strategies.

## Important Note

This is an experimental legal-document QA system, not a substitute for professional legal advice. Answers should be checked against the original legal provisions and authoritative sources.
