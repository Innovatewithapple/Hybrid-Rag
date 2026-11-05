# HybridRag

A hybrid Retrieval-Augmented Generation (RAG) system for legal-document
question answering. It combines dense and keyword retrieval,
cross-encoder reranking, article-aware parent-child chunking,
document-overview routing, and semantic response caching.

## Evaluation & Performance

Evaluated across **5 legal PDFs** with different structures, including
article/chapter-based legislation and unstructured plain text.

### Retrieval quality

Primary benchmark: **100-question GDPR evaluation set**.

| Metric                                        |    Result |
|:----------------------------------------------|----------:|
| Mean Reciprocal Rank (MRR)                    | **0.925** |
| nDCG@5                                        | **0.918** |
| Correct chunk ranked first                    |   **87%** |
| Relevant chunks retrieved in top 5 (Recall@5) |   **96%** |
| MRR improvement over baseline                 |  **+59%** |

Additional evaluations across smaller datasets (5–10 questions per
document) achieved MRR scores between **0.88 and 1.00**. The weakest
results were on unstructured plain-text documents, where document
structure provides fewer reliable article and section boundaries.

### End-to-end latency

Measured across **100 questions × 3 repetitions (300 attempted runs)**,
using LangSmith tracing and per-stage profiling.

| Metric               | Earlier configuration | Optimized configuration |
|:---------------------|----------------------:|------------------------:|
| P50 / median latency |                 4.64s |               **1.70s** |
| P50 reduction        |                     — |                 **63%** |
| BGE reranking time   |                ~0.70s |              **~0.20s** |
| P95 latency          |                     — |               **7.80s** |
| Mean latency         |                     — |               **2.59s** |

The optimized configuration combined: - **Gemini 3.5 Flash-Lite**
instead of self-hosted Qwen3-30B for generation. - **FP16 inference**
for the BGE cross-encoder reranker.

The latest benchmark summary recorded **292 successful runs and 8 failed
runs** out of 300 attempts. On successful runs, minimum latency was
**0.87s** and maximum latency was **52.70s**.

*P50 is the median: half of successful requests completed at or below
this latency. P95 is the threshold at or below which 95% of successful
requests completed. The 63% P50 reduction compares configurations that
changed both the generation provider and reranking precision; it should
not be attributed to either change alone. Results depend on hardware,
model-serving conditions, network latency, and workload.*

## How It Works

1.  **Document processing:** Load legal PDFs, extract text, create
    semantic chunks, and generate reusable document summaries.
2.  **Article-aware parent-child chunking:** Preserve complete legal
    clauses and article context while retrieving smaller child chunks.
3.  **Hybrid retrieval:** Combine Nomic dense retrieval through FAISS
    with BM25 keyword retrieval using a weighted ensemble.
4.  **Reranking:** Score retrieved parent-level context with
    `BAAI/bge-reranker-large`.
5.  **Overview routing:** Route broad questions to document summaries
    and provision-specific questions to retrieval.
6.  **Answer generation:** Format the selected context and send it to an
    OpenAI-compatible chat-completion endpoint.
7.  **Semantic response cache:** Redis can return a previously generated
    answer for a sufficiently similar question, subject to the
    configured similarity threshold and document-version metadata.

## Architecture

``` text
                         User question
                               |
                    Semantic response cache
                         /           \
                     Cache hit      Cache miss
                        |               |
                   Cached answer   Overview routing
                                      /       \
                              Broad query   Detailed query
                                  |              |
                          Document summary   Hybrid retrieval
                                             /          \
                                        FAISS dense    BM25 lexical
                                             \          /
                                           Weighted ensemble
                                                  |
                                      Parent-child reranking
                                                  |
                                          Context formatting
                                                  |
                                            LLM generation
                                                  |
                                               Answer
```

## Tech Stack

| Component           | Technology                                                                    |
|:--------------------|:------------------------------------------------------------------------------|
| Language            | Python 3.11                                                                   |
| Embeddings          | `nomic-ai/nomic-embed-text-v1.5`                                              |
| Dense retrieval     | FAISS                                                                         |
| Keyword retrieval   | BM25                                                                          |
| Hybrid retrieval    | LangChain `EnsembleRetriever`                                                 |
| Reranking           | `BAAI/bge-reranker-large`                                                     |
| Chunking            | Semantic chunking; article-aware parent-child structure                       |
| Response cache      | Redis                                                                         |
| LLM                 | OpenAI-compatible chat-completion endpoint (Qwen3-30B or configured provider) |
| Document processing | PyMuPDF / LangChain                                                           |
| Tracing             | LangSmith                                                                     |
| Containerization    | Docker (Redis)                                                                |

## Project Structure

``` text
HybridRag/
├── data/                       # Source documents and processed data
├── src/                        # Models, embeddings, reranking, utilities
├── Prompt/
│   └── legel_prompt.py         # RAG prompt
├── evaluation_reports/         # Generated evaluation outputs
├── main.py                     # Main RAG application / pipeline entry point
├── benchmark_latency.py        # End-to-end latency benchmark
├── .env.example                # Environment-variable template
└── README.md
```

*This is a high-level structure; filenames and data subfolders may vary
with the checked-out version.*

## Setup

### 1. Create the environment

``` bash
uv venv --python 3.11
source .venv/bin/activate
```

Install the dependencies used by the project:

``` bash
uv sync
```

If your checkout does not include a `pyproject.toml` / lockfile, install
the project’s dependencies into the active environment rather than
copying a virtual environment between machines.

### 2. Configure environment variables

Create a local `.env` file using `.env.example` as a guide. Configure
the LLM server URL and any required credentials.

For the latency benchmark, ensure the configured endpoint is reachable
for the full duration of the run. Keep API keys, private endpoint URLs,
and other secrets out of version control.

### 3. Start Redis

Redis is used by the semantic response cache.

``` bash
docker run -d \
  --name hybridrag-redis \
  -p 127.0.0.1:6379:6379 \
  redis:7-alpine
```

If the container already exists, start it with:

``` bash
docker start hybridrag-redis
```

Verify the connection:

``` bash
docker exec -it hybridrag-redis redis-cli ping
```

Expected output:

``` text
PONG
```

## Run the Application

Run the main RAG application / pipeline from the project root:

``` bash
uv run --active python main.py
```

## Run the Latency Benchmark

Run the end-to-end latency benchmark from the project root:

``` bash
uv run --active python benchmark_latency.py
```

The benchmark performs a warm-up, runs the configured question set
repeatedly, records per-run latency, and writes its reports to:

``` text
evaluation_reports/latency_benchmark/
├── latency_results.csv
└── latency_summary.json
```

The CSV contains per-run measurements. The JSON contains aggregate
statistics such as successful/failed runs, P50, P95, mean, minimum, and
maximum latency.

**Benchmark notes** - The reported benchmark uses 100 questions with 3
repetitions each (300 attempted runs). - Keep the LLM endpoint available
throughout the benchmark. - Hardware, GPU backend, model precision,
endpoint location, and network conditions affect results. - Compare runs
only when the question set, repetition count, model configuration, and
measurement boundaries are consistent. - `benchmark_latency.py` measures
runtime performance; it does not by itself calculate retrieval-quality
metrics such as MRR or nDCG.

## Redis Management

``` bash
# Check running containers
docker ps

# Open Redis CLI
docker exec -it hybridrag-redis redis-cli

# Stop Redis
docker stop hybridrag-redis

# Remove the container (stored cache data is lost without a persistent volume)
docker rm hybridrag-redis
```

## Limitations

- Retrieval quality depends on document extraction quality, chunk
  boundaries, and the completeness of relevance labels in the evaluation
  dataset.
- Plain-text legal documents without explicit article or section
  structure can be harder to retrieve precisely.
- Semantic cache similarity does not guarantee identical legal intent.
  Cache thresholds and cached-answer validity should be evaluated
  carefully, especially when source documents change.
- This project is an experimental legal-document QA system, not a
  substitute for professional legal advice. Verify answers against the
  original legal provisions and authoritative sources.
