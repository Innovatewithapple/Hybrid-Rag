import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langsmith import traceable

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

DATASET_PATH = (
    PROJECT_ROOT
    / "Real_Metrics"
    / "gdpr_questions_011_100.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "evaluation_reports" / "latency_benchmark"
CSV_PATH = OUTPUT_DIR / "latency_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "latency_summary.json"

REPETITIONS = 3

# --------------------------------------------------
# IMPORT EXISTING RAG PIPELINE
# --------------------------------------------------

from main import build_retrieval_components, build_rag_chain
from src.models import Call_LLM
from Prompt.legel_prompt import prompt


# --------------------------------------------------
# DATASET
# --------------------------------------------------

def load_questions(path):

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    questions = []

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not row.get("id"):
                raise ValueError(
                    f"Missing question ID on line {line_number}"
                )

            if not row.get("question", "").strip():
                raise ValueError(
                    f"Missing question text on line {line_number}"
                )

            questions.append(row)

    if not questions:
        raise ValueError("Dataset contains no questions.")

    ids = [row["id"] for row in questions]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs found.")

    texts = [row["question"].strip() for row in questions]

    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate question text found.")

    return questions


# --------------------------------------------------
# LANGSMITH TRACED EXECUTION
# --------------------------------------------------

@traceable(
    name="HybridRAG-Benchmark-Query",
    run_type="chain",
)
def execute_rag(chain, question):

    return chain.invoke(question)


# --------------------------------------------------
# BENCHMARK
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions(DATASET_PATH)

    print(f"Questions loaded: {len(questions)}")
    print(f"Repetitions per question: {REPETITIONS}")
    print(f"Expected measured runs: {len(questions) * REPETITIONS}")

    # Build retrieval components only once.
    print("\nBuilding retrieval components...")

    components = build_retrieval_components()

    # Build the RAG chain only once.
    rag_chain = build_rag_chain(
        hybrid_retriever=components["hybrid_retriever"],
        parent_child_reranker=components["parent_child_reranker"],
        prompt=prompt,
        call_llm=Call_LLM,
    )

    print("\nRAG pipeline initialized.")

    # --------------------------------------------------
    # WARM-UP
    # --------------------------------------------------

    print("\nRunning warm-up (excluded from measurements)...")

    warmup_question = (
        "Briefly explain the main purpose and scope of this regulation."
    )

    warmup_start = time.perf_counter()

    rag_chain.invoke(warmup_question)

    warmup_seconds = time.perf_counter() - warmup_start

    print(f"Warm-up completed: {warmup_seconds:.2f}s")

    # --------------------------------------------------
    # MEASURED EXECUTIONS
    # --------------------------------------------------

    results = []
    successful_latencies = []

    total_runs = len(questions) * REPETITIONS
    run_number = 0

    fieldnames = [
        "timestamp",
        "question_id",
        "repeat",
        "latency_ms",
        "status",
        "answer_characters",
        "error",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        csv_file.flush()
        
        for question_data in questions:

            question_id = question_data["id"]
            question = question_data["question"]

            for repetition in range(1, REPETITIONS + 1):

                run_number += 1

                print(
                    f"\n[{run_number}/{total_runs}] "
                    f"{question_id} | Run {repetition}"
                )

                start = time.perf_counter()

                try:

                    answer = execute_rag(
                        rag_chain,
                        question,
                    )

                    latency_ms = (
                        time.perf_counter() - start
                    ) * 1000

                    successful_latencies.append(latency_ms)

                    result = {
                        "timestamp": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "question_id": question_id,
                        "repeat": repetition,
                        "latency_ms": round(latency_ms, 2),
                        "status": "success",
                        "answer_characters": len(answer),
                        "error": "",
                    }

                    print(f"Latency: {latency_ms / 1000:.2f}s")

                except Exception as exc:

                    latency_ms = (
                        time.perf_counter() - start
                    ) * 1000

                    result = {
                        "timestamp": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "question_id": question_id,
                        "repeat": repetition,
                        "latency_ms": round(latency_ms, 2),
                        "status": "failed",
                        "answer_characters": 0,
                        "error": str(exc),
                    }

                    print(f"FAILED: {exc}")

                results.append(result)

                writer.writerow(result)
                csv_file.flush()
                time.sleep(5)

    # --------------------------------------------------
    # LATENCY STATISTICS
    # --------------------------------------------------

    if not successful_latencies:
        raise RuntimeError(
            "All benchmark executions failed. "
            "No latency statistics can be calculated."
        )

    p50, p95 = np.percentile(
        successful_latencies,
        [50, 95],
    )

    summary = {
        "dataset": DATASET_PATH.name,
        "unique_questions": len(questions),
        "repetitions_per_question": REPETITIONS,
        "expected_runs": total_runs,
        "successful_runs": len(successful_latencies),
        "failed_runs": total_runs - len(successful_latencies),
        "warmup_latency_seconds": round(warmup_seconds, 3),
        "latency_unit": "milliseconds",
        "p50_latency_ms": round(float(p50), 2),
        "p95_latency_ms": round(float(p95), 2),
        "mean_latency_ms": round(
            float(np.mean(successful_latencies)), 2
        ),
        "min_latency_ms": round(
            float(np.min(successful_latencies)), 2
        ),
        "max_latency_ms": round(
            float(np.max(successful_latencies)), 2
        ),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("LATENCY BENCHMARK COMPLETED")
    print("=" * 50)

    print(f"Unique questions : {len(questions)}")
    print(f"Successful runs  : {len(successful_latencies)}")
    print(f"Failed runs      : {total_runs - len(successful_latencies)}")
    print(f"P50 latency      : {p50 / 1000:.3f}s")
    print(f"P95 latency      : {p95 / 1000:.3f}s")
    print(f"Mean latency     : {np.mean(successful_latencies) / 1000:.3f}s")

    print(f"\nCSV report     : {CSV_PATH}")
    print(f"Summary report : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()