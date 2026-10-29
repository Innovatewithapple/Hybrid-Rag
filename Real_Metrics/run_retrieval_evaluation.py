"""Run standalone retrieval metrics on the verified GDPR dataset.

Run from the HybridRag project root:

uv run --active python run_retrieval_evaluation.py
"""

from __future__ import annotations

import json
from pathlib import Path

from retrieval_metrics import RetrievalMetrics
from retrieval_adapter import HybridRagRetrievalAdapter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "gdpr_evaluation_dataset_5_verified.jsonl"
)

CSV_PATH = (
    PROJECT_ROOT
    / "evaluation_reports"
    / "retrieval_metrics.csv"
)

K = 5
RETRIEVAL_K = 10


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}\n"
            "Copy gdpr_evaluation_dataset_5_verified.jsonl "
            "into the project's data/ folder."
        )

    cases = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    if not cases:
        raise ValueError(
            f"No evaluation cases found in {path}"
        )

    return cases


def main() -> None:
    cases = load_jsonl(DATASET_PATH)

    for index, case in enumerate(cases, start=1):
        case.setdefault(
            "question_id",
            case.get("id", f"question_{index:03d}"),
        )

    # Build evaluation-only indexes and validate the chunk IDs.
    adapter = HybridRagRetrievalAdapter(
        retrieval_k=RETRIEVAL_K
    )

    for case in cases:
        missing = (
            set(case["relevant_ids"])
            - adapter.valid_chunk_ids
        )

        if missing:
            raise ValueError(
                f"{case.get('question_id', 'unknown')} "
                f"references missing chunk IDs: {sorted(missing)}"
            )

    metrics = RetrievalMetrics(k=K)

    results = metrics.evaluate(
        evaluation_cases=cases,
        retrieve=adapter.retrieve_ids,
    )

    metrics.print_report(results)

    saved_path = metrics.save_csv(
        results,
        CSV_PATH,
    )

    print(f"\nCSV saved to: {saved_path}")


if __name__ == "__main__":
    main()