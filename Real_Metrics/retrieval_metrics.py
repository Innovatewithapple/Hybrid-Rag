"""Reusable retrieval metrics for RAG evaluation."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MetricResult:
    question_id: str
    question: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    retrieved_count: int
    relevant_count: int
    hits_at_k: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "retrieved_count": self.retrieved_count,
            "relevant_count": self.relevant_count,
            "hits_at_k": self.hits_at_k,
        }


class RetrievalMetrics:
    """Calculate Precision@K, Recall@K, MRR and nDCG@K."""

    def __init__(self, k: int = 5):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k

    @staticmethod
    def _unique_in_order(values: Iterable[str]) -> list[str]:
        seen = set()
        output = []

        for value in values:
            if value not in seen:
                seen.add(value)
                output.append(value)

        return output

    def score_one(
        self,
        *,
        question_id: str,
        question: str,
        ranked_ids: Sequence[str],
        relevant_ids: Sequence[str],
    ) -> MetricResult:
        """Score one question against its manually labelled relevant chunks."""

        ranked = self._unique_in_order(ranked_ids)
        relevant = set(relevant_ids)

        if not relevant:
            raise ValueError(
                f"{question_id}: relevant_ids cannot be empty"
            )

        top_k = ranked[:self.k]

        hits = sum(
            1 for chunk_id in top_k
            if chunk_id in relevant
        )

        # Precision@K: relevant results in top K / K
        precision = hits / self.k

        # Recall@K: relevant results in top K / all known relevant chunks
        recall = hits / len(relevant)

        # MRR: reciprocal rank of the first relevant result
        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(ranked, start=1):
            if chunk_id in relevant:
                reciprocal_rank = 1.0 / rank
                break

        # DCG@K with binary relevance
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, chunk_id in enumerate(top_k, start=1)
            if chunk_id in relevant
        )

        # Ideal DCG@K
        ideal_count = min(len(relevant), self.k)

        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )

        ndcg = dcg / idcg if idcg else 0.0

        return MetricResult(
            question_id=question_id,
            question=question,
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=reciprocal_rank,
            ndcg_at_k=ndcg,
            retrieved_count=len(ranked),
            relevant_count=len(relevant),
            hits_at_k=hits,
        )

    def evaluate(
        self,
        evaluation_cases: Sequence[Mapping[str, Any]],
        retrieve: Callable[[str], Sequence[str]],
    ) -> list[MetricResult]:
        """Run retrieval and calculate metrics for every question."""

        results = []

        for index, case in enumerate(evaluation_cases, start=1):
            question_id = str(
                case.get(
                    "question_id",
                    case.get("id", f"question_{index:03d}")
                )
            )

            question = str(case["question"])
            relevant_ids = case["relevant_ids"]

            # Must return chunk IDs in ranked order, best first.
            ranked_ids = retrieve(question)

            results.append(
                self.score_one(
                    question_id=question_id,
                    question=question,
                    ranked_ids=ranked_ids,
                    relevant_ids=relevant_ids,
                )
            )

        if not results:
            raise ValueError("evaluation_cases cannot be empty")

        return results

    @staticmethod
    def summary(
        results: Sequence[MetricResult]
    ) -> dict[str, float]:
        """Calculate macro-average metrics across questions."""

        if not results:
            raise ValueError("results cannot be empty")

        return {
            "precision_at_k": mean(
                result.precision_at_k for result in results
            ),
            "recall_at_k": mean(
                result.recall_at_k for result in results
            ),
            "mrr": mean(
                result.mrr for result in results
            ),
            "ndcg_at_k": mean(
                result.ndcg_at_k for result in results
            ),
        }

    def print_report(
        self,
        results: Sequence[MetricResult]
    ) -> None:
        """Print per-question metrics and overall averages."""

        if not results:
            raise ValueError("results cannot be empty")

        headers = (
            "Question ID",
            "P@K",
            "R@K",
            "MRR",
            "nDCG@K",
            "Hits@K",
        )

        rows = [
            (
                result.question_id,
                f"{result.precision_at_k:.3f}",
                f"{result.recall_at_k:.3f}",
                f"{result.mrr:.3f}",
                f"{result.ndcg_at_k:.3f}",
                f"{result.hits_at_k}/{self.k}",
            )
            for result in results
        ]

        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rows))
            for i in range(len(headers))
        ]

        line = "-+-".join("-" * width for width in widths)

        print(
            f"\nRETRIEVAL EVALUATION | K={self.k} "
            f"| Questions={len(results)}"
        )

        print(
            " | ".join(
                headers[i].ljust(widths[i])
                for i in range(len(headers))
            )
        )

        print(line)

        for row in rows:
            print(
                " | ".join(
                    row[i].ljust(widths[i])
                    for i in range(len(headers))
                )
            )

        averages = self.summary(results)

        print(line)

        mean_row = [
            "MEAN",
            f"{averages['precision_at_k']:.3f}",
            f"{averages['recall_at_k']:.3f}",
            f"{averages['mrr']:.3f}",
            f"{averages['ndcg_at_k']:.3f}",
            "-",
        ]

        print(
            " | ".join(
                mean_row[i].ljust(widths[i])
                for i in range(len(headers))
            )
        )

        print(
            "\nP@K and R@K use the top K results. "
            "MRR uses the full returned ranking."
        )

    @staticmethod
    def save_csv(
        results: Sequence[MetricResult],
        output_path: str | Path,
    ) -> Path:
        """Save individual results and the overall mean to CSV."""

        if not results:
            raise ValueError("results cannot be empty")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(results[0].as_dict().keys())

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for result in results:
                writer.writerow(result.as_dict())

            averages = RetrievalMetrics.summary(results)

            mean_row = {field: "" for field in fieldnames}
            mean_row.update({
                "question_id": "MEAN",
                **averages,
            })

            writer.writerow(mean_row)

        return path