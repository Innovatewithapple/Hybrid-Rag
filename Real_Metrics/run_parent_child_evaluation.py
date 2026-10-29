"""Evaluate parent-child hybrid retrieval with Precision/Recall/MRR/nDCG.

Expected JSONL schema per row:
{
  "id": "gdpr_pc_001",
  "question": "...",
  "relevant_ids": ["parent_0001_child_0002", ...]
}

IMPORTANT: old semantic-chunk IDs are not valid labels for this new index.
Create/review parent-child relevant_ids before running this script.
"""

import csv
import json
import math
from pathlib import Path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from main import build_retrieval_components, retrieve_ranked_documents


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = Path(__file__).resolve().parent / "gdpr_parent_child_evaluation_hard.jsonl"
REPORT_DIR = PROJECT_DIR / "evaluation_reports"
REPORT_PATH = REPORT_DIR / "parent_child_retrieval_metrics.csv"
K_VALUES = (1, 3, 5)

def to_parent_id(child_id):
    return child_id.rsplit("_child_", 1)[0]

def precision_at_k(ranked_ids, relevant_ids, k):
    if k <= 0:
        return 0.0
    return sum(item in relevant_ids for item in ranked_ids[:k]) / k

def r_precision(ranked_ids, relevant_ids):
    r = len(relevant_ids)
    return sum(item in relevant_ids for item in ranked_ids[:r]) / r if r else 0.0

def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids):
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k):
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(ranked_ids[:k], start=1)
        if item_id in relevant_ids
    )
    ideal_relevant_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))
    return dcg / idcg if idcg else 0.0


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}\n"
            "Create this JSONL file with fresh parent-child relevant_ids labels."
        )
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("question"):
                raise ValueError(f"Line {line_number} is missing 'question'.")
            if not isinstance(row.get("relevant_ids"), list) or not row["relevant_ids"]:
                raise ValueError(
                    f"Line {line_number} must contain a non-empty 'relevant_ids' list."
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No evaluation questions found in {path}")
    return rows


def main():
    evaluation_rows = load_dataset(DATASET_PATH)
    components = build_retrieval_components()
    hybrid_retriever = components["hybrid_retriever"]
    parent_child_reranker = components["parent_child_reranker"]

    known_child_ids = {doc.metadata["child_id"] for doc in components["chunks"]}
    labelled_ids = {
        item_id
        for row in evaluation_rows
        for item_id in row["relevant_ids"]
    }
    unknown_ids = labelled_ids - known_child_ids
    if unknown_ids:
        preview = sorted(unknown_ids)[:10]
        raise ValueError(
            "Some relevant_ids do not exist in the current parent-child chunks. "
            "This usually means the old semantic-chunk labels were reused. "
            f"Unknown IDs (first 10): {preview}"
        )

    report_rows = []
    macro_totals = {"mrr": 0.0, "r_precision": 0.0}
    for k in K_VALUES:
        macro_totals[f"precision@{k}"] = 0.0
        macro_totals[f"recall@{k}"] = 0.0
        macro_totals[f"ndcg@{k}"] = 0.0

    for row in evaluation_rows:
        question = row["question"]
        relevant_ids = set(row["relevant_ids"])                      # child ids (kept for the report)
        relevant_parents = {to_parent_id(i) for i in relevant_ids}   # NEW: dedupes to parents

        ranked_docs = retrieve_ranked_documents(
            question, hybrid_retriever, parent_child_reranker
        )
        ranked_ids = [doc.metadata["child_id"] for doc in ranked_docs]        # for the report
        ranked_parents = [doc.metadata["parent_id"] for doc in ranked_docs]   # NEW: used for metrics

        result = {
            "id": row.get("id", ""),
            "question": question,
            "relevant_ids": json.dumps(sorted(relevant_ids), ensure_ascii=False),
            "retrieved_ids": json.dumps(ranked_ids, ensure_ascii=False),
            "retrieved_count": len(ranked_ids),
            "mrr": reciprocal_rank(ranked_parents, relevant_parents),
            "r_precision": r_precision(ranked_parents, relevant_parents),
        }
        for k in K_VALUES:
            result[f"precision@{k}"] = precision_at_k(ranked_parents, relevant_parents, k)
            result[f"recall@{k}"] = recall_at_k(ranked_parents, relevant_parents, k)
            result[f"ndcg@{k}"] = ndcg_at_k(ranked_parents, relevant_parents, k)

        report_rows.append(result)
        for metric in macro_totals:
            macro_totals[metric] += result[metric]

        print(f"\n[{row.get('id', 'question')}] {question}")
        print("Relevant child IDs:", sorted(relevant_ids))
        print("Ranked child IDs:  ", ranked_ids)
        print(
            " | ".join(
                [f"MRR={result['mrr']:.3f}"]
                + [f"P@{k}={result[f'precision@{k}']:.3f}" for k in K_VALUES]
                + [f"R@{k}={result[f'recall@{k}']:.3f}" for k in K_VALUES]
                + [f"nDCG@{k}={result[f'ndcg@{k}']:.3f}" for k in K_VALUES]
            )
        )

    count = len(report_rows)
    averages = {metric: value / count for metric, value in macro_totals.items()}
    print("\n========== MACRO-AVERAGED RETRIEVAL METRICS ==========")
    for metric, value in averages.items():
        print(f"{metric}: {value:.4f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(report_rows[0].keys())
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    summary_path = REPORT_DIR / "parent_child_retrieval_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"question_count": count, "macro_averages": averages},
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved per-question metrics: {REPORT_PATH}")
    print(f"Saved summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
