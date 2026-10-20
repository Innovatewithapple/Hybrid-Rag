"""Run from the HybridRag project root after installing this toolkit."""
from pathlib import Path
from rag_eval.dataset import load_jsonl
from rag_eval.judge import CallableJudge
from rag_eval.runner import evaluate_cases
from rag_eval.reports import write_csv, write_json, aggregate_numeric
from rag_eval.qwen_client import Call_Evaluation_LLM
from hybridrag_eval_adapter import run_rag

DATASET = Path("evaluation_dataset.jsonl")
REPORTS = Path("evaluation_reports")

cases = load_jsonl(DATASET)
judge = CallableJudge(Call_Evaluation_LLM)
rows = evaluate_cases(
    cases,
    run_rag,
    judge=judge,
    k_values=(1, 3, 5, 10),
    judge_context_metrics=True,
)
write_csv(rows, REPORTS / "evaluation.csv")
write_json(rows, REPORTS / "evaluation.json")
write_json([aggregate_numeric(rows)], REPORTS / "summary.json")
print(f"Saved evaluation reports to: {REPORTS.resolve()}")
print(aggregate_numeric(rows))
