import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR      = PROJECT_ROOT                      # where your PDFs are
METRICS_DIR  = PROJECT_ROOT / "Real_Metrics"     # chunk JSONs and eval .jsonl files
REPORT_DIR   = PROJECT_ROOT / "evaluation_reports"

# ===== EDIT ONLY THESE TWO LINES WHEN YOU SWITCH PDF =====
PDF_NAME  = os.environ.get("RAG_PDF",  "regulation eu 2016679 of the european parliament and-DOC_1.pdf")
EVAL_FILE = os.environ.get("RAG_EVAL", "gdpr_questions_011_100.jsonl")
# =========================================================

# Everything below is derived from the two lines above. Don't edit.
DOC_LABEL = re.sub(r"[^A-Za-z0-9]+", "_", Path(PDF_NAME).stem).strip("_").lower()[:40]

PDF_PATH         = PDF_DIR / PDF_NAME
CHUNKS_JSON_PATH = METRICS_DIR / f"{DOC_LABEL}_parent_child_chunks_with_ids.json"
EVAL_PATH        = METRICS_DIR / EVAL_FILE
REPORT_PATH      = REPORT_DIR / f"{DOC_LABEL}_{EVAL_PATH.stem}_metrics.csv"
SUMMARY_PATH     = REPORT_DIR / f"{DOC_LABEL}_{EVAL_PATH.stem}_summary.json"