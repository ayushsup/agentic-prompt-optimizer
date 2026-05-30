# reset.py — clears optimization state, preserves OCR cache
#
# Tables cleared:
#   optimization_trajectory  — iteration history and accepted/rejected prompts
#   llm_logs                 — all LLM call records
#   metric_cache             — stochastic metric scores (string_semantic, array_llm)
#   prediction_cache         — extractor outputs keyed by (prompt, doc_id)
#
# Table preserved:
#   ocr_cache                — PDF text extraction results (expensive, keyed by file hash)
#
# Usage:
#   python reset.py          — full reset (keeps OCR cache)
#   python reset.py --all    — full reset including OCR cache

import sqlite3
import sys

keep_ocr = "--all" not in sys.argv

conn = sqlite3.connect("run_state.db")

tables_to_clear = ["optimization_trajectory", "llm_logs", "metric_cache", "prediction_cache"]
if not keep_ocr:
    tables_to_clear.append("ocr_cache")

for table in tables_to_clear:
    try:
        conn.execute(f"DELETE FROM {table}")
        print(f"  ✓ Cleared: {table}")
    except Exception as e:
        print(f"  ⚠  Could not clear {table}: {e}")

conn.commit()
conn.close()

if keep_ocr:
    print("\nDone. OCR cache preserved — PDFs will not be re-extracted.")
else:
    print("\nDone. Full reset including OCR cache.")