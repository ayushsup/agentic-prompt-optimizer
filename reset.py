# reset.py — clears optimization state, preserves OCR cache
import sqlite3
conn = sqlite3.connect("run_state.db")
for table in ("optimization_trajectory", "llm_logs", "metric_cache"):
    conn.execute(f"DELETE FROM {table}")
conn.commit()
conn.close()
print("Done. OCR cache intact.")