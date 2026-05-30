"""
SQLite-backed persistence layer for the optimizer.

Responsibilities:
  - Log every LLM call (input, output, token usage, latency, cost)
  - Track the optimization trajectory (prompt, val_score, accepted) per iteration
  - Cache stochastic metric results (string_semantic, array_llm) for determinism
  - Cache OCR / PDF text extraction results so re-runs skip expensive re-extraction
  - Cache extractor predictions keyed by (prompt_hash, doc_id) so rejected iterations
    that evaluate the same prompt on the same document never call the LLM twice
  - Enable interrupted runs to resume from the last valid checkpoint

Database: run_state.db (created automatically in the working directory)
"""

import hashlib
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


class StateManager:
    def __init__(self, db_path: str = "run_state.db"):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Every LLM interaction
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_logs (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    role          TEXT    NOT NULL,
                    model         TEXT    NOT NULL,
                    prompt        TEXT    NOT NULL,
                    response      TEXT    NOT NULL,
                    input_tokens  INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    cost          REAL    DEFAULT 0.0,
                    latency_ms    REAL    DEFAULT 0.0,
                    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Prompt iteration history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_trajectory (
                    iteration INTEGER PRIMARY KEY,
                    prompt    TEXT    NOT NULL,
                    val_score REAL    NOT NULL,
                    accepted  BOOLEAN NOT NULL,
                    breakdown TEXT    DEFAULT '{}',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Deterministic cache for stochastic metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metric_cache (
                    cache_key TEXT PRIMARY KEY,
                    score     REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # OCR / PDF text extraction cache.
            # Keyed by SHA-256 of the raw PDF bytes so the cache is
            # automatically invalidated if a PDF file is replaced.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ocr_cache (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    text      TEXT NOT NULL,
                    method    TEXT NOT NULL DEFAULT 'unknown',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Extractor prediction cache.
            # Keyed by SHA-256(prompt + doc_id) so the same prompt evaluated on
            # the same document is never re-sent to the LLM.  This eliminates
            # the API call on every rejected iteration that reverts to best_prompt
            # before a new mutation is proposed.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    cache_key  TEXT PRIMARY KEY,
                    doc_id     TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    # ------------------------------------------------------------------
    # LLM call logging
    # ------------------------------------------------------------------

    def log_llm_call(
        self,
        role: str,
        prompt: str,
        response: str,
        model: str,
        usage: Dict[str, int],
        cost: float,
        latency_ms: float = 0.0,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO llm_logs
                    (role, model, prompt, response, input_tokens, output_tokens, cost, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    role,
                    model,
                    prompt,
                    response,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    cost,
                    latency_ms,
                ),
            )
            conn.commit()

    def get_total_cost(self) -> float:
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("SELECT SUM(cost) FROM llm_logs").fetchone()[0]
        return result if result else 0.0

    # ------------------------------------------------------------------
    # Optimization trajectory
    # ------------------------------------------------------------------

    def log_iteration(
        self,
        iteration: int,
        prompt: str,
        val_score: float,
        accepted: bool,
        breakdown: Optional[Dict] = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO optimization_trajectory
                    (iteration, prompt, val_score, accepted, breakdown)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    iteration,
                    prompt,
                    val_score,
                    accepted,
                    json.dumps(breakdown or {}),
                ),
            )
            conn.commit()

    def get_trajectory(self) -> List[Dict]:
        """Return the full recorded trajectory, ordered by iteration."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT iteration, prompt, val_score, accepted FROM optimization_trajectory ORDER BY iteration"
            ).fetchall()
        return [
            {"iteration": r[0], "prompt": r[1], "val_score": r[2], "accepted": bool(r[3])}
            for r in rows
        ]

    def get_best_state(self) -> Optional[Dict]:
        """
        Return the best accepted prompt and score from previous runs.
        Used for resumability: the optimizer can warm-start from here.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT prompt, val_score FROM optimization_trajectory
                WHERE accepted = 1
                ORDER BY val_score DESC, iteration DESC
                LIMIT 1
                """
            ).fetchone()
        if row:
            return {"prompt": row[0], "val_score": row[1]}
        return None

    def get_last_completed_iteration(self) -> int:
        """Returns the highest recorded iteration index, or -1 if none."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(iteration) FROM optimization_trajectory"
            ).fetchone()
        return row[0] if row[0] is not None else -1

    def get_stall_count(self) -> int:
        """
        Return the number of consecutive rejected iterations at the end of
        the recorded trajectory.  Used to restore stall_count on resume so
        the Mutator's escalation logic (train-example injection, beam secondary,
        radical rewrite) fires at the right iteration after an interrupted run.
        """
        trajectory = self.get_trajectory()
        count = 0
        for entry in reversed(trajectory):
            if not entry["accepted"]:
                count += 1
            else:
                break
        return count

    def get_rejected_prompts(self) -> List[str]:
        """Return all prompts that were evaluated but rejected."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT prompt FROM optimization_trajectory WHERE accepted = 0 ORDER BY iteration"
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Stochastic metric cache
    # ------------------------------------------------------------------

    def get_metric_cache(self, cache_key: str) -> Optional[float]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT score FROM metric_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        return row[0] if row else None

    def set_metric_cache(self, cache_key: str, score: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metric_cache (cache_key, score) VALUES (?, ?)",
                (cache_key, score),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # OCR / PDF text extraction cache
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(file_path: str) -> str:
        """Return the SHA-256 hex digest of a file's raw bytes."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_ocr_cache(self, file_path: str) -> Optional[str]:
        """
        Return cached extracted text for a PDF, or None if not cached.

        The cache key is the SHA-256 of the file bytes, so replacing a file
        with a new version automatically bypasses the stale cache entry.
        """
        try:
            file_hash = self._hash_file(file_path)
        except OSError:
            return None
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT text FROM ocr_cache WHERE file_hash = ?", (file_hash,)
            ).fetchone()
        return row[0] if row else None

    def set_ocr_cache(self, file_path: str, text: str, method: str = "unknown") -> None:
        """Persist extracted text for a PDF so future runs skip re-extraction."""
        try:
            file_hash = self._hash_file(file_path)
        except OSError:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ocr_cache (file_hash, file_path, text, method)
                VALUES (?, ?, ?, ?)
                """,
                (file_hash, file_path, text, method),
            )
            conn.commit()

    def get_ocr_cache_stats(self) -> Dict:
        """Return summary stats on the OCR cache (useful for debugging)."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM ocr_cache").fetchone()[0]
            by_method = conn.execute(
                "SELECT method, COUNT(*) FROM ocr_cache GROUP BY method"
            ).fetchall()
        return {
            "total_cached": total,
            "by_method": {row[0]: row[1] for row in by_method},
        }

    # ------------------------------------------------------------------
    # Extractor prediction cache
    # ------------------------------------------------------------------

    @staticmethod
    def _prediction_key(prompt: str, doc_id: str) -> str:
        """
        Deterministic SHA-256 key for a (prompt, doc_id) pair.

        Using only doc_id (not the full document text) is intentional: the
        document text is fixed for the lifetime of a run, so doc_id is a
        sufficient discriminator.  This keeps the key short and stable.
        """
        payload = f"{prompt}\n|||PIBIT_SEP|||\n{doc_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_prediction(self, prompt: str, doc_id: str) -> Optional[str]:
        """
        Return the cached extractor output for this (prompt, doc_id) pair,
        or None if not yet cached.

        Cache hits avoid an LLM call entirely — critical for stall iterations
        where the prompt has not changed between evaluations.
        """
        key = self._prediction_key(prompt, doc_id)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT prediction FROM prediction_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set_prediction(self, prompt: str, doc_id: str, prediction: str) -> None:
        """Persist an extractor output for future cache lookups."""
        key = self._prediction_key(prompt, doc_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO prediction_cache (cache_key, doc_id, prediction)
                VALUES (?, ?, ?)
                """,
                (key, doc_id, prediction),
            )
            conn.commit()

    def get_prediction_cache_stats(self) -> Dict:
        """Return summary stats on the prediction cache."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM prediction_cache").fetchone()[0]
        return {"total_cached": total}