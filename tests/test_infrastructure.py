"""
Tests for OCR caching, prediction caching, stall count recovery,
BaseAgent four-tier fallback, and the Loader cache integration.

Fallback chain tested:
  Tier 1 — OpenRouter  (primary)
  Tier 2 — GitHub Models
  Tier 3 — Gemini text
  Tier 4 — Ollama      (terminal)

Run with: pytest tests/test_infrastructure.py -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

from src.core.state_manager import StateManager


# ─────────────────────────────────────────────────────────────────────────────
# StateManager OCR cache tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOCRCache:
    """Verify the SQLite OCR cache stores, retrieves, and invalidates correctly."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.sm = StateManager(db_path=self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def _write_tmp_pdf(self, content: bytes) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_cache_miss_returns_none(self):
        pdf = self._write_tmp_pdf(b"%PDF-1.4 fake content")
        try:
            assert self.sm.get_ocr_cache(pdf) is None
        finally:
            os.unlink(pdf)

    def test_set_and_get_roundtrip(self):
        pdf = self._write_tmp_pdf(b"%PDF-1.4 fake content A")
        try:
            self.sm.set_ocr_cache(pdf, "Extracted text here", method="pymupdf")
            assert self.sm.get_ocr_cache(pdf) == "Extracted text here"
        finally:
            os.unlink(pdf)

    def test_cache_invalidated_on_file_change(self):
        pdf = self._write_tmp_pdf(b"%PDF-1.4 version one")
        try:
            self.sm.set_ocr_cache(pdf, "Old text", method="pymupdf")
            with open(pdf, "wb") as f:
                f.write(b"%PDF-1.4 version TWO -- completely different")
            assert self.sm.get_ocr_cache(pdf) is None
        finally:
            os.unlink(pdf)

    def test_set_overwrites_existing_entry(self):
        pdf = self._write_tmp_pdf(b"%PDF-1.4 stable bytes")
        try:
            self.sm.set_ocr_cache(pdf, "First extraction", method="pymupdf")
            self.sm.set_ocr_cache(pdf, "Updated extraction", method="pdfplumber")
            assert self.sm.get_ocr_cache(pdf) == "Updated extraction"
        finally:
            os.unlink(pdf)

    def test_cache_stats_empty(self):
        stats = self.sm.get_ocr_cache_stats()
        assert stats["total_cached"] == 0
        assert stats["by_method"] == {}

    def test_cache_stats_after_inserts(self):
        pdfs = []
        try:
            for i, method in enumerate(["pymupdf", "pymupdf", "gemini_ocr"]):
                pdf = self._write_tmp_pdf(f"%PDF fake {i}".encode())
                pdfs.append(pdf)
                self.sm.set_ocr_cache(pdf, f"text {i}", method=method)
            stats = self.sm.get_ocr_cache_stats()
            assert stats["total_cached"] == 3
            assert stats["by_method"]["pymupdf"] == 2
            assert stats["by_method"]["gemini_ocr"] == 1
        finally:
            for pdf in pdfs:
                os.unlink(pdf)

    def test_nonexistent_file_returns_none(self):
        assert self.sm.get_ocr_cache("/nonexistent/path/file.pdf") is None

    def test_cache_does_not_affect_metric_cache(self):
        pdf = self._write_tmp_pdf(b"%PDF-1.4 content")
        try:
            self.sm.set_ocr_cache(pdf, "text", method="pymupdf")
            assert self.sm.get_metric_cache("some_key") is None
        finally:
            os.unlink(pdf)


# ─────────────────────────────────────────────────────────────────────────────
# StateManager prediction cache tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionCache:
    """
    Verify the prediction cache stores, retrieves, and discriminates
    (prompt, doc_id) pairs correctly.

    The prediction cache is the main token-conservation mechanism: when a
    rejected iteration reverts to best_prompt, the subsequent evaluation of
    best_prompt should be served entirely from the cache.
    """

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.sm = StateManager(db_path=self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_miss_returns_none(self):
        assert self.sm.get_prediction("some prompt", "doc-001") is None

    def test_set_and_get_roundtrip(self):
        self.sm.set_prediction("prompt A", "doc-001", '{"name": "Alice"}')
        result = self.sm.get_prediction("prompt A", "doc-001")
        assert result == '{"name": "Alice"}'

    def test_different_prompts_are_independent(self):
        self.sm.set_prediction("prompt A", "doc-001", "result A")
        self.sm.set_prediction("prompt B", "doc-001", "result B")
        assert self.sm.get_prediction("prompt A", "doc-001") == "result A"
        assert self.sm.get_prediction("prompt B", "doc-001") == "result B"

    def test_different_docs_are_independent(self):
        self.sm.set_prediction("same prompt", "doc-001", "result 1")
        self.sm.set_prediction("same prompt", "doc-002", "result 2")
        assert self.sm.get_prediction("same prompt", "doc-001") == "result 1"
        assert self.sm.get_prediction("same prompt", "doc-002") == "result 2"

    def test_overwrite_updates_value(self):
        self.sm.set_prediction("prompt", "doc-001", "old result")
        self.sm.set_prediction("prompt", "doc-001", "new result")
        assert self.sm.get_prediction("prompt", "doc-001") == "new result"

    def test_prediction_cache_stats_empty(self):
        stats = self.sm.get_prediction_cache_stats()
        assert stats["total_cached"] == 0

    def test_prediction_cache_stats_after_inserts(self):
        self.sm.set_prediction("p1", "doc-001", "r1")
        self.sm.set_prediction("p1", "doc-002", "r2")
        self.sm.set_prediction("p2", "doc-001", "r3")
        stats = self.sm.get_prediction_cache_stats()
        assert stats["total_cached"] == 3

    def test_prediction_cache_isolated_from_ocr_cache(self):
        """Prediction cache writes must not affect the OCR cache and vice versa."""
        self.sm.set_prediction("p", "doc-001", "pred")
        ocr_stats = self.sm.get_ocr_cache_stats()
        assert ocr_stats["total_cached"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# StateManager stall count recovery tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStallCountRecovery:
    """
    Verify get_stall_count() returns the correct number of consecutive
    rejections at the tail of the trajectory.

    This is used on resume to restore stall_count so the Mutator's escalation
    logic (train-example injection, beam secondary, radical rewrite) fires at
    the correct iteration after an interrupted run.
    """

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.sm = StateManager(db_path=self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_empty_trajectory_returns_zero(self):
        assert self.sm.get_stall_count() == 0

    def test_all_accepted_returns_zero(self):
        self.sm.log_iteration(0, "p0", 0.5, accepted=True)
        self.sm.log_iteration(1, "p1", 0.6, accepted=True)
        assert self.sm.get_stall_count() == 0

    def test_single_rejection_at_end(self):
        self.sm.log_iteration(0, "p0", 0.5, accepted=True)
        self.sm.log_iteration(1, "p1", 0.4, accepted=False)
        assert self.sm.get_stall_count() == 1

    def test_multiple_consecutive_rejections(self):
        self.sm.log_iteration(0, "p0", 0.5, accepted=True)
        self.sm.log_iteration(1, "p1", 0.4, accepted=False)
        self.sm.log_iteration(2, "p2", 0.4, accepted=False)
        self.sm.log_iteration(3, "p3", 0.4, accepted=False)
        assert self.sm.get_stall_count() == 3

    def test_acceptance_resets_stall_count(self):
        self.sm.log_iteration(0, "p0", 0.5, accepted=True)
        self.sm.log_iteration(1, "p1", 0.4, accepted=False)
        self.sm.log_iteration(2, "p2", 0.4, accepted=False)
        self.sm.log_iteration(3, "p3", 0.6, accepted=True)   # accepted!
        assert self.sm.get_stall_count() == 0

    def test_rejections_after_acceptance(self):
        self.sm.log_iteration(0, "p0", 0.5, accepted=True)
        self.sm.log_iteration(1, "p1", 0.4, accepted=False)
        self.sm.log_iteration(2, "p2", 0.6, accepted=True)   # accepted
        self.sm.log_iteration(3, "p3", 0.5, accepted=False)
        self.sm.log_iteration(4, "p4", 0.5, accepted=False)
        assert self.sm.get_stall_count() == 2   # only last 2 rejections count


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build a BaseAgent without a real network
# ─────────────────────────────────────────────────────────────────────────────

def _build_agent(tmp_db: str):
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
    from src.agents.base_agent import BaseAgent

    sm    = StateManager(db_path=tmp_db)
    agent = BaseAgent.__new__(BaseAgent)
    agent.model_name          = "test-model"
    agent.state_manager       = sm
    agent._github_token       = "gh-test-token"
    agent._github_model       = "gpt-4o-mini"
    agent._gemini_key         = "gemini-test-key"
    agent._gemini_text_model  = "gemini-2.5-flash"
    agent._ollama_base        = "http://localhost:11434"
    agent._ollama_model       = "llama3"
    agent._github_client      = None
    agent._ollama_client      = None
    agent._openrouter_client  = MagicMock()
    return agent


def _make_ok_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = text
    resp.usage.model_dump.return_value = {}
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# Four-tier fallback tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFourTierFallback:
    """Verify the OpenRouter → GitHub → Gemini → Ollama chain."""

    def test_openrouter_success_no_other_tier_called(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            agent = _build_agent(tmp.name)
            agent._openrouter_client.chat.completions.create.return_value = (
                _make_ok_response("OpenRouter OK")
            )
            mock_github  = MagicMock()
            agent._github_client = mock_github

            result = agent.call_llm("sys", "user", "Test")
            assert result == "OpenRouter OK"
            mock_github.chat.completions.create.assert_not_called()
        finally:
            os.unlink(tmp.name)

    def test_openrouter_daily_limit_falls_to_github(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            agent = _build_agent(tmp.name)
            agent._openrouter_client.chat.completions.create.side_effect = Exception(
                "429 free-models-per-day limit reached"
            )
            mock_github_client = MagicMock()
            mock_github_client.chat.completions.create.return_value = (
                _make_ok_response("GitHub Models OK")
            )
            agent._github_client = mock_github_client

            result = agent.call_llm("sys", "user", "Test")
            assert result == "GitHub Models OK"
        finally:
            os.unlink(tmp.name)

    def test_github_failure_falls_to_gemini(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            agent = _build_agent(tmp.name)
            agent._openrouter_client.chat.completions.create.side_effect = Exception(
                "429 free-models-per-day"
            )
            mock_github = MagicMock()
            mock_github.chat.completions.create.side_effect = Exception(
                "GitHub Models quota exceeded"
            )
            agent._github_client = mock_github

            with patch("src.agents.base_agent.BaseAgent._call_gemini_text",
                       return_value="Gemini text OK") as mock_gemini_call:
                result = agent.call_llm("sys", "user", "Test")
                assert result == "Gemini text OK"
                mock_gemini_call.assert_called_once()
        finally:
            os.unlink(tmp.name)

    def test_gemini_failure_falls_to_ollama(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            agent = _build_agent(tmp.name)
            agent._openrouter_client.chat.completions.create.side_effect = Exception(
                "429 free-models-per-day"
            )
            mock_github = MagicMock()
            mock_github.chat.completions.create.side_effect = Exception("GitHub quota")
            agent._github_client = mock_github

            mock_ollama_client = MagicMock()
            mock_ollama_client.chat.completions.create.return_value = (
                _make_ok_response("Ollama OK")
            )
            agent._ollama_client = mock_ollama_client

            from src.agents.base_agent import _TierExhausted
            with patch.object(agent, "_call_gemini_text",
                               side_effect=_TierExhausted("gemini exhausted")):
                result = agent.call_llm("sys", "user", "Test")
            mock_ollama_client.chat.completions.create.assert_called_once()
        finally:
            os.unlink(tmp.name)

    def test_all_tiers_exhausted_raises_daily_limit_error(self):
        from src.agents.base_agent import DailyLimitError, _TierExhausted
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            agent = _build_agent(tmp.name)
            agent._openrouter_client.chat.completions.create.side_effect = Exception(
                "429 free-models-per-day"
            )
            mock_github = MagicMock()
            mock_github.chat.completions.create.side_effect = Exception("GitHub quota")
            agent._github_client = mock_github

            mock_ollama = MagicMock()
            mock_ollama.chat.completions.create.side_effect = Exception("connection refused")
            agent._ollama_client = mock_ollama

            with patch.object(agent, "_call_gemini_text",
                               side_effect=_TierExhausted("gemini exhausted")):
                with pytest.raises(DailyLimitError):
                    agent.call_llm("sys", "user", "Test")
        finally:
            os.unlink(tmp.name)

    def test_openrouter_rate_limit_retries_then_succeeds(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            agent = _build_agent(tmp.name)
            agent._openrouter_client.chat.completions.create.side_effect = [
                Exception("429 Too Many Requests"),
                Exception("429 Too Many Requests"),
                _make_ok_response("Retry success"),
            ]
            with patch("time.sleep"):
                result = agent.call_llm("sys", "user", "Test")
            assert result == "Retry success"
        finally:
            os.unlink(tmp.name)

    def test_missing_github_token_skips_to_gemini(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            agent = _build_agent(tmp.name)
            agent._github_token = ""
            agent._openrouter_client.chat.completions.create.side_effect = Exception(
                "429 free-models-per-day"
            )
            with patch.object(agent, "_call_gemini_text",
                               return_value="Gemini fallback OK"):
                result = agent.call_llm("sys", "user", "Test")
            assert result == "Gemini fallback OK"
        finally:
            os.unlink(tmp.name)

    def test_missing_gemini_key_skips_to_ollama(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            agent = _build_agent(tmp.name)
            agent._gemini_key   = ""
            agent._github_token = ""
            agent._openrouter_client.chat.completions.create.side_effect = Exception(
                "429 free-models-per-day"
            )
            mock_ollama = MagicMock()
            mock_ollama.chat.completions.create.return_value = _make_ok_response("Ollama OK")
            agent._ollama_client = mock_ollama

            result = agent.call_llm("sys", "user", "Test")
            assert result == "Ollama OK"
        finally:
            os.unlink(tmp.name)


# ─────────────────────────────────────────────────────────────────────────────
# Loader OCR cache integration
# ─────────────────────────────────────────────────────────────────────────────

class TestLoaderCacheIntegration:
    """Verify ExtractBenchLoader._extract_with_cache uses the StateManager cache."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.sm = StateManager(db_path=self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_cache_hit_skips_extraction(self):
        from src.data.loader import ExtractBenchLoader

        loader = ExtractBenchLoader(
            base_path="/fake",
            schema_name="fake/schema",
            state_manager=self.sm,
        )
        pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf.write(b"%PDF-1.4 cached document")
        pdf.close()

        try:
            self.sm.set_ocr_cache(pdf.name, "Pre-cached text", method="pymupdf")
            with patch("src.data.loader.extract_text_from_pdf") as mock_extract:
                result = loader._extract_with_cache(pdf.name)
            assert result == "Pre-cached text"
            mock_extract.assert_not_called()
        finally:
            os.unlink(pdf.name)

    def test_cache_miss_calls_extraction_and_stores(self):
        from src.data.loader import ExtractBenchLoader

        loader = ExtractBenchLoader(
            base_path="/fake",
            schema_name="fake/schema",
            state_manager=self.sm,
        )
        pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf.write(b"%PDF-1.4 fresh document")
        pdf.close()

        try:
            with patch(
                "src.data.loader.extract_text_from_pdf",
                return_value=("Freshly extracted text", "pymupdf"),
            ):
                result = loader._extract_with_cache(pdf.name)

            assert result == "Freshly extracted text"
            assert self.sm.get_ocr_cache(pdf.name) == "Freshly extracted text"
        finally:
            os.unlink(pdf.name)