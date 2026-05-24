"""
Tests for OCR caching (StateManager) and Ollama fallback (BaseAgent).

Run with: pytest tests/test_infrastructure.py -v
"""

import hashlib
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.core.state_manager import StateManager


# ---------------------------------------------------------------------------
# StateManager OCR cache tests
# ---------------------------------------------------------------------------

class TestOCRCache:
    """Verify the SQLite OCR cache stores, retrieves, and invalidates correctly."""

    def setup_method(self):
        # Each test gets its own fresh in-memory-like DB via a temp file
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.sm = StateManager(db_path=self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def _write_tmp_pdf(self, content: bytes) -> str:
        """Write fake 'PDF' bytes to a temp file and return its path."""
        f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_cache_miss_returns_none(self):
        pdf = self._write_tmp_pdf(b"%PDF-1.4 fake content")
        try:
            result = self.sm.get_ocr_cache(pdf)
            assert result is None
        finally:
            os.unlink(pdf)

    def test_set_and_get_roundtrip(self):
        pdf = self._write_tmp_pdf(b"%PDF-1.4 fake content A")
        try:
            self.sm.set_ocr_cache(pdf, "Extracted text here", method="pymupdf")
            result = self.sm.get_ocr_cache(pdf)
            assert result == "Extracted text here"
        finally:
            os.unlink(pdf)

    def test_cache_invalidated_on_file_change(self):
        """Replacing file bytes produces a new hash → cache miss."""
        pdf = self._write_tmp_pdf(b"%PDF-1.4 version one")
        try:
            self.sm.set_ocr_cache(pdf, "Old text", method="pymupdf")
            # Overwrite with different content
            with open(pdf, "wb") as f:
                f.write(b"%PDF-1.4 version TWO -- completely different")
            result = self.sm.get_ocr_cache(pdf)
            assert result is None, "Cache should miss after file content changes"
        finally:
            os.unlink(pdf)

    def test_set_overwrites_existing_entry(self):
        pdf = self._write_tmp_pdf(b"%PDF-1.4 stable bytes")
        try:
            self.sm.set_ocr_cache(pdf, "First extraction", method="pymupdf")
            self.sm.set_ocr_cache(pdf, "Updated extraction", method="pdfplumber")
            result = self.sm.get_ocr_cache(pdf)
            assert result == "Updated extraction"
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
        result = self.sm.get_ocr_cache("/nonexistent/path/file.pdf")
        assert result is None

    def test_cache_does_not_affect_metric_cache(self):
        """OCR cache and metric cache are independent tables."""
        pdf = self._write_tmp_pdf(b"%PDF-1.4 content")
        try:
            self.sm.set_ocr_cache(pdf, "text", method="pymupdf")
            # metric cache should still be empty
            assert self.sm.get_metric_cache("some_key") is None
        finally:
            os.unlink(pdf)


# ---------------------------------------------------------------------------
# BaseAgent Ollama fallback tests
# ---------------------------------------------------------------------------

class TestOllamaFallback:
    """
    Verify the Ollama fallback logic in BaseAgent without making real HTTP calls.
    """

    def _make_agent(self, tmp_db: str):
        """Build a BaseAgent with a real StateManager but mocked OpenRouter client."""
        # Must have OPENROUTER_API_KEY set (value doesn't matter for mocking)
        os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
        from src.agents.base_agent import BaseAgent
        sm = StateManager(db_path=tmp_db)
        agent = BaseAgent.__new__(BaseAgent)
        agent.model_name    = "test-model"
        agent.state_manager = sm
        agent._ollama_client = None
        agent._ollama_model  = os.environ.get("OLLAMA_MODEL", "gemma:2b")
        agent._ollama_base   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        # Mock the OpenRouter client
        agent.client = MagicMock()
        return agent

    def test_ollama_called_on_daily_limit(self):
        """When OpenRouter returns daily-limit 429, Ollama is invoked."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            agent = self._make_agent(tmp.name)

            # OpenRouter raises daily-limit error
            agent.client.chat.completions.create.side_effect = Exception(
                "429 Too Many Requests — free-models-per-day limit reached"
            )

            # Ollama returns successfully
            mock_ollama_response = MagicMock()
            mock_ollama_response.choices[0].message.content = "Ollama response text"
            mock_ollama_response.usage = None

            mock_ollama_client = MagicMock()
            mock_ollama_client.chat.completions.create.return_value = mock_ollama_response
            agent._ollama_client = mock_ollama_client

            result = agent.call_llm("sys", "user", "TestRole")
            assert result == "Ollama response text"
            mock_ollama_client.chat.completions.create.assert_called_once()
        finally:
            os.unlink(tmp.name)

    def test_daily_limit_error_raised_when_ollama_unreachable(self):
        """When both OpenRouter quota and Ollama are unavailable, DailyLimitError raised."""
        from src.agents.base_agent import DailyLimitError
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            agent = self._make_agent(tmp.name)

            agent.client.chat.completions.create.side_effect = Exception(
                "429 free-models-per-day"
            )

            mock_ollama_client = MagicMock()
            mock_ollama_client.chat.completions.create.side_effect = Exception(
                "connection refused"
            )
            agent._ollama_client = mock_ollama_client

            with pytest.raises(DailyLimitError):
                agent.call_llm("sys", "user", "TestRole")
        finally:
            os.unlink(tmp.name)

    def test_openrouter_success_does_not_call_ollama(self):
        """Happy path: OpenRouter works, Ollama is never touched."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            agent = self._make_agent(tmp.name)

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "OpenRouter response"
            mock_response.usage.model_dump.return_value = {}
            agent.client.chat.completions.create.return_value = mock_response

            mock_ollama = MagicMock()
            agent._ollama_client = mock_ollama

            result = agent.call_llm("sys", "user", "TestRole")
            assert result == "OpenRouter response"
            mock_ollama.chat.completions.create.assert_not_called()
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Loader OCR cache integration (lightweight, no real PDFs)
# ---------------------------------------------------------------------------

class TestLoaderCacheIntegration:
    """
    Verify ExtractBenchLoader._extract_with_cache uses the StateManager cache.
    """

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.sm = StateManager(db_path=self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_cache_hit_skips_extraction(self):
        """If text is cached, extract_text_from_pdf should never be called."""
        from src.data.loader import ExtractBenchLoader

        loader = ExtractBenchLoader(
            base_path="/fake",
            schema_name="fake/schema",
            state_manager=self.sm,
        )

        # Write a real temp PDF so _hash_file works
        pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf.write(b"%PDF-1.4 cached document")
        pdf.close()

        try:
            # Pre-populate cache
            self.sm.set_ocr_cache(pdf.name, "Pre-cached text", method="pymupdf")

            with patch("src.data.loader.extract_text_from_pdf") as mock_extract:
                result = loader._extract_with_cache(pdf.name)

            assert result == "Pre-cached text"
            mock_extract.assert_not_called()
        finally:
            os.unlink(pdf.name)

    def test_cache_miss_calls_extraction_and_stores(self):
        """On a cache miss, extract_text_from_pdf is called and result is stored."""
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

            # Should now be in cache
            cached = self.sm.get_ocr_cache(pdf.name)
            assert cached == "Freshly extracted text"
        finally:
            os.unlink(pdf.name)