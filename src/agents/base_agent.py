"""
Base LLM agent with a four-tier resilient fallback chain.

Fallback ladder (tried in order)
─────────────────────────────────
1. OpenRouter          (primary — OPENROUTER_API_KEY)
   - 429 per-minute   → exponential back-off, retry up to MAX_RETRIES
   - 429 daily quota  → immediately skip to tier 2
   - 5xx server error → fixed pause, retry up to MAX_RETRIES
   - other error      → immediately skip to tier 2

2. GitHub Models       (GITHUB_TOKEN, Azure AI inference endpoint)
   - OpenAI-compatible, free via GitHub Student Developer Pack
   - Model configured via `github_model` in config / GITHUB_MODEL env var
   - On failure → skip to tier 3

3. Gemini              (GEMINI_API_KEY, text-generation path)
   - Uses gemini-2.0-flash by default (GEMINI_TEXT_MODEL env var to override)
   - Separate from the OCR/vision use of Gemini in the loader
   - On failure → skip to tier 4

4. Ollama              (local, OLLAMA_BASE_URL / OLLAMA_MODEL)
   - http://localhost:11434 by default
   - If also unreachable → raises DailyLimitError so the loop can persist
     state and exit cleanly

Environment variables
─────────────────────
  OPENROUTER_API_KEY   (required for tier 1)
  GITHUB_TOKEN         (required for tier 2)
  GITHUB_MODEL         (optional; default: gpt-4o-mini)
  GEMINI_API_KEY       (required for tier 3 — same key used for vision OCR)
  GEMINI_TEXT_MODEL    (optional; default: gemini-2.0-flash)
  OLLAMA_BASE_URL      (optional; default: http://localhost:11434)
  OLLAMA_MODEL         (optional; default: llama3)
"""

from __future__ import annotations

import os
import time
from typing import Optional

from openai import OpenAI

from src.core.state_manager import StateManager


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class DailyLimitError(Exception):
    """Raised when the entire fallback chain is exhausted."""


class _TierExhausted(Exception):
    """Internal signal: current tier failed, try the next one."""


# ─────────────────────────────────────────────────────────────────────────────
# Tier constants
# ─────────────────────────────────────────────────────────────────────────────

_GITHUB_BASE_URL   = "https://models.inference.ai.azure.com"
_DEFAULT_GITHUB_MODEL      = "gpt-4o-mini"
_DEFAULT_GEMINI_TEXT_MODEL = "gemini-2.0-flash"
_DEFAULT_OLLAMA_BASE       = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL      = "llama3"


# ─────────────────────────────────────────────────────────────────────────────
# BaseAgent
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    """
    Shared LLM client used by Extractor, Critic, and Mutator.

    All four tiers use the same (system_prompt, user_prompt) interface so
    every agent subclass gets resilient multi-tier fallback for free.
    """

    MAX_RETRIES      = 5
    BASE_DELAY_429   = 10   # seconds; doubles each attempt
    DELAY_5XX        = 15   # seconds; fixed
    INTER_CALL_PAUSE = 3    # seconds; polite pause after each successful call

    def __init__(self, model_name: str, state_manager: StateManager,
                 github_model: Optional[str] = None):
        self.model_name    = model_name
        self.state_manager = state_manager

        # ── Tier 1: OpenRouter ───────────────────────────────────────
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. "
                "Export it before running: export OPENROUTER_API_KEY=your_key"
            )
        self._openrouter_client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )

        # ── Tier 2: GitHub Models ────────────────────────────────────
        self._github_token = os.environ.get("GITHUB_TOKEN", "")
        self._github_model = (
            github_model
            or os.environ.get("GITHUB_MODEL", _DEFAULT_GITHUB_MODEL)
        )
        self._github_client: Optional[OpenAI] = None   # lazy init

        # ── Tier 3: Gemini (text) ────────────────────────────────────
        self._gemini_key        = os.environ.get("GEMINI_API_KEY", "")
        self._gemini_text_model = os.environ.get("GEMINI_TEXT_MODEL",
                                                  _DEFAULT_GEMINI_TEXT_MODEL)

        # ── Tier 4: Ollama ───────────────────────────────────────────
        self._ollama_base  = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE)
        self._ollama_model = os.environ.get("OLLAMA_MODEL",    _DEFAULT_OLLAMA_MODEL)
        self._ollama_client: Optional[OpenAI] = None   # lazy init

    # ─────────────────────────────────────────────────────────────────
    # Lazy client constructors
    # ─────────────────────────────────────────────────────────────────

    def _get_github_client(self) -> OpenAI:
        if self._github_client is None:
            if not self._github_token:
                raise _TierExhausted("GITHUB_TOKEN not set — skipping GitHub Models tier.")
            self._github_client = OpenAI(
                api_key=self._github_token,
                base_url=_GITHUB_BASE_URL,
            )
        return self._github_client

    def _get_ollama_client(self) -> OpenAI:
        if self._ollama_client is None:
            self._ollama_client = OpenAI(
                api_key="ollama",
                base_url=f"{self._ollama_base}/v1",
            )
        return self._ollama_client

    # ─────────────────────────────────────────────────────────────────
    # Per-tier call helpers
    # ─────────────────────────────────────────────────────────────────

    def _call_openrouter(
        self,
        system_prompt: str,
        user_prompt: str,
        role_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Call OpenRouter with exponential back-off on 429s.
        Raises _TierExhausted when the daily quota is hit or all retries fail.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self.MAX_RETRIES):
            try:
                t0 = time.monotonic()
                resp = self._openrouter_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency_ms = (time.monotonic() - t0) * 1000
                content = resp.choices[0].message.content or ""

                self.state_manager.log_llm_call(
                    role=role_name,
                    prompt=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
                    response=content,
                    model=self.model_name,
                    usage=resp.usage.model_dump() if resp.usage else {},
                    cost=0.0,
                    latency_ms=latency_ms,
                )
                time.sleep(self.INTER_CALL_PAUSE)
                return content

            except Exception as exc:
                last_exc = exc
                msg = str(exc)

                # Daily quota → skip immediately
                if "429" in msg and any(k in msg for k in
                        ("free-models-per-day", "per-day", "daily")):
                    print(f"\n🚫 [{role_name}] OpenRouter daily quota exhausted.")
                    raise _TierExhausted(f"OpenRouter daily limit: {exc}") from exc

                # Per-minute 429 → back-off
                if "429" in msg:
                    if attempt >= self.MAX_RETRIES - 1:
                        raise _TierExhausted(f"OpenRouter max retries (429): {exc}") from exc
                    delay = self.BASE_DELAY_429 * (2 ** attempt)
                    print(f"⚠️  [{role_name}] Rate-limited. Retrying in {delay}s "
                          f"(attempt {attempt + 1}/{self.MAX_RETRIES})…")
                    time.sleep(delay)
                    continue

                # Server error → fixed pause
                if any(code in msg for code in ("500", "502", "503")):
                    if attempt >= self.MAX_RETRIES - 1:
                        raise _TierExhausted(f"OpenRouter server error after max retries: {exc}") from exc
                    print(f"⚠️  [{role_name}] Server error. Retrying in {self.DELAY_5XX}s…")
                    time.sleep(self.DELAY_5XX)
                    continue

                # Any other error → skip tier immediately
                print(f"⚠️  [{role_name}] OpenRouter error: {msg[:120]}")
                raise _TierExhausted(f"OpenRouter non-retryable: {exc}") from exc

        raise _TierExhausted(
            f"OpenRouter exhausted after {self.MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    def _call_github_models(
        self,
        system_prompt: str,
        user_prompt: str,
        role_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Tier 2: GitHub Models via Azure AI inference endpoint.
        Free for GitHub Student Developer Pack holders.
        Raises _TierExhausted on any failure.
        """
        print(f"  🔄 [{role_name}] Trying GitHub Models "
              f"(model={self._github_model})…")
        try:
            client = self._get_github_client()
            t0 = time.monotonic()
            resp = client.chat.completions.create(
                model=self._github_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            content = resp.choices[0].message.content or ""

            self.state_manager.log_llm_call(
                role=f"{role_name}[github]",
                prompt=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
                response=content,
                model=f"github/{self._github_model}",
                usage=resp.usage.model_dump() if resp.usage else {},
                cost=0.0,
                latency_ms=latency_ms,
            )
            print(f"  ✅ [{role_name}] GitHub Models responded ({latency_ms:.0f} ms).")
            time.sleep(self.INTER_CALL_PAUSE)
            return content

        except _TierExhausted:
            raise
        except Exception as exc:
            msg = str(exc)
            # Rate-limit / quota on GitHub Models
            if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                print(f"  🚫 [{role_name}] GitHub Models quota/rate-limit: {msg[:120]}")
            else:
                print(f"  ❌ [{role_name}] GitHub Models error: {msg[:120]}")
            raise _TierExhausted(f"GitHub Models failed: {exc}") from exc

    def _call_gemini_text(
        self,
        system_prompt: str,
        user_prompt: str,
        role_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Tier 3: Gemini text-generation (separate from the OCR/vision path).
        Uses the google-genai SDK with GEMINI_API_KEY.
        Raises _TierExhausted on any failure.
        """
        print(f"  🔄 [{role_name}] Trying Gemini text "
              f"(model={self._gemini_text_model})…")
        if not self._gemini_key:
            raise _TierExhausted("GEMINI_API_KEY not set — skipping Gemini text tier.")

        try:
            from google import genai as google_genai
            from google.genai import types as genai_types

            client = google_genai.Client(api_key=self._gemini_key)

            # Combine system + user into a single prompt for Gemini
            combined = f"{system_prompt}\n\n{user_prompt}"

            t0 = time.monotonic()
            response = client.models.generate_content(
                model=self._gemini_text_model,
                contents=combined,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            latency_ms = (time.monotonic() - t0) * 1000
            content = (response.text or "").strip()

            self.state_manager.log_llm_call(
                role=f"{role_name}[gemini]",
                prompt=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
                response=content,
                model=f"gemini/{self._gemini_text_model}",
                usage={},
                cost=0.0,
                latency_ms=latency_ms,
            )
            print(f"  ✅ [{role_name}] Gemini text responded ({latency_ms:.0f} ms).")
            time.sleep(self.INTER_CALL_PAUSE)
            return content

        except ImportError:
            raise _TierExhausted(
                "google-genai not installed. Run: pip install google-genai"
            )
        except Exception as exc:
            msg = str(exc)
            if any(k in msg.lower() for k in ("quota", "429", "resource exhausted",
                                               "rate", "limit")):
                print(f"  🚫 [{role_name}] Gemini quota/rate-limit: {msg[:120]}")
            else:
                print(f"  ❌ [{role_name}] Gemini text error: {msg[:120]}")
            raise _TierExhausted(f"Gemini text failed: {exc}") from exc

    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        role_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Tier 4: Local Ollama instance.
        Raises DailyLimitError (terminal) if Ollama is also unreachable.
        """
        print(f"  🔄 [{role_name}] Falling back to Ollama "
              f"(model={self._ollama_model}, base={self._ollama_base})…")
        try:
            client = self._get_ollama_client()
            t0 = time.monotonic()
            resp = client.chat.completions.create(
                model=self._ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            content = resp.choices[0].message.content or ""

            self.state_manager.log_llm_call(
                role=f"{role_name}[ollama]",
                prompt=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
                response=content,
                model=f"ollama/{self._ollama_model}",
                usage=resp.usage.model_dump() if resp.usage else {},
                cost=0.0,
                latency_ms=latency_ms,
            )
            print(f"  ✅ [{role_name}] Ollama responded ({latency_ms:.0f} ms).")
            time.sleep(self.INTER_CALL_PAUSE)
            return content

        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("connection refused", "name or service not known",
                                      "connect call failed", "failed to establish")):
                print(
                    f"\n❌ [{role_name}] Ollama is not reachable at {self._ollama_base}.\n"
                    "   Start Ollama: ollama serve\n"
                    "   Pull a model: ollama pull llama3\n"
                    "   State is safely persisted — re-run when ready."
                )
            else:
                print(f"  ❌ [{role_name}] Ollama error: {str(exc)[:200]}")
            raise DailyLimitError(
                "Entire fallback chain exhausted "
                "(OpenRouter → GitHub Models → Gemini → Ollama). "
                f"Last error: {exc}"
            ) from exc

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        role_name: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        """
        Call the LLM with automatic four-tier fallback.

        Tier order: OpenRouter → GitHub Models → Gemini → Ollama

        Raises
        ------
        DailyLimitError  : When all four tiers are exhausted.
        """
        # ── Tier 1: OpenRouter ───────────────────────────────────────
        try:
            return self._call_openrouter(
                system_prompt, user_prompt, role_name, temperature, max_tokens
            )
        except _TierExhausted as exc:
            print(f"  ⚡ [{role_name}] OpenRouter tier exhausted → trying GitHub Models…")

        # ── Tier 2: GitHub Models ────────────────────────────────────
        try:
            return self._call_github_models(
                system_prompt, user_prompt, role_name, temperature, max_tokens
            )
        except _TierExhausted:
            print(f"  ⚡ [{role_name}] GitHub Models tier exhausted → trying Gemini text…")

        # ── Tier 3: Gemini text ──────────────────────────────────────
        try:
            return self._call_gemini_text(
                system_prompt, user_prompt, role_name, temperature, max_tokens
            )
        except _TierExhausted:
            print(f"  ⚡ [{role_name}] Gemini text tier exhausted → trying Ollama…")

        # ── Tier 4: Ollama (terminal) ────────────────────────────────
        return self._call_ollama(
            system_prompt, user_prompt, role_name, temperature, max_tokens
        )