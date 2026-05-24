"""
Base LLM agent with resilient retry logic, Ollama fallback, latency tracking,
and state logging.

All agents inherit from BaseAgent.  The primary backend is OpenRouter's
OpenAI-compatible API.  When OpenRouter is unavailable, rate-limited
(daily quota exhausted), or returns repeated errors, the agent automatically
falls back to a locally-running Ollama instance.

NOTE: Gemini remains the exclusive path for vision/OCR (PDF extraction).
      This fallback only applies to the text-only agents: Extractor, Critic,
      and Mutator.

Fallback ladder
---------------
1. OpenRouter (primary)
   - 429 per-minute rate-limit → exponential backoff, retry up to MAX_RETRIES
   - 429 daily quota           → immediately skip to Ollama fallback
   - 5xx server error          → fixed pause, retry up to MAX_RETRIES
2. Ollama (fallback, http://localhost:11434 by default)
   - Used when OpenRouter daily quota is exhausted OR when every OpenRouter
     retry fails with a non-recoverable error.
   - Model: configured via OLLAMA_MODEL env var (default: "llama3.2").
   - If Ollama is also unreachable, DailyLimitError is raised so the loop
     can shut down cleanly and persist state for tomorrow's resume.

Environment variables
---------------------
  OPENROUTER_API_KEY   (required for OpenRouter path)
  OLLAMA_BASE_URL      (optional, default: http://localhost:11434)
  OLLAMA_MODEL         (optional, default: llama3.2)
"""

import os
import time
from typing import Optional

from openai import OpenAI

from src.core.state_manager import StateManager


class DailyLimitError(Exception):
    """Raised when both OpenRouter quota and Ollama fallback are exhausted."""


class BaseAgent:
    """
    Shared LLM client used by Extractor, Critic, and Mutator.

    Retry strategy (OpenRouter):
      - daily limit (429 + "free-models-per-day") : skip to Ollama immediately
      - 429 (rate limit)   : exponential backoff starting at 10 s
      - 5xx (server error) : fixed 15 s pause
      - other exceptions   : re-raised immediately

    Ollama fallback:
      - Invoked when the OpenRouter path is exhausted or quota-blocked.
      - Uses the openai SDK pointed at the local Ollama /v1 endpoint.
      - If Ollama is also unavailable, DailyLimitError propagates upward.
    """

    MAX_RETRIES      = 5
    BASE_DELAY_429   = 10   # seconds; doubles each attempt
    DELAY_5XX        = 15   # seconds; fixed
    INTER_CALL_PAUSE = 3    # seconds; polite pause between successful calls

    def __init__(self, model_name: str, state_manager: StateManager):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Export it before running: export OPENROUTER_API_KEY=your_key"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_name    = model_name
        self.state_manager = state_manager

        # Ollama fallback client — lazily initialised on first use
        self._ollama_client: Optional[OpenAI] = None
        self._ollama_model  = os.environ.get("OLLAMA_MODEL", "gemma:2b")
        self._ollama_base   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    # ------------------------------------------------------------------
    # Ollama client (lazy init)
    # ------------------------------------------------------------------

    def _get_ollama_client(self) -> OpenAI:
        if self._ollama_client is None:
            self._ollama_client = OpenAI(
                api_key="ollama",   # Ollama ignores the key but the SDK requires one
                base_url=f"{self._ollama_base}/v1",
            )
        return self._ollama_client

    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        role_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Call the local Ollama instance.  Raises DailyLimitError if unreachable.
        """
        print(
            f"  🔄 [{role_name}] Falling back to Ollama "
            f"(model={self._ollama_model}, base={self._ollama_base})…"
        )
        try:
            client = self._get_ollama_client()
            t_start = time.monotonic()
            response = client.chat.completions.create(
                model=self._ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = (time.monotonic() - t_start) * 1000
            content = response.choices[0].message.content or ""

            self.state_manager.log_llm_call(
                role=f"{role_name}[ollama]",
                prompt=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
                response=content,
                model=f"ollama/{self._ollama_model}",
                usage=response.usage.model_dump() if response.usage else {},
                cost=0.0,
                latency_ms=latency_ms,
            )
            print(f"  ✅ [{role_name}] Ollama responded ({latency_ms:.0f} ms).")
            time.sleep(self.INTER_CALL_PAUSE)
            return content

        except Exception as exc:
            err = str(exc)
            # Connection refused / name resolution failure → Ollama not running
            if any(k in err.lower() for k in ("connection refused", "name or service not known",
                                               "connect call failed", "failed to establish")):
                print(
                    f"\n❌ [{role_name}] Ollama is not reachable at {self._ollama_base}.\n"
                    "   Start Ollama with: ollama serve\n"
                    "   Then pull a model: ollama pull llama3.2\n"
                    "   State is safely persisted — re-run when ready."
                )
                raise DailyLimitError(
                    f"OpenRouter quota exhausted and Ollama unavailable: {exc}"
                ) from exc
            # Any other Ollama error — log and re-raise as DailyLimitError so the
            # loop can persist state rather than crash.
            print(f"  ❌ [{role_name}] Ollama error: {err[:200]}")
            raise DailyLimitError(f"Ollama fallback failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Primary call method
    # ------------------------------------------------------------------

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        role_name: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        """
        Call the LLM with retry/backoff on OpenRouter, then fall back to Ollama.

        Parameters
        ----------
        system_prompt : Instruction context for the model.
        user_prompt   : User-facing input (document text, critique, etc.)
        role_name     : Label used in logs (e.g. "Extractor", "Critic").
        temperature   : Sampling temperature; lower = more deterministic.
        max_tokens    : Hard cap on response length.

        Raises
        ------
        DailyLimitError  : When both OpenRouter quota AND Ollama are exhausted.
        Exception        : After MAX_RETRIES failed attempts (non-quota errors).
        """
        last_exception = None

        for attempt in range(self.MAX_RETRIES):
            try:
                t_start = time.monotonic()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency_ms = (time.monotonic() - t_start) * 1000
                content = response.choices[0].message.content or ""

                self.state_manager.log_llm_call(
                    role=role_name,
                    prompt=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
                    response=content,
                    model=self.model_name,
                    usage=response.usage.model_dump() if response.usage else {},
                    cost=0.0,
                    latency_ms=latency_ms,
                )
                time.sleep(self.INTER_CALL_PAUSE)
                return content

            except Exception as exc:
                last_exception = exc
                error_msg = str(exc)

                if "429" in error_msg:
                    # --- Daily quota exhausted → fall back to Ollama immediately ---
                    if "free-models-per-day" in error_msg or "per-day" in error_msg:
                        print(
                            f"\n🚫 [{role_name}] OpenRouter daily free-model quota exhausted. "
                            "Trying Ollama fallback…"
                        )
                        return self._call_ollama(
                            system_prompt, user_prompt, role_name, temperature, max_tokens
                        )

                    if attempt >= self.MAX_RETRIES - 1:
                        print(
                            f"❌ [{role_name}] Max retries ({self.MAX_RETRIES}) reached "
                            "on OpenRouter. Trying Ollama fallback…"
                        )
                        return self._call_ollama(
                            system_prompt, user_prompt, role_name, temperature, max_tokens
                        )

                    delay = self.BASE_DELAY_429 * (2 ** attempt)
                    print(
                        f"⚠️  [{role_name}] Rate-limited (429). Retrying in {delay}s "
                        f"(attempt {attempt + 1}/{self.MAX_RETRIES})…"
                    )
                    time.sleep(delay)

                elif any(code in error_msg for code in ("500", "502", "503")):
                    if attempt >= self.MAX_RETRIES - 1:
                        print(
                            f"❌ [{role_name}] Max retries ({self.MAX_RETRIES}) reached "
                            "on OpenRouter (server error). Trying Ollama fallback…"
                        )
                        return self._call_ollama(
                            system_prompt, user_prompt, role_name, temperature, max_tokens
                        )

                    print(
                        f"⚠️  [{role_name}] Server error. Retrying in {self.DELAY_5XX}s "
                        f"(attempt {attempt + 1}/{self.MAX_RETRIES})…"
                    )
                    time.sleep(self.DELAY_5XX)

                else:
                    # Non-retryable OpenRouter error — try Ollama once
                    print(
                        f"⚠️  [{role_name}] Non-retryable OpenRouter error: "
                        f"{error_msg[:120]}. Trying Ollama fallback…"
                    )
                    return self._call_ollama(
                        system_prompt, user_prompt, role_name, temperature, max_tokens
                    )

        # All retries exhausted — last attempt via Ollama
        print(
            f"❌ [{role_name}] All {self.MAX_RETRIES} OpenRouter attempts failed. "
            "Trying Ollama fallback…"
        )
        return self._call_ollama(
            system_prompt, user_prompt, role_name, temperature, max_tokens
        )