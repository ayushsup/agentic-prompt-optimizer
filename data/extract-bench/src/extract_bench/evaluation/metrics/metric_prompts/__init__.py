"""Prompt templates for LLM-judge metrics."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def _load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text().strip()


LLM_JUDGE_PROMPT = _load_prompt("llm_judge.txt")
ARRAY_LLM_PROMPT = _load_prompt("array_llm.txt")
STRING_SEMANTIC_PROMPT = _load_prompt("string_semantic.txt")

__all__ = ["LLM_JUDGE_PROMPT", "ARRAY_LLM_PROMPT", "STRING_SEMANTIC_PROMPT"]
