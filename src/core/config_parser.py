"""
Configuration loader and validator.

All optimizer behaviour is driven by a YAML config file. Changing the
dataset, models, budget, seed prompt, vision model, or fallback models
requires only config file edits — no code changes.

Fallback model priority (per agent):
  1. OpenRouter  → `models.extractor / critic / mutator`
  2. GitHub Mdls → `models.github_model`
  3. Gemini text → GEMINI_API_KEY + GEMINI_TEXT_MODEL env vars
  4. Ollama      → OLLAMA_BASE_URL / OLLAMA_MODEL env vars
"""

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name:        str
    base_path:   str
    split_seed:  int   = 42
    train_ratio: float = Field(0.5,  ge=0.01, le=0.95)
    # val_ratio cap raised from 0.5 → 0.7 to allow larger val sets on small
    # datasets (e.g. 3 val docs out of 7 requires val_ratio ≈ 0.57).
    val_ratio:   float = Field(0.2,  ge=0.01, le=0.70)


class BudgetConfig(BaseModel):
    max_iterations:   int   = Field(20, ge=1)
    max_cost_dollars: float = Field(
        0.0,
        description="Max spend in USD. 0 = unlimited (free tier).",
    )


class ModelsConfig(BaseModel):
    extractor:    str
    critic:       str
    mutator:      str
    # Tier-2 fallback via the GitHub Models Azure AI inference endpoint.
    # Free for GitHub Student Developer Pack holders (GITHUB_TOKEN env var).
    github_model: str = "gpt-4o-mini"


class OptimizerConfig(BaseModel):
    dataset:      DatasetConfig
    budget:       BudgetConfig
    models:       ModelsConfig
    seed_prompt:  str
    # Vision/OCR model used by the PDF loader (Gemini path in loader.py).
    # Separate from the Gemini text-generation fallback in base_agent.py.
    vision_model: str = "google/gemini-2.0-flash-exp:free"


def load_config(yaml_path: str) -> OptimizerConfig:
    """Load and validate a YAML configuration file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return OptimizerConfig(**raw)