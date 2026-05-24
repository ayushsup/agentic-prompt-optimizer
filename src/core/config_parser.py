"""
Configuration loader and validator.

All optimizer behaviour is driven by a YAML config file. Changing the dataset,
models, budget, seed prompt, or vision model requires only config file edits.
"""

import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    name: str
    base_path: str
    split_seed: int = 42
    train_ratio: float = Field(0.5, ge=0.01, le=0.95)
    val_ratio:   float = Field(0.2, ge=0.01, le=0.5)


class BudgetConfig(BaseModel):
    max_iterations:   int   = Field(20, ge=1)
    max_cost_dollars: float = Field(
        0.0,
        description="Max spend in USD. 0 = unlimited (free tier).",
    )


class ModelsConfig(BaseModel):
    extractor: str
    critic:    str
    mutator:   str


class OptimizerConfig(BaseModel):
    dataset:      DatasetConfig
    budget:       BudgetConfig
    models:       ModelsConfig
    seed_prompt:  str
    vision_model: str = "google/gemini-2.0-flash-exp:free"


def load_config(yaml_path: str) -> OptimizerConfig:
    """Load and validate a YAML configuration file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return OptimizerConfig(**raw)