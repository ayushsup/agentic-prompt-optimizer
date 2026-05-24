"""Prompt-facing metric semantics for LLM-based evaluation.

Metric descriptors provide an authoritative description of pass/score rules
and parameter meanings for prompts.
"""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class MetricPromptDescriptor:
    """Renderable, JSON-serializable metric semantics for prompts."""

    metric_id: str
    summary: str
    pass_rule: str | None = None
    score_rule: str | None = None
    params: Mapping[str, Any] | None = None
    notes: tuple[str, ...] = ()

    def describe(
        self, *, params_examples: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"metric_id": self.metric_id, "summary": self.summary}
        if self.pass_rule is not None:
            payload["pass_rule"] = self.pass_rule
        if self.score_rule is not None:
            payload["score_rule"] = self.score_rule
        if self.params is not None:
            payload["params"] = dict(self.params)
        if self.notes:
            payload["notes"] = list(self.notes)
        if params_examples:
            payload["params_examples"] = params_examples
        return payload
