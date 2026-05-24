"""
Agentic components of the optimization pipeline.

Extractor : Produces structured JSON from raw document text using the current prompt.
Critic    : Performs a surgical semantic diff between prediction and gold standard,
            identifying specific failure modes with actionable, field-level labels.
Mutator   : Prompt-engineer agent that synthesizes critiques into a non-regressive
            prompt improvement, maintaining a rejection memory to avoid re-proposing
            failed variants and escalating boldness during stalls.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor(BaseAgent):
    """Executes the current prompt against a document to produce JSON."""

    _SYSTEM_TEMPLATE = """{current_prompt}

TARGET SCHEMA (JSON Schema):
{schema}

OUTPUT RULES (non-negotiable):
- Return ONLY valid JSON that conforms to the schema above. No markdown fences, no commentary, no preamble.
- Every top-level key defined in the schema MUST appear in your output (use null for absent scalars, [] for absent arrays).
- Do NOT invent or hallucinate data that is not explicitly present in the document.
- Preserve exact spelling, capitalisation, and punctuation for all extracted string values.
- For fields with anyOf containing null: output null if the value is genuinely absent.
"""

    def extract(self, document_text: str, current_prompt: str, schema: str) -> str:
        system_prompt = self._SYSTEM_TEMPLATE.format(
            current_prompt=current_prompt,
            schema=schema,
        )
        return self.call_llm(
            system_prompt=system_prompt,
            user_prompt=f"DOCUMENT:\n{document_text}",
            role_name="Extractor",
            temperature=0.0,  # Fully deterministic for extraction
        )


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic(BaseAgent):
    """
    Analyses extraction failures and returns structured, actionable critiques.

    The output format is designed to be directly consumable by the Mutator:
    each entry identifies a failure category, the specific field path, the
    predicted vs gold values, and a concrete one-sentence fix instruction.
    """

    _SYSTEM_PROMPT = """You are a precision data-extraction auditor specialising in JSON schema compliance.

Your task: compare a predicted JSON extraction against the gold-standard annotation and produce
actionable, field-level diagnoses that a prompt engineer can act on immediately.

For EACH discrepancy, output a critique in EXACTLY this format:
  [FIELD: <dot.path.to.field>] [TYPE: <MISSING|WRONG_VALUE|TYPE_MISMATCH|HALLUCINATED|FORMAT_ERROR|ARRAY_MISMATCH|NULL_WHEN_PRESENT>]
  Predicted: <value or 'ABSENT' or 'null'>
  Gold:      <value>
  Fix:       <one concrete instruction for improving the extraction prompt>

Rules:
- Be surgical: skip fields that match correctly.
- Prioritise the top 5 most impactful failures (those affecting the most leaf fields).
- Consider type mismatches (e.g. string "2020" vs integer 2020, missing array vs null scalar).
- Consider field name mismatches (e.g. model outputs "company" but schema expects "employer").
- Do NOT suggest model fine-tuning, data changes, or post-processing — only prompt wording fixes.
- If the extraction is perfect, output exactly: NO_FAILURES
"""

    def critique(
        self,
        document_text: str,
        predicted_json: str,
        gold_json: str,
    ) -> str:
        user_prompt = (
            f"DOCUMENT (first 2000 chars):\n{document_text[:2000]}\n\n"
            f"PREDICTED JSON:\n{predicted_json}\n\n"
            f"GOLD STANDARD JSON:\n{gold_json}\n\n"
            "List all discrepancies using the specified format."
        )
        return self.call_llm(
            system_prompt=self._SYSTEM_PROMPT,
            user_prompt=user_prompt,
            role_name="Critic",
            temperature=0.0,
        )


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------

class Mutator(BaseAgent):
    """
    Automated prompt engineer.

    Strategy:
    1. Receives the current best prompt + a batch of structured critiques.
    2. Maintains a rejection memory to avoid re-proposing previously failed variants.
    3. Detects stalls (repeated no-improvement iterations) and escalates to bolder changes.
    4. Returns ONLY the new prompt text — no preamble, no labels, no explanation.
    """

    _SYSTEM_PROMPT = """You are a world-class prompt engineer specialising in structured JSON extraction from documents.

Your task: rewrite the given extraction prompt to fix every listed failure mode WITHOUT
degrading performance on fields that currently extract correctly.

Internal reasoning process (do NOT include this in the output):
1. Group each critique by failure type (missing field, wrong value, format error, type mismatch, etc.)
2. Identify which instructions in the current prompt are absent, ambiguous, or contradicted.
3. Draft targeted additions or clarifications for each failure group.
4. Verify that new rules do not conflict with currently working extraction rules.
5. Write the final, self-contained prompt.

Key improvement strategies by failure type:
- MISSING field     : Add an explicit extraction rule naming the exact field and describing where to find it.
- WRONG_VALUE       : Add a clarification about which value to prefer (e.g. most recent, as written).
- TYPE_MISMATCH     : Add an explicit type rule (e.g. "output years as integers, not strings").
- HALLUCINATED      : Strengthen the "extract only from the document" prohibition.
- FORMAT_ERROR      : Add a precise format example (e.g. "ISO 8601 date as YYYY-MM-DD").
- ARRAY_MISMATCH    : Clarify ordering policy (most recent first) and completeness requirement.
- NULL_WHEN_PRESENT : Stress that a field present in the document must never be null.

OUTPUT FORMAT:
Return ONLY the final prompt text — no labels, no markdown, no explanation.
The prompt must be completely self-contained; it is sent directly to the extraction model.
"""

    def mutate(
        self,
        current_prompt: str,
        critiques: list[str],
        rejected_prompts: list[str] | None = None,
        stall_count: int = 0,
    ) -> str:
        """
        Propose an improved prompt.

        Parameters
        ----------
        current_prompt  : The currently accepted best prompt.
        critiques       : List of Critic outputs from failed documents.
        rejected_prompts: Prompts already tried and rejected (last 5 shown).
        stall_count     : Consecutive iterations with no improvement.
                          If >= 3, the mutator is instructed to try a bolder change.
        """
        critique_block = "\n\n---\n\n".join(
            f"Critique {i + 1}:\n{c}" for i, c in enumerate(critiques)
        )

        rejection_block = ""
        if rejected_prompts:
            recent = rejected_prompts[-5:]
            rejection_block = (
                "\n\nREJECTED PROMPTS — do NOT reproduce these variants (they scored worse):\n"
                + "\n\n".join(
                    f"[REJECTED {i + 1}]:\n{p}" for i, p in enumerate(recent)
                )
            )

        stall_note = ""
        if stall_count >= 3:
            stall_note = (
                f"\n\n⚠️  STALL ALERT: The validation score has not improved for "
                f"{stall_count} consecutive iterations. The incremental approach is not "
                "working. Try a significantly different strategy: restructure the field rules, "
                "add a concrete worked extraction example, or decompose a complex field into "
                "explicit sub-steps."
            )

        user_prompt = (
            f"CURRENT BEST PROMPT:\n{current_prompt}\n\n"
            f"FAILURE CRITIQUES FROM VALIDATION SET:\n{critique_block}"
            f"{rejection_block}"
            f"{stall_note}\n\n"
            "Write the improved prompt now."
        )

        return self.call_llm(
            system_prompt=self._SYSTEM_PROMPT,
            user_prompt=user_prompt,
            role_name="Mutator",
            temperature=0.4,
        )
