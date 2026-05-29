"""
Agentic components of the optimization pipeline.

Extractor : Produces structured JSON from raw document text using the current prompt.
Critic    : Performs a surgical semantic diff between prediction and gold standard,
            identifying specific failure modes with actionable, field-level labels
            and severity scores.
Mutator   : Prompt-engineer agent that synthesises critiques into a non-regressive
            prompt improvement.  Supports:
              - Rejection memory to avoid re-proposing failed variants.
              - Escalating boldness during stalls.
              - Few-shot example injection from the train set.
              - Beam-aware restart hints when the beam's second candidate is provided.

All three agents inherit BaseAgent's four-tier fallback:
  OpenRouter → GitHub Models → Gemini text → Ollama
"""

from __future__ import annotations
import json  # Added for context compression

from src.agents.base_agent import BaseAgent
from src.core.state_manager import StateManager


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor(BaseAgent):
    """Executes the current prompt against a document to produce JSON."""

    _SYSTEM_TEMPLATE = """\
══════════════════════════════════════════════════════════
EXTRACTION ENGINE — OUTPUT CONTRACT (highest priority)
══════════════════════════════════════════════════════════
- Return ONLY raw JSON.  No ```json fences, no preamble, no trailing text.
- The root object MUST contain EXACTLY these top-level keys — never more:
    personalInfo  workExperience  education  skills  languages  socialLinks
    certificationsAndAwards  publications  media  other
- FORBIDDEN root keys: schema_definition, name, data, result, cv, resume,
  or ANY key not listed above.
- Absent scalar → null.   Absent array → [].
- Year-only dates → INTEGER (2020 not "2020").
- isCurrent → boolean (true / false — never a string).
- Do NOT invent or hallucinate data not present in the document.
══════════════════════════════════════════════════════════

{current_prompt}

══════════════════════════════════════════════════════════
TARGET JSON SCHEMA
══════════════════════════════════════════════════════════
{schema}
"""

    def __init__(self, model_name: str, state_manager: StateManager,
                 github_model: str | None = None):
        super().__init__(model_name, state_manager, github_model=github_model)

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
    Analyses extraction failures and returns structured, prioritised critiques.
    """

    _SYSTEM_PROMPT = """\
You are a precision data-extraction auditor specialising in JSON schema compliance.
Your job: compare a predicted JSON extraction against gold-standard annotations and
produce actionable, field-level diagnoses that a prompt engineer can act on immediately.

For EACH discrepancy output a critique block in EXACTLY this format:

  [SEVERITY: HIGH|MEDIUM|LOW]
  [FIELD: <dot.path.to.field>]
  [TYPE: MISSING|WRONG_VALUE|TYPE_MISMATCH|HALLUCINATED|FORMAT_ERROR|ARRAY_MISMATCH|NULL_WHEN_PRESENT|FORBIDDEN_KEY]
  Predicted : <value, or ABSENT, or null>
  Gold      : <expected value>
  Fix       : <one concrete instruction for rewording the extraction prompt>

Severity guidelines:
  HIGH   — affects a required top-level field or produces wrong root structure
  MEDIUM — affects a repeated array (workExperience, education) or key nested field
  LOW    — affects optional/rare fields (publications, media, other)

Prioritise by severity (HIGH first), then list up to 7 total failures.

Rules:
  • Skip fields that match correctly — only list failures.
  • Focus on TYPE_MISMATCH for year-as-string-vs-integer and isCurrent-as-string.
  • Focus on FORBIDDEN_KEY if the root object contains schema_definition or other illegal keys.
  • Do NOT suggest fine-tuning, data changes, or post-processing — only prompt wording fixes.
  • If the extraction is structurally correct, output exactly: NO_FAILURES
"""

    def __init__(self, model_name: str, state_manager: StateManager,
                 github_model: str | None = None):
        super().__init__(model_name, state_manager, github_model=github_model)

    def critique(
        self,
        document_text: str,
        predicted_json: str,
        gold_json: str,
    ) -> str:
        # STRATEGY: Minify JSONs to drastically save input tokens
        try:
            pred_min = json.dumps(json.loads(predicted_json))
        except:
            pred_min = predicted_json
            
        try:
            gold_min = json.dumps(json.loads(gold_json))
        except:
            gold_min = gold_json

        # STRATEGY: Cut document snippet from 2000 down to 1200 chars. 
        user_prompt = (
            f"DOCUMENT (first 1200 chars):\n{document_text[:1200]}\n\n"
            f"PREDICTED JSON:\n{pred_min}\n\n"
            f"GOLD STANDARD JSON:\n{gold_min}\n\n"
            "List all discrepancies using the specified format, severity-first."
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
    """

    _SYSTEM_PROMPT = """\
You are a world-class prompt engineer specialising in structured JSON extraction from documents.

Your task: rewrite the given extraction prompt to fix every listed failure mode WITHOUT
degrading performance on fields that currently extract correctly.

Internal reasoning (do NOT include in your output):
  1. Group critiques by failure type (missing, wrong value, type mismatch, format, etc.)
  2. Identify which instructions are absent, ambiguous, or contradicted in the current prompt.
  3. Draft targeted additions / clarifications for each failure group.
  4. Check new rules do not conflict with currently-working extraction rules.
  5. If a train example is provided, embed it verbatim as a WORKED EXAMPLE section.
  6. Write the final, self-contained prompt.

Improvement strategies by failure type:
  MISSING field      → Add an explicit rule naming the exact field and where to find it.
  WRONG_VALUE        → Clarify which value to prefer (most recent, exact as-written, etc.).
  TYPE_MISMATCH      → Add an explicit type rule with example ("output years as integers: 2020").
  HALLUCINATED       → Strengthen "extract ONLY from the document" prohibition.
  FORMAT_ERROR       → Add a precise format example matching the schema.
  ARRAY_MISMATCH     → Clarify ordering (most-recent first) and completeness requirement.
  NULL_WHEN_PRESENT  → Stress that a field present in the document must never be null.
  FORBIDDEN_KEY      → Add the exact forbidden key to the root-key prohibition list.

Critical constraints:
  • The improved prompt MUST preserve the root-key contract:
    personalInfo  workExperience  education  skills  socialLinks
    certificationsAndAwards  publications  media  other
  • Never suggest a field name that differs from the schema (e.g. do not use "company"
    when the schema uses "employer", do not use "degree" for "qualificationTitle").
  • NEVER add root-level keys not in: personalInfo workExperience education skills languages socialLinks certificationsAndAwards publications media other.
    EXPLICITLY FORBIDDEN extra keys: 'certifications', 'summary', 'profile', 'contact', 'awards'.
  • ANTI-REGRESSION RULE: You must retain ALL existing instructions for fields that are currently working correctly. Do NOT delete or shorten rules for fields that do not appear in the Failure Critiques.
  • The prompt must work standalone — it is sent directly to the extraction model.

OUTPUT FORMAT:
Return ONLY the final prompt text — no labels, no markdown, no explanation.
"""

    def __init__(self, model_name: str, state_manager: StateManager,
                 github_model: str | None = None):
        super().__init__(model_name, state_manager, github_model=github_model)

    def mutate(
        self,
        current_prompt: str,
        critiques: list[str],
        rejected_prompts: list[str] | None = None,
        stall_count: int = 0,
        train_example: str | None = None,
        secondary_prompt: str | None = None,
    ) -> str:
        
        critique_block = "\n\n---\n\n".join(
            f"Critique {i + 1}:\n{c}" for i, c in enumerate(critiques)
        )

        rejection_block = ""
        if rejected_prompts:
            # STRATEGY: Keep only the last 3 rejected prompts (instead of 5) 
            # and truncate them to 400 chars (instead of 600) to save tokens.
            recent = rejected_prompts[-3:]
            rejection_block = (
                "\n\nREJECTED PROMPTS — do NOT reproduce these variants:\n"
                + "\n\n".join(
                    f"[REJECTED {i + 1}]:\n{p[:400]}…" for i, p in enumerate(recent)
                )
            )

        stall_note = ""
        if stall_count >= 5:
            stall_note = (
                f"\n\n⚠️ SEVERE STALL ({stall_count} iterations without improvement). "
                "Take a RADICAL approach: restructure the field rules, change instruction order, "
                "or decompose a failing field into explicit numbered sub-steps."
            )
        elif stall_count >= 3:
            stall_note = (
                f"\n\n⚠️ STALL ({stall_count} iterations without improvement). "
                "Try a significantly different strategy: add concrete format examples inline."
            )

        example_block = ""
        if train_example:
            example_block = (
                "\n\n📌 TRAIN EXAMPLE:\n"
                f"{train_example}"
            )

        beam_block = ""
        if secondary_prompt:
            # STRATEGY: Truncate secondary prompt heavily.
            beam_block = (
                "\n\nALTERNATIVE PROMPT (second-best from beam):\n"
                f"{secondary_prompt[:800]}…"
            )

        user_prompt = (
            f"CURRENT BEST PROMPT:\n{current_prompt}\n\n"
            f"FAILURE CRITIQUES (severity-ranked):\n{critique_block}"
            f"{rejection_block}"
            f"{stall_note}"
            f"{example_block}"
            f"{beam_block}\n\n"
            "Write the improved prompt now."
        )

        return self.call_llm(
            system_prompt=self._SYSTEM_PROMPT,
            user_prompt=user_prompt,
            role_name="Mutator",
            temperature=0.4,
        )