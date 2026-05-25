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
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor(BaseAgent):
    """Executes the current prompt against a document to produce JSON."""

    # The schema and hard output-contract rules are injected here so they
    # are always present regardless of what the mutator produces.  The
    # current_prompt supplies the field-level extraction instructions.
    _SYSTEM_TEMPLATE = """\
══════════════════════════════════════════════════════════
EXTRACTION ENGINE — OUTPUT CONTRACT (highest priority)
══════════════════════════════════════════════════════════
• Return ONLY raw JSON.  No ```json fences, no preamble, no trailing text.
• The root object MUST contain EXACTLY these top-level keys — never more:
    personalInfo  workExperience  education  skills  socialLinks
    certificationsAndAwards  publications  media  other
• FORBIDDEN root keys: schema_definition, name, data, result, cv, resume,
  or ANY key not listed above.  If you add forbidden keys the output is wrong.
• Absent scalar → null.   Absent array → [].
• Year-only dates → INTEGER (2020 not "2020").
• isCurrent → boolean (true / false — never a string).
• Do NOT invent or hallucinate data not present in the document.
══════════════════════════════════════════════════════════

{current_prompt}

══════════════════════════════════════════════════════════
TARGET JSON SCHEMA
══════════════════════════════════════════════════════════
{schema}
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
    Analyses extraction failures and returns structured, prioritised critiques.

    Output format is designed for direct consumption by the Mutator: each
    entry has a severity score, failure type, exact field path, predicted vs
    gold values, and a one-sentence actionable fix instruction.
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

    Strategy
    --------
    1. Receives the current best prompt + prioritised critiques from Critic.
    2. Maintains rejection memory to avoid re-proposing failed variants.
    3. Detects stalls and escalates to bolder rewrites.
    4. When a train example is provided (stall >= 2), embeds a concrete
       worked example directly into the improved prompt — the single most
       effective technique for smaller models.
    5. When a secondary beam prompt is provided (severe stall >= 5), uses it
       as an alternative starting point for the rewrite.
    6. Returns ONLY the new prompt text — no preamble, labels, or explanation.
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
  • The prompt must work standalone — it is sent directly to the extraction model.

OUTPUT FORMAT:
Return ONLY the final prompt text — no labels, no markdown, no explanation.
"""

    def mutate(
        self,
        current_prompt: str,
        critiques: list[str],
        rejected_prompts: list[str] | None = None,
        stall_count: int = 0,
        train_example: str | None = None,
        secondary_prompt: str | None = None,
    ) -> str:
        """
        Propose an improved prompt.

        Parameters
        ----------
        current_prompt   : The currently accepted best prompt.
        critiques        : List of Critic outputs from failed documents.
        rejected_prompts : Prompts already tried and rejected (last 5 shown).
        stall_count      : Consecutive iterations with no improvement.
        train_example    : Formatted (text → gold JSON) example from the train set.
                           Injected when stall_count >= 2 to give the model a
                           concrete grounding example to embed in the prompt.
        secondary_prompt : Second-best prompt from the beam (provided on severe stall).
                           When present, the mutator is asked to synthesise ideas
                           from both prompts rather than mutate from the primary alone.
        """
        critique_block = "\n\n---\n\n".join(
            f"Critique {i + 1}:\n{c}" for i, c in enumerate(critiques)
        )

        rejection_block = ""
        if rejected_prompts:
            recent = rejected_prompts[-5:]
            rejection_block = (
                "\n\nREJECTED PROMPTS — do NOT reproduce these variants:\n"
                + "\n\n".join(
                    f"[REJECTED {i + 1}]:\n{p[:600]}…" for i, p in enumerate(recent)
                )
            )

        stall_note = ""
        if stall_count >= 5:
            stall_note = (
                f"\n\n⚠️  SEVERE STALL ({stall_count} iterations without improvement). "
                "Incremental changes are not working. Take a RADICAL approach: completely "
                "restructure the field rules, change the instruction order, or decompose "
                "a failing field into explicit numbered sub-steps. Do NOT produce a prompt "
                "that resembles any rejected variant."
            )
        elif stall_count >= 3:
            stall_note = (
                f"\n\n⚠️  STALL ({stall_count} iterations without improvement). "
                "Incremental edits are not working. Try a significantly different strategy: "
                "add concrete format examples inline, reorder the field rules by importance, "
                "or break down a complex field into explicit sub-steps."
            )

        example_block = ""
        if train_example:
            example_block = (
                "\n\n📌 TRAIN EXAMPLE — embed a worked example like this in your improved prompt "
                "so the model has a concrete grounding reference:\n"
                f"{train_example}"
            )

        beam_block = ""
        if secondary_prompt:
            beam_block = (
                "\n\nALTERNATIVE PROMPT (second-best from beam — mine it for useful ideas "
                "and synthesise the best elements of both into your rewrite):\n"
                f"{secondary_prompt[:1200]}…"
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