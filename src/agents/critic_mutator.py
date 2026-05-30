"""
Agentic components of the optimization pipeline.

Extractor : Produces structured JSON from raw document text using the current prompt.
Critic    : Performs a surgical semantic diff between prediction and gold standard,
            identifying specific failure modes with actionable, field-level labels
            and severity scores.
Mutator   : Prompt-engineer agent that synthesises critiques into a non-regressive
            prompt improvement. Supports rejection memory, escalating boldness,
            train-example injection, and beam-aware synthesis.

All three agents inherit BaseAgent's four-tier fallback:
  OpenRouter → GitHub Models (gpt-4o) → Gemini text → Ollama
"""

from __future__ import annotations

import json

from src.agents.base_agent import BaseAgent
from src.core.state_manager import StateManager


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor(BaseAgent):
    """Executes the current prompt against a document to produce JSON."""

    # Hard output contract is injected here so it always overrides whatever
    # the mutator does to the per-field instruction section.
    _SYSTEM_TEMPLATE = """\
══════════════════════════════════════════════════════════
EXTRACTION ENGINE — OUTPUT CONTRACT (highest priority)
══════════════════════════════════════════════════════════
• Return ONLY raw JSON. No ```json fences, no preamble, no trailing text.
• The root object MUST contain EXACTLY these top-level keys — never more:
    personalInfo  workExperience  education  skills  languages  socialLinks
    certificationsAndAwards  publications  media  other
• FORBIDDEN root keys: schema_definition, name, data, result, cv, resume,
  contact, profile, summary, awards, certifications, or ANY key not listed above.
• Absent scalar → null.   Absent array → [].
• Year-only dates → INTEGER (2020 not "2020" — NEVER use ISO timestamp strings).
• isCurrent → boolean (true / false — never a string).
• Do NOT invent data not present in the document.
• Extract EVERY item in each section — do not truncate long arrays.
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
            temperature=0.0,
        )


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic(BaseAgent):
    """
    Analyses extraction failures and returns structured, prioritised critiques.
    Minifies JSON inputs to stay within GitHub Models token limits.
    """

    _SYSTEM_PROMPT = """\
You are a precision data-extraction auditor specialising in JSON schema compliance.
Your job: compare a predicted JSON extraction against gold-standard annotations and
produce actionable, field-level diagnoses that a prompt engineer can act on immediately.

For EACH discrepancy output a critique block in EXACTLY this format:

  [SEVERITY: HIGH|MEDIUM|LOW]
  [FIELD: <dot.path.to.field>]
  [TYPE: MISSING|WRONG_VALUE|TYPE_MISMATCH|HALLUCINATED|FORMAT_ERROR|ARRAY_MISMATCH|NULL_WHEN_PRESENT|FORBIDDEN_KEY|TRUNCATED_ARRAY]
  Predicted : <value, or ABSENT, or null>
  Gold      : <expected value or count>
  Fix       : <one concrete instruction for rewording the extraction prompt>

Severity guidelines:
  HIGH   — affects a required top-level field or wrong root structure
  MEDIUM — affects a repeated array (workExperience, education, publications) or key nested field
  LOW    — affects optional/rare fields (media, other)

Use TRUNCATED_ARRAY when the prediction has fewer items than gold for publications,
workExperience, certificationsAndAwards, or education — this means the model stopped
extracting early and the prompt needs a stronger completeness instruction.

Prioritise by severity (HIGH first), then list up to 8 total failures.

Rules:
  • Skip fields that match correctly — only list failures.
  • Focus on TYPE_MISMATCH for year-as-string vs integer (e.g. "2020" vs 2020).
  • Do NOT suggest ISO timestamp formats — gold data always uses integer years.
  • Focus on FORBIDDEN_KEY for any unexpected root keys.
  • Do NOT suggest fine-tuning, data changes, or post-processing.
  • If structurally correct with no failures: output exactly: NO_FAILURES
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
        # Minify JSONs to save tokens
        try:
            pred_min = json.dumps(json.loads(predicted_json))
        except Exception:
            pred_min = predicted_json[:3000]

        try:
            gold_min = json.dumps(json.loads(gold_json))
        except Exception:
            gold_min = gold_json[:3000]

        user_prompt = (
            f"DOCUMENT (first 1500 chars):\n{document_text[:1500]}\n\n"
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

    Key improvements over baseline:
    - Explicit zero-field recovery strategies for publications, skills, certs
    - Correct languages field in root-key contract
    - Anti-regression rule to preserve working fields
    - Explicit prohibition of ISO timestamp instructions
    """

    _SYSTEM_PROMPT = """\
You are a world-class prompt engineer specialising in structured JSON extraction from documents.

Your task: rewrite the given extraction prompt to fix every listed failure WITHOUT
degrading performance on fields that currently extract correctly.

══════════════════════════════════════════════════════════
IMPROVEMENT STRATEGIES BY FAILURE TYPE
══════════════════════════════════════════════════════════

MISSING / TRUNCATED_ARRAY field:
  publications  → The model likely doesn't recognise the section. Add ALL known
                  section headers: "Publications", "Peer-Reviewed Articles", "Journal
                  Articles", "Conference Papers", "Conference Proceedings", "Book
                  Chapters", "Books", "Working Papers", "Preprints", "Technical Reports",
                  "Presentations", "Published Works", "Research Output".
                  Add a completeness warning: "Academic CVs have 5–30+ publications."
  workExperience → Add a completeness warning and list ALL section headers for
                  academic CVs: "Academic Appointments", "Research Experience",
                  "Teaching Experience", "Industry Experience", "Consulting",
                  "Visiting Positions", "Postdoctoral Research".
  certificationsAndAwards → Add: named fellowships and research grants → "Award",
                  bar admissions → "License", academic honours → "Honor", society
                  memberships → "Membership". Include "License" as a valid category.
  skills        → Add ALL heading variants: "Technical Skills", "Research Methods",
                  "Expertise", "Core Competencies", "Programming Languages", "Software",
                  "Tools", "Areas of Expertise", "Competencies".
  socialLinks   → Explicitly mention ORCID (orcid.org/...), ResearchGate, Google Scholar,
                  lab website, portfolio as URL types to look for.

WRONG_VALUE       → Clarify which value to prefer (exact as-written, most recent, etc.)
TYPE_MISMATCH     → Add explicit type rule: "output years as integers: 2020 not '2020'"
                    ⚠️  NEVER instruct the model to use ISO 8601 timestamp strings —
                    the gold data ALWAYS uses plain integer years or short strings.
HALLUCINATED      → Strengthen "extract ONLY from the document" prohibition.
FORMAT_ERROR      → Add a precise format example matching the schema.
NULL_WHEN_PRESENT → Stress that a field present in the document must never be null.
FORBIDDEN_KEY     → Add the exact forbidden key to the root-key prohibition list.

══════════════════════════════════════════════════════════
CRITICAL CONSTRAINTS — NEVER VIOLATE
══════════════════════════════════════════════════════════
1. ROOT KEY CONTRACT — the improved prompt MUST list EXACTLY these keys:
     personalInfo  workExperience  education  skills  languages  socialLinks
     certificationsAndAwards  publications  media  other
   (Note: `languages` is REQUIRED in rule 2. Do not remove it.)

2. SCHEMA FIDELITY — never rename fields:
   "employer" not "company", "qualificationTitle" not "degree",
   "certificationsAndAwards" not "certifications"

3. ANTI-REGRESSION — retain ALL existing instructions for fields NOT mentioned in
   the critiques. Do not delete or shorten rules for fields that are working.

4. NO ISO TIMESTAMPS — never add instructions to format dates as ISO 8601 strings
   (e.g. "2019-02-28T23:00:00.000Z"). Gold data uses plain integers (2019) or short
   strings ("Spring 2010"). Any ISO timestamp instruction will cause all date fields
   to score 0.0. This is the most damaging regression possible.

5. STANDALONE — the prompt must work on its own; it is sent directly to the model.

══════════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════════
Return ONLY the final prompt text — no labels, no markdown, no preamble.
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
            recent = rejected_prompts[-3:]
            rejection_block = (
                "\n\nREJECTED PROMPTS — do NOT reproduce these variants:\n"
                + "\n\n".join(
                    f"[REJECTED {i+1}]:\n{p[:500]}…" for i, p in enumerate(recent)
                )
            )

        stall_note = ""
        if stall_count >= 5:
            stall_note = (
                f"\n\n⚠️  SEVERE STALL ({stall_count} iterations without improvement). "
                "Take a RADICAL approach: completely restructure the failing field rules, "
                "reorder sections, or decompose a complex field into numbered sub-steps. "
                "Do NOT reproduce any rejected variant."
            )
        elif stall_count >= 3:
            stall_note = (
                f"\n\n⚠️  STALL ({stall_count} iterations). Incremental edits are not "
                "working. Try a significantly different approach for the failing fields."
            )

        example_block = ""
        if train_example:
            example_block = (
                "\n\n📌 TRAIN EXAMPLE — embed a worked example like this in the "
                "improved prompt to give the model a concrete grounding reference:\n"
                f"{train_example[:2000]}"
            )

        beam_block = ""
        if secondary_prompt:
            beam_block = (
                "\n\nALTERNATIVE PROMPT (second-best from beam — mine for useful ideas):\n"
                f"{secondary_prompt[:900]}…"
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