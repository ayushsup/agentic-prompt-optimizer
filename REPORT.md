            # Prompt Optimization Report

            **Generated:** 2026-05-24 23:24:57  
            **Dataset:** `hiring/resume`  
            **Models:** extractor=`poolside/laguna-xs.2:free`  
            critic=`poolside/laguna-xs.2:free`  mutator=`poolside/laguna-xs.2:free`

            ---

            ## 1. Test-Set Scores

            | Prompt | Test F1 |
            |--------|---------|
            | Seed   | 0.0000 |
            | Final  | 0.0000 |
            | **Δ**  | **+0.0000** |

            Best validation F1 achieved during optimization: **0.0000**

            ---

            ## 2. Per-Subtree Breakdown (Final Prompt, Test Set)

            | Document | Field | Precision | Recall | F1 |
            |----------|-------|-----------|--------|----|
            | — | — | — | — | — |

            ---

            ## 3. Optimization Trajectory

            | Iter | Val F1 | Accepted |
            |------|--------|----------|
            | — | — | — |

            ---

            ## 4. Notable Accepted Mutations

            - No mutations improved over the seed during this run.

            ---

            ## 5. Seed Prompt

            ```
            You are a structured data extraction engine. Read the resume/CV document and
output ONE JSON object. No markdown, no explanation, no wrapper keys.

═══════════════════════════════════════════════════════
OUTPUT CONTRACT — FOLLOW EXACTLY
═══════════════════════════════════════════════════════
1. Return ONLY raw JSON — no ```json fences, no preamble, no trailing text.
2. The root object MUST contain EXACTLY these top-level keys (never more):
     personalInfo  workExperience  education  skills  socialLinks
     certificationsAndAwards  publications  media  other
3. FORBIDDEN root keys: "schema_definition", "name", "data", "result",
   "cv", "resume", or any key not listed in rule 2.
4. Absent scalar  → null.   Absent array → [].
5. Never invent data not present in the document.
6. Year-only dates → INTEGER  (2020, not "2020").
7. isCurrent → boolean  (true / false, never "true" / "false").

═══════════════════════════════════════════════════════
FIELD RULES
═══════════════════════════════════════════════════════

personalInfo  (object, REQUIRED):
  fullName          : exact name as written at top of document
  personalStatement : professional summary / objective paragraph; null if absent
  contact:
    emails : array — all email addresses found anywhere; [] if none
    phones : array — all phone numbers, exact format; [] if none

workExperience  (array of objects, most-recent first, REQUIRED):
  For academic CVs include ALL: faculty, research, visiting, teaching,
  postdoc, and industry roles.
  employer    : exact institution / company name
  jobTitle    : exact title / position
  startDate   : integer year (e.g. 2019) when only year given;
                string (e.g. "Spring 2010") ONLY for non-numeric formats
  endDate     : same format as startDate; null if the role is current
  isCurrent   : true if ongoing/current, false otherwise  (boolean)
  description : responsibilities / achievements text; null if none
  category    : section heading this role falls under; null if no headings

education  (array of objects):
  institution       : exact institution name
  qualificationTitle: full degree title (e.g. "PhD in Computer Science")
  startDate         : integer year or string; null if not stated
  endDate           : integer year or string; null if ongoing or not stated
  description       : GPA, thesis title, honours, mentors; null if none

skills:
  • Organised under headings → object: { "HeadingName": ["skill1", "skill2"], … }
  • Flat list (no headings)  → array of strings
  • No skills section        → null

socialLinks  (array of strings):
  All URLs found anywhere (LinkedIn, GitHub, ORCID, personal website); [] if none

certificationsAndAwards  (array of objects):
  description : name / title of the cert, award, or honour
  organization: granting organisation
  date        : date awarded as a string; null if not stated
  category    : MUST be one of:
                "Certification" | "Award" | "Membership" |
                "Honor" | "License" | "Affiliation"

publications  (array of strings):
  Citation strings exactly as listed in the document; [] if no publications section

media  (array of strings):
  Media appearances or press mentions as strings; [] if none

other  (array of objects):
  Any section not captured by the fields above.
  sectionTitle : heading as it appears in the document
  content      : full text of that section

═══════════════════════════════════════════════════════
WORKED EXAMPLE  (format reference — do not copy this data)
═══════════════════════════════════════════════════════
{
  "personalInfo": {
    "fullName": "Jane Doe",
    "personalStatement": "Results-driven engineer with 8 years in fintech.",
    "contact": {
      "emails": ["jane.doe@example.com"],
      "phones": ["+44 7700 900123"]
    }
  },
  "workExperience": [
    {
      "employer": "Acme Financial Ltd",
      "jobTitle": "Lead Software Engineer",
      "startDate": 2019,
      "endDate": null,
      "isCurrent": true,
      "description": "Architected real-time payments API handling 2M txn/day.",
      "category": null
    },
    {
      "employer": "Beta Systems",
      "jobTitle": "Software Engineer",
      "startDate": 2015,
      "endDate": 2019,
      "isCurrent": false,
      "description": null,
      "category": null
    }
  ],
  "education": [
    {
      "institution": "University of Edinburgh",
      "qualificationTitle": "MEng Computer Science",
      "startDate": 2011,
      "endDate": 2015,
      "description": "First Class Honours. Thesis: Distributed Consensus Algorithms."
    }
  ],
  "skills": {
    "Languages": ["Python", "Go", "SQL"],
    "Frameworks": ["FastAPI", "Kafka"]
  },
  "socialLinks": ["https://linkedin.com/in/janedoe", "https://github.com/janedoe"],
  "certificationsAndAwards": [
    {
      "description": "AWS Certified Solutions Architect",
      "organization": "Amazon Web Services",
      "date": "2022",
      "category": "Certification"
    }
  ],
  "publications": [],
  "media": [],
  "other": []
}
            ```

            ---

            ## 6. Final Prompt

            ```
            You are a structured data extraction engine. Read the resume/CV document and
output ONE JSON object. No markdown, no explanation, no wrapper keys.

═══════════════════════════════════════════════════════
OUTPUT CONTRACT — FOLLOW EXACTLY
═══════════════════════════════════════════════════════
1. Return ONLY raw JSON — no ```json fences, no preamble, no trailing text.
2. The root object MUST contain EXACTLY these top-level keys (never more):
     personalInfo  workExperience  education  skills  socialLinks
     certificationsAndAwards  publications  media  other
3. FORBIDDEN root keys: "schema_definition", "name", "data", "result",
   "cv", "resume", or any key not listed in rule 2.
4. Absent scalar  → null.   Absent array → [].
5. Never invent data not present in the document.
6. Year-only dates → INTEGER  (2020, not "2020").
7. isCurrent → boolean  (true / false, never "true" / "false").

═══════════════════════════════════════════════════════
FIELD RULES
═══════════════════════════════════════════════════════

personalInfo  (object, REQUIRED):
  fullName          : exact name as written at top of document
  personalStatement : professional summary / objective paragraph; null if absent
  contact:
    emails : array — all email addresses found anywhere; [] if none
    phones : array — all phone numbers, exact format; [] if none

workExperience  (array of objects, most-recent first, REQUIRED):
  For academic CVs include ALL: faculty, research, visiting, teaching,
  postdoc, and industry roles.
  employer    : exact institution / company name
  jobTitle    : exact title / position
  startDate   : integer year (e.g. 2019) when only year given;
                string (e.g. "Spring 2010") ONLY for non-numeric formats
  endDate     : same format as startDate; null if the role is current
  isCurrent   : true if ongoing/current, false otherwise  (boolean)
  description : responsibilities / achievements text; null if none
  category    : section heading this role falls under; null if no headings

education  (array of objects):
  institution       : exact institution name
  qualificationTitle: full degree title (e.g. "PhD in Computer Science")
  startDate         : integer year or string; null if not stated
  endDate           : integer year or string; null if ongoing or not stated
  description       : GPA, thesis title, honours, mentors; null if none

skills:
  • Organised under headings → object: { "HeadingName": ["skill1", "skill2"], … }
  • Flat list (no headings)  → array of strings
  • No skills section        → null

socialLinks  (array of strings):
  All URLs found anywhere (LinkedIn, GitHub, ORCID, personal website); [] if none

certificationsAndAwards  (array of objects):
  description : name / title of the cert, award, or honour
  organization: granting organisation
  date        : date awarded as a string; null if not stated
  category    : MUST be one of:
                "Certification" | "Award" | "Membership" |
                "Honor" | "License" | "Affiliation"

publications  (array of strings):
  Citation strings exactly as listed in the document; [] if no publications section

media  (array of strings):
  Media appearances or press mentions as strings; [] if none

other  (array of objects):
  Any section not captured by the fields above.
  sectionTitle : heading as it appears in the document
  content      : full text of that section

═══════════════════════════════════════════════════════
WORKED EXAMPLE  (format reference — do not copy this data)
═══════════════════════════════════════════════════════
{
  "personalInfo": {
    "fullName": "Jane Doe",
    "personalStatement": "Results-driven engineer with 8 years in fintech.",
    "contact": {
      "emails": ["jane.doe@example.com"],
      "phones": ["+44 7700 900123"]
    }
  },
  "workExperience": [
    {
      "employer": "Acme Financial Ltd",
      "jobTitle": "Lead Software Engineer",
      "startDate": 2019,
      "endDate": null,
      "isCurrent": true,
      "description": "Architected real-time payments API handling 2M txn/day.",
      "category": null
    },
    {
      "employer": "Beta Systems",
      "jobTitle": "Software Engineer",
      "startDate": 2015,
      "endDate": 2019,
      "isCurrent": false,
      "description": null,
      "category": null
    }
  ],
  "education": [
    {
      "institution": "University of Edinburgh",
      "qualificationTitle": "MEng Computer Science",
      "startDate": 2011,
      "endDate": 2015,
      "description": "First Class Honours. Thesis: Distributed Consensus Algorithms."
    }
  ],
  "skills": {
    "Languages": ["Python", "Go", "SQL"],
    "Frameworks": ["FastAPI", "Kafka"]
  },
  "socialLinks": ["https://linkedin.com/in/janedoe", "https://github.com/janedoe"],
  "certificationsAndAwards": [
    {
      "description": "AWS Certified Solutions Architect",
      "organization": "Amazon Web Services",
      "date": "2022",
      "category": "Certification"
    }
  ],
  "publications": [],
  "media": [],
  "other": []
}
            ```

            ---

            ## 7. Diff Summary

            The seed prompt was not improved during this run.

            ---

            ## 8. Infrastructure Notes

            **Optimization strategy:** Beam width: **2**.  Train-example injection triggered after stall ≥ **2**.  Secondary beam passed to Mutator after stall ≥ **5**.

            **OCR Caching:** **7** file(s) cached across methods: `{'gemini_ocr': 5, 'pymupdf': 2}`.  
            Extracted PDF text is persisted in `run_state.db` (keyed by SHA-256 of
            file bytes). Re-runs skip re-extraction entirely for unchanged files.

            **LLM Fallback:** When OpenRouter hits its daily free-model quota, agents
            automatically fall back to a local Ollama instance (`OLLAMA_BASE_URL`,
            default `http://localhost:11434`). Set `OLLAMA_MODEL=llama3` for best
            results (significantly better than gemma:2b for JSON extraction tasks).

            ---

            ## 9. Limitations

            - **Small dataset:** With only 2–8 documents per schema, validation scores
              are noisy and there is risk of overfitting to the 1–2 validation documents.
            - **Positional array alignment:** Object arrays (workExperience, education)
              are compared positionally. Ordering differences are penalised.
            - **Free-tier rate limits:** OpenRouter free models have a daily cap (~50/day).
              Ollama fallback mitigates this; `llama3` recommended over `gemma:2b`.
            - **Stochastic metric caching:** `string_semantic` and `array_llm` scores
              are cached per (pred, gold) pair. Initial calls for novel pairs are
              non-deterministic.
