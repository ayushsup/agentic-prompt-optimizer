            # Prompt Optimization Report

            **Generated:** 2026-05-29 19:00:55  
            **Dataset:** `hiring/resume`  
            **Split:** seed=7  
            1 train | 3 val | 3 test  
            **Models:** extractor=`google/gemma-4-31b-it:free`  
            critic=`google/gemma-4-31b-it:free`  mutator=`google/gemma-4-31b-it:free`

            ---

            ## 1. Test-Set Scores

            | Prompt | Test F1 |
            |--------|---------|
            | Seed   | 0.6186 |
            | Final  | 0.5423 |
            | **Δ**  | **-0.0762** |

            Best validation F1 during optimization: **0.5885**

            ---

            ## 2. Per-Subtree Breakdown (Final Prompt, Test Set)

            | Document | Field | Precision | Recall | F1 |
            |----------|-------|-----------|--------|----|
            | Resume-IT | media | 0.000 | 0.000 | 0.000 |
| Resume-IT | other | 0.000 | 0.000 | 0.000 |
| Resume-IT | skills | 0.000 | 0.000 | 0.000 |
| Resume-IT | education | 0.982 | 0.982 | 0.982 |
| Resume-IT | languages | 1.000 | 1.000 | 1.000 |
| Resume-IT | socialLinks | 0.500 | 0.500 | 0.500 |
| Resume-IT | personalInfo | 1.000 | 1.000 | 1.000 |
| Resume-IT | publications | 0.000 | 0.000 | 0.000 |
| Resume-IT | workExperience | 0.985 | 0.985 | 0.985 |
| Resume-IT | certificationsAndAwards | 1.000 | 1.000 | 1.000 |
| Resume-Academic02 | media | 0.000 | 0.000 | 0.000 |
| Resume-Academic02 | other | 0.000 | 0.000 | 0.000 |
| Resume-Academic02 | skills | 0.000 | 0.000 | 0.000 |
| Resume-Academic02 | education | 1.000 | 1.000 | 1.000 |
| Resume-Academic02 | languages | 0.000 | 0.000 | 0.000 |
| Resume-Academic02 | socialLinks | 0.000 | 0.000 | 0.000 |
| Resume-Academic02 | personalInfo | 1.000 | 1.000 | 1.000 |
| Resume-Academic02 | publications | 0.000 | 0.000 | 0.000 |
| Resume-Academic02 | workExperience | 0.897 | 0.897 | 0.897 |
| Resume-Academic02 | certificationsAndAwards | 0.376 | 0.223 | 0.280 |
| Resume-Finance | media | 0.000 | 0.000 | 0.000 |
| Resume-Finance | other | 0.000 | 0.000 | 0.000 |
| Resume-Finance | skills | 0.000 | 0.000 | 0.000 |
| Resume-Finance | education | 0.400 | 0.400 | 0.400 |
| Resume-Finance | languages | 1.000 | 1.000 | 1.000 |
| Resume-Finance | socialLinks | 0.000 | 0.000 | 0.000 |
| Resume-Finance | personalInfo | 1.000 | 1.000 | 1.000 |
| Resume-Finance | publications | 0.000 | 0.000 | 0.000 |
| Resume-Finance | workExperience | 0.703 | 0.703 | 0.703 |
| Resume-Finance | certificationsAndAwards | 0.000 | 0.000 | 0.000 |

            ---

            ## 3. Optimization Trajectory

            | Iter | Val F1 | Accepted |
            |------|--------|----------|
            |   1 | 0.4608 | ✅ |
|   2 | 0.4495 | ❌ |
|   3 | 0.4522 | ❌ |
|   4 | 0.4854 | ✅ |
|   5 | 0.4877 | ✅ |
|   6 | 0.5885 | ✅ |
|   7 | 0.5022 | ❌ |
|   8 | 0.5017 | ❌ |
|   9 | 0.5030 | ❌ |
|  10 | 0.5285 | ❌ |
|  11 | 0.4948 | ❌ |
|  12 | 0.5006 | ❌ |
|  13 | 0.4680 | ❌ |
|  14 | 0.4824 | ❌ |
|  15 | 0.4711 | ❌ |
|  16 | 0.4739 | ❌ |
|  17 | 0.5017 | ❌ |
|  18 | 0.5130 | ❌ |
|  19 | 0.4759 | ❌ |
|  20 | 0.4798 | ❌ |

            ---

            ## 4. Notable Accepted Mutations

            - **Iteration 4** — Val F1: 0.4854  (+12 lines  -13 lines)
- **Iteration 5** — Val F1: 0.4877  (+3 lines  -3 lines)
- **Iteration 6** — Val F1: 0.5885  (+28 lines  -29 lines)

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
     personalInfo  workExperience  education  skills  languages  socialLinks
     certificationsAndAwards  publications  media  other
3. FORBIDDEN root keys: "schema_definition", "name", "data", "result",
   "cv", "resume", "contact", "profile", "summary", "awards", "certifications",
   or any key not listed in rule 2.
4. Absent scalar  → null.   Absent array → [].
5. Never invent data not present in the document. Extract ONLY information
   explicitly stated in the document.
6. Year-only dates → INTEGER  (e.g. 2020, not "2020" and NOT an ISO timestamp).
7. isCurrent → boolean  (true / false, never "true" / "false").

═══════════════════════════════════════════════════════
FIELD RULES
═══════════════════════════════════════════════════════

languages (array of strings):
  All spoken or written languages explicitly listed, with proficiency if stated
  (e.g., "English (Native)", "French (Fluent)"); [] if none

personalInfo  (object, REQUIRED):
  fullName          : exact name as written at top of document
  personalStatement : professional summary / objective paragraph; null if absent
  contact:
    emails : array — all email addresses found anywhere; [] if none
    phones : array — all phone numbers, exact format; [] if none

workExperience  (array of objects, most-recent first, REQUIRED):
  ⚠️  COMPLETENESS IS CRITICAL — extract EVERY role listed in the document.
  For academic CVs this means ALL sections: faculty appointments, research
  positions, visiting roles, postdoctoral positions, teaching assistantships,
  industry roles, and consulting. Do NOT stop early. If the document has 15
  roles, output all 15.
  employer    : exact institution / company name
  jobTitle    : exact title / position
  startDate   : integer year (e.g. 2019) when only year given;
                string (e.g. "Spring 2010") ONLY for non-numeric formats
  endDate     : same format as startDate; null if the role is current
  isCurrent   : true if ongoing/current, false otherwise  (boolean)
  description : responsibilities / achievements text; if bullet points,
                join with semicolons into one string; null if none
  category    : section heading this role falls under (e.g. "Research
                Experience", "Teaching Experience"); null if no headings

education  (array of objects, most-recent first):
  institution       : exact institution name
  qualificationTitle: full degree title (e.g. "PhD in Computer Science")
  startDate         : integer year or string; null if not stated
  endDate           : integer year or string; null if ongoing or not stated
  description       : GPA, thesis title, honours, advisors, distinctions; null if none

skills:
  • Organised under headings → object: { "HeadingName": ["skill1", "skill2"], … }
  • Flat list (no headings)  → array of strings
  • No skills section        → null
  • Split comma-separated items into individual elements
    (e.g., "Tableau, Power BI" → ["Tableau", "Power BI"])

socialLinks  (array of strings):
  All URLs found anywhere (LinkedIn, GitHub, ORCID, ResearchGate, personal
  website); [] if none

certificationsAndAwards  (array of objects):
  ⚠️  COMPLETENESS IS CRITICAL — extract ALL certifications, awards, honours,
  memberships, and affiliations. Licenses go into the `other` field.
  description : exact name as written in the document
  organization: granting body or organisation; null if not stated
  date        : date as a string (e.g. "2022", "March 2021"); null if not stated
  category    : MUST be EXACTLY one of these five strings (case-sensitive):
                "Certification" → professional certs (AWS, CPA, PMP, etc.)
                "Award"         → prizes, scholarships, fellowships
                "Membership"    → professional body memberships
                "Honor"         → dean's list, cum laude, distinctions
                "Affiliation"   → society/association affiliations
                When uncertain, prefer "Award" over leaving blank.

publications  (array of strings):
  ⚠️  COMPLETENESS IS CRITICAL — include EVERY publication listed: journal
  articles, conference papers, book chapters, preprints, technical reports.
  Copy each citation string exactly as written in the document; [] if none

media  (array of strings):
  Media appearances or press mentions as strings; [] if none

other  (array of objects):
  Any section not captured above, including licenses.
  sectionTitle : heading as it appears in the document (e.g. "Licenses")
  content      : full text of that section

═══════════════════════════════════════════════════════
WORKED EXAMPLE  (format reference only — do not copy this data)
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
      "employer": "State University",
      "jobTitle": "Visiting Researcher",
      "startDate": 2017,
      "endDate": 2018,
      "isCurrent": false,
      "description": "Conducted research on distributed consensus algorithms.",
      "category": "Research Experience"
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
    "Programming": ["Python", "Go", "SQL"],
    "Frameworks": ["FastAPI", "Kafka"]
  },
  "languages": ["English (Native)", "Spanish (Conversational)"],
  "socialLinks": ["https://linkedin.com/in/janedoe"],
  "certificationsAndAwards": [
    {
      "description": "AWS Certified Solutions Architect",
      "organization": "Amazon Web Services",
      "date": "2022",
      "category": "Certification"
    }
  ],
  "publications": [
    "Doe, J. (2021). Distributed Consensus in Edge Networks. IEEE Trans. Networking, 29(3), 112-124."
  ],
  "media": [],
  "other": []
}
            ```

            ---

            ## 6. Final Prompt

            ```
            You are a structured data extraction engine. Read the resume/CV document and output ONE JSON object. No markdown, no explanation, no wrapper keys.

═══════════════════════════════════════════════════════
OUTPUT CONTRACT — FOLLOW EXACTLY
═══════════════════════════════════════════════════════
1. Return ONLY raw JSON — no ```json fences, no preamble, no trailing text.
2. The root object MUST contain EXACTLY these top-level keys (never more):
     personalInfo  workExperience  education  skills  languages  socialLinks
     certificationsAndAwards  publications  media  other
3. FORBIDDEN root keys: "schema_definition", "name", "data", "result",
   "cv", "resume", "contact", "profile", "summary", "awards", "certifications",
   or any key not listed in rule 2.
4. Absent scalar  → null.   Absent array → [].
5. Never invent data not present in the document. Extract ONLY information
   explicitly stated in the document.
6. Year-only dates → INTEGER (e.g. 2020, not "2020").
7. isCurrent → STRING ("true" or "false", never boolean true/false).

═══════════════════════════════════════════════════════
FIELD RULES
═══════════════════════════════════════════════════════

languages (array of strings):
  All spoken or written languages explicitly listed, with proficiency if stated
  (e.g., "English (Native)", "French (Fluent)"); [] if none

personalInfo  (object, REQUIRED):
  fullName          : exact name as written at top of document. Include professional titles (e.g. "Dr.") and suffixes (e.g. "MD") if present.
  personalStatement : professional summary / objective paragraph; null if absent
  contact:
    emails : array — all email addresses found anywhere; [] if none
    phones : array — all phone numbers, exact format; [] if none

workExperience  (array of objects, most-recent first, REQUIRED):
  ⚠️  COMPLETENESS IS CRITICAL — extract EVERY role listed in the document.
  For academic CVs this means ALL sections: faculty appointments, research
  positions, visiting roles, postdoctoral positions, teaching assistantships,
  industry roles, and consulting. Do NOT stop early. If the document has 15
  roles, output all 15.
  employer    : exact institution / company name
  jobTitle    : exact title / position, omitting any additional descriptors (e.g. "– Cardiology Department").
  startDate   : integer year (e.g. 2020) or the exact string as written (e.g. "Spring 2010"); null if not stated
  endDate     : integer year (e.g. 2000) or the exact string as written; null if the role is current
  isCurrent   : "true" if ongoing/current, "false" otherwise (MUST be a string)
  description : responsibilities / achievements text; include descriptions for all roles including visiting professor roles. If bullet points,
                join with semicolons into one string; null if none
  category    : section heading this role falls under (e.g. "Research
                Experience", "Teaching Experience"); null if no headings

education  (array of objects, most-recent first):
  institution       : exact institution name
  qualificationTitle: full degree title (e.g. "PhD in Computer Science")
  startDate         : integer year or the exact string as written; null if not stated
  endDate           : integer year or the exact string as written; null if ongoing or not stated
  description       : GPA, thesis title, honours, advisors, distinctions; null if none

skills:
  • Extract as a FLAT ARRAY of strings (e.g., ["Python", "Legal research and analysis"]).
  • Do NOT use a keyed object or headings.
  • Split comma-separated items into individual elements.
  • No skills section        → null
  • Only include skills if explicitly stated in the source.

socialLinks  (array of strings):
  All URLs found anywhere (LinkedIn, GitHub, ORCID, ResearchGate, personal
  website); [] if none

certificationsAndAwards  (array of objects):
  ⚠️  COMPLETENESS IS CRITICAL — extract ALL certifications, awards, honours,
  memberships, and affiliations.
  description : exact name as written in the document
  organization: granting body or organisation; null if not stated
  date        : date as a string (e.g. "2022", "March 2021"); null if not stated
  category    : MUST be EXACTLY one of these five strings (case-sensitive):
                "Certification" → professional certs (AWS, CPA, PMP, etc.)
                "Award"         → prizes, scholarships, fellowships
                "Membership"    → professional body memberships
                "Honor"         → dean's list, cum laude, distinctions
                "Affiliation"   → society/association affiliations
                When uncertain, prefer "Award" over leaving blank.

publications  (array of strings):
  ⚠️  COMPLETENESS IS CRITICAL — include EVERY publication listed: journal
  articles, conference papers, book chapters, preprints, technical reports.
  Copy each citation string exactly as written in the document; [] if none

media  (array of strings):
  Media appearances or press mentions as strings; [] if none

other  (array of objects):
  Any section not captured above, including licenses.
  content      : full text of that section (do NOT include a sectionTitle key)

═══════════════════════════════════════════════════════
WORKED EXAMPLE  (format reference only — do not copy this data)
═══════════════════════════════════════════════════════
{
  "personalInfo": {
    "fullName": "Dr. Jane Doe, MD",
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
      "isCurrent": "true",
      "description": "Architected real-time payments API handling 2M txn/day.",
      "category": null
    },
    {
      "employer": "State University",
      "jobTitle": "Visiting Researcher",
      "startDate": "Spring 2017",
      "endDate": 2018,
      "isCurrent": "false",
      "description": "Conducted research on distributed consensus algorithms.",
      "category": "Research Experience"
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
  "skills": ["Python", "Go", "SQL", "FastAPI", "Kafka"],
  "languages": ["English (Native)", "Spanish (Conversational)"],
  "socialLinks": ["https://linkedin.com/in/janedoe"],
  "certificationsAndAwards": [
    {
      "description": "AWS Certified Solutions Architect",
      "organization": "Amazon Web Services",
      "date": "2022",
      "category": "Certification"
    }
  ],
  "publications": [
    "Doe, J. (2021). Distributed Consensus in Edge Networks. IEEE Trans. Networking, 29(3), 112-124."
  ],
  "media": [],
  "other": [
    {
      "content": "• Riverside State Bar (2011)"
    }
  ]
}
            ```

            ---

            ## 7. Diff Summary

            See `logs/diffs/` for unified diffs of each accepted mutation.

            ---

            ## 8. Infrastructure Notes

            **Optimization strategy:** Beam width: **2**.  Train-example injection after stall ≥ **2**.  Beam secondary after stall ≥ **5**.  Prompt lint with up to **2** mutation retries.

            **LLM Fallback:** Four-tier chain: **OpenRouter** → **GitHub Models** (`gpt-4o`, ✅) → **Gemini text** (`gemini-2.0-flash`, ✅) → **Ollama** (`qwen2.5:3b`).

            **OCR Caching:** **7** file(s) cached across methods: `{'gemini_ocr': 5, 'pymupdf': 2}`.  
            Extracted PDF text is persisted in `run_state.db` (keyed by SHA-256 of
            file bytes). Re-runs skip re-extraction entirely for unchanged files.

            ---

            ## 9. Limitations

            - **Small dataset:** With only 2–8 documents per schema, validation scores
              are noisy. The 3-doc val set (seed=7) mitigates but does not eliminate
              overfitting risk.
            - **Positional array alignment:** Object arrays (workExperience, education)
              are compared positionally. Ordering differences are penalised even when
              content matches.
            - **Free-tier rate limits:** Each tier has its own daily cap. The four-tier
              chain significantly extends total available quota before any stall.
            - **Stochastic metric caching:** `string_semantic` and `array_llm` scores
              are cached per (pred, gold) pair. Initial calls for novel pairs are
              non-deterministic.
            - **Prompt linting:** The linter catches ISO timestamp and languages
              inconsistency regressions. Other regression classes (e.g. field removal)
              are caught by the val score drop but still consume one iteration.
