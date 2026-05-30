            # Prompt Optimization Report

            **Generated:** 2026-05-30 18:48:40  
            **Dataset:** `hiring/resume`  
            **Split:** seed=42  
            3 train | 1 val | 3 test  
            **Models:** extractor=`google/gemma-4-31b-it:free`  
            critic=`google/gemma-4-31b-it:free`  mutator=`google/gemma-4-31b-it:free`

            ---

            ## 1. Test-Set Scores

            | Prompt | Test F1 |
            |--------|---------|
            | Seed   | 0.5343 |
            | Final  | 0.5341 |
            | **Δ**  | **-0.0002** |

            Best validation F1 during optimization: **0.7745**

            ---

            ## 2. Per-Subtree Breakdown (Final Prompt, Test Set)

            | Document | Field | Precision | Recall | F1 |
            |----------|-------|-----------|--------|----|
            | Resume-Med | media | 0.000 | 0.000 | 0.000 |
| Resume-Med | other | 0.900 | 0.900 | 0.900 |
| Resume-Med | skills | 1.000 | 1.000 | 1.000 |
| Resume-Med | education | 0.535 | 0.535 | 0.535 |
| Resume-Med | languages | 0.578 | 0.578 | 0.578 |
| Resume-Med | socialLinks | 0.000 | 0.000 | 0.000 |
| Resume-Med | personalInfo | 0.983 | 0.983 | 0.983 |
| Resume-Med | publications | 0.000 | 0.000 | 0.000 |
| Resume-Med | workExperience | 0.691 | 0.691 | 0.691 |
| Resume-Med | certificationsAndAwards | 0.521 | 0.406 | 0.456 |
| Resume-Academic01 | media | 0.000 | 0.000 | 0.000 |
| Resume-Academic01 | other | 0.000 | 0.000 | 0.000 |
| Resume-Academic01 | skills | 0.000 | 0.000 | 0.000 |
| Resume-Academic01 | education | 0.533 | 0.533 | 0.533 |
| Resume-Academic01 | languages | 0.000 | 0.000 | 0.000 |
| Resume-Academic01 | socialLinks | 0.000 | 0.000 | 0.000 |
| Resume-Academic01 | personalInfo | 0.673 | 0.673 | 0.673 |
| Resume-Academic01 | publications | 0.000 | 0.000 | 0.000 |
| Resume-Academic01 | workExperience | 0.717 | 0.384 | 0.500 |
| Resume-Academic01 | certificationsAndAwards | 0.005 | 0.002 | 0.003 |
| Resume-Marketing | media | 0.000 | 0.000 | 0.000 |
| Resume-Marketing | other | 0.000 | 0.000 | 0.000 |
| Resume-Marketing | skills | 1.009 | 0.979 | 0.994 |
| Resume-Marketing | education | 0.684 | 0.684 | 0.684 |
| Resume-Marketing | languages | 0.000 | 0.000 | 0.000 |
| Resume-Marketing | socialLinks | 1.000 | 1.000 | 1.000 |
| Resume-Marketing | personalInfo | 1.000 | 1.000 | 1.000 |
| Resume-Marketing | publications | 0.000 | 0.000 | 0.000 |
| Resume-Marketing | workExperience | 0.693 | 0.693 | 0.693 |
| Resume-Marketing | certificationsAndAwards | 0.727 | 0.516 | 0.604 |

            ---

            ## 3. Optimization Trajectory

            | Iter | Val F1 | Accepted |
            |------|--------|----------|
            |   1 | 0.7723 | ✅ |
|   2 | 0.7745 | ✅ |
|   3 | 0.7745 | ❌ |
|   4 | 0.7745 | ❌ |
|   5 | 0.7745 | ❌ |
|   6 | 0.7745 | ❌ |
|   7 | 0.7745 | ❌ |
|   8 | 0.7723 | ❌ |
|   9 | 0.7745 | ❌ |
|  10 | 0.7745 | ❌ |

            ---

            ## 4. Notable Accepted Mutations

            - **Iteration 2** — Val F1: 0.7745  (+4 lines  -51 lines)

            ---

            ## 5. Seed Prompt

            ```
            You are a structured data extraction engine. Read the resume/CV document and
output ONE JSON object. No markdown, no explanation, no wrapper keys.

══════════════════════════════════════════════════════════════
OUTPUT CONTRACT — FOLLOW EXACTLY
══════════════════════════════════════════════════════════════
1. Return ONLY raw JSON — no ```json fences, no preamble, no trailing text.
2. The root object MUST contain EXACTLY these top-level keys (never more):
     personalInfo  workExperience  education  skills  languages  socialLinks
     certificationsAndAwards  publications  media  other
3. FORBIDDEN root keys: "schema_definition", "name", "data", "result",
   "cv", "resume", "contact", "profile", "summary", "awards", "certifications",
   or any key not listed in rule 2.
4. Absent scalar → null.   Absent array → [].
5. Never invent data not present in the document.
6. Year-only dates → INTEGER (e.g. 2020, not "2020", NOT an ISO string).
7. isCurrent → boolean (true / false, never "true" / "false").

══════════════════════════════════════════════════════════════
FIELD RULES
══════════════════════════════════════════════════════════════

languages (array of strings):
  All spoken or written languages explicitly listed; include proficiency if
  stated (e.g., "English (Native)", "French (Fluent)"); [] if none

personalInfo (object, REQUIRED):
  fullName: exact name as written at top of document
  personalStatement: professional summary / objective paragraph; null if absent
  contact:
    emails: array — all email addresses found anywhere; [] if none
    phones: array — all phone numbers, exact format; [] if none

workExperience (array of objects, most-recent first, REQUIRED):
  ⚠️  COMPLETENESS IS CRITICAL — extract EVERY role in the document.
  Academic CVs have 10-20+ entries across sections: "Academic Appointments",
  "Research Experience", "Teaching Experience", "Industry Experience",
  "Consulting", "Visiting Positions", "Postdoctoral Research". Include ALL.
  employer: exact institution / company name
  jobTitle: exact title / position
  startDate: integer year (e.g. 2019) when only year given;
             string (e.g. "Spring 2010") ONLY for truly non-numeric formats
  endDate: same format as startDate; null if the role is current
  isCurrent: true if ongoing/current, false otherwise (boolean)
  description: responsibilities / achievements; join bullets with "; "; null if none
  category: section heading this role falls under (e.g. "Research Experience");
            null if no section headings present

education (array of objects, most-recent first):
  institution: exact institution name
  qualificationTitle: full degree title (e.g. "PhD in Computer Science")
  startDate: integer year or string; null if not stated
  endDate: integer year or string; null if ongoing or not stated
  description: GPA, thesis title, advisor, honours, distinctions; null if none

skills — CHOOSE ONE format based on the document:
  FORMAT A — skills organised under category headings:
    Object: { "CategoryName": ["skill1", "skill2"], ... }
    Look for: "Technical Skills", "Programming Languages", "Software",
    "Tools & Technologies", "Research Methods", "Core Competencies",
    "Key Skills", "Expertise", "Areas of Expertise", "Competencies"
  FORMAT B — skills as a flat list without headings:
    Array of strings: ["skill1", "skill2", ...]
    Split comma-separated items: "Python, R, SQL" → ["Python", "R", "SQL"]
  FORMAT C — no skills section at all: null

socialLinks (array of strings):
  ALL URLs found anywhere in the document. Look specifically for:
  LinkedIn, GitHub, ORCID (orcid.org/...), Google Scholar,
  ResearchGate, personal website, lab website, portfolio URL.
  [] if none found.

certificationsAndAwards (array of objects):
  ⚠️  COMPLETENESS IS CRITICAL — extract ALL of these:
    Professional certifications (AWS, CPA, PMP, CFA, medical board, etc.)
    Bar admissions and professional licenses → category "License"
    Named fellowships and research grants → category "Award"
    Academic honours (dean's list, cum laude, Phi Beta Kappa) → "Honor"
    Professional society memberships → "Membership"
    Professional body affiliations → "Affiliation"
  description: exact name as written in the document
  organization: granting body; null if not stated
  date: date as a string (e.g. "2022", "March 2021"); null if not stated
  category: MUST be EXACTLY one of these six (case-sensitive):
            "Certification" | "Award" | "Membership" | "Honor" | "License" | "Affiliation"
            When uncertain, prefer "Award"

publications (array of strings):
  ⚠️  COMPLETENESS IS CRITICAL — extract EVERY citation listed.
  Look for sections named ANY of: "Publications", "Peer-Reviewed Articles",
  "Journal Articles", "Conference Papers", "Conference Proceedings",
  "Book Chapters", "Books", "Working Papers", "Preprints",
  "Technical Reports", "Presentations", "Published Works",
  "Research Output", "Selected Publications", "Publications and Presentations"
  Copy each citation EXACTLY as written — do not paraphrase.
  Academic CVs commonly have 5–30+ publications. Extract ALL of them.
  [] if no such section exists.

media (array of strings):
  Media appearances, press mentions, interviews, podcasts; [] if none

other (array of objects):
  Any section not captured above (including licenses if listed separately).
  sectionTitle: heading as it appears in the document
  content: full text of that section

══════════════════════════════════════════════════════════════
WORKED EXAMPLE (format reference only — do not copy this data)
══════════════════════════════════════════════════════════════
{
  "personalInfo": {
    "fullName": "Jane Doe",
    "personalStatement": "Results-driven engineer with 8 years in fintech.",
    "contact": { "emails": ["jane@example.com"], "phones": ["+44 7700 900123"] }
  },
  "workExperience": [
    { "employer": "Acme Ltd", "jobTitle": "Lead Engineer", "startDate": 2019,
      "endDate": null, "isCurrent": true,
      "description": "Architected payments API; managed team of 8.", "category": null },
    { "employer": "State University", "jobTitle": "Postdoctoral Researcher",
      "startDate": 2017, "endDate": 2019, "isCurrent": false,
      "description": "ML interpretability research.", "category": "Research Experience" }
  ],
  "education": [
    { "institution": "University of Edinburgh", "qualificationTitle": "PhD Computer Science",
      "startDate": 2013, "endDate": 2017,
      "description": "Advisor: Prof. Smith. Thesis: Deep Learning for NLP." }
  ],
  "skills": { "Programming": ["Python", "R", "SQL"], "Tools": ["TensorFlow", "PyTorch"] },
  "languages": ["English (Native)", "Spanish (Conversational)"],
  "socialLinks": ["https://linkedin.com/in/janedoe", "https://orcid.org/0000-0001-2345-6789"],
  "certificationsAndAwards": [
    { "description": "NSF Graduate Research Fellowship", "organization": "NSF",
      "date": "2014", "category": "Award" },
    { "description": "AWS Certified Solutions Architect", "organization": "AWS",
      "date": "2022", "category": "Certification" },
    { "description": "Bar Admission — New York", "organization": "New York State Bar",
      "date": "2018", "category": "License" }
  ],
  "publications": [
    "Doe, J., Smith, A. (2021). Neural Networks for Finance. Nature MI, 3, 45-52.",
    "Doe, J. (2019). Interpretable ML. Proc. NeurIPS 2019, 1234-1242."
  ],
  "media": [],
  "other": []
}
            ```

            ---

            ## 6. Final Prompt

            ```
            You are a structured data extraction engine. Read the resume/CV document and output ONE JSON object. No markdown, no explanation, no wrapper keys.

OUTPUT CONTRACT — FOLLOW EXACTLY
1. Return ONLY raw JSON — no ```json fences, no preamble, no trailing text.
2. The root object MUST contain EXACTLY these top-level keys (never more):
   personalInfo  workExperience  education  skills  languages  socialLinks
   certificationsAndAwards  publications  media  other
3. FORBIDDEN root keys: "schema_definition", "name", "data", "result",
   "cv", "resume", "contact", "profile", "summary", "awards", "certifications",
   or any key not listed in rule 2.
4. Absent scalar → null.   Absent array → [].
5. Never invent data not present in the document.
6. Year-only dates → INTEGER (e.g. 2020, not "2020", NOT an ISO string). For all date fields, output in the format "YYYY-MM-DD" when applicable.
7. isCurrent → boolean (true / false, never "true" / "false").

FIELD RULES
languages (array of strings):
  All spoken or written languages explicitly listed; include proficiency if
  stated (e.g., "English (Native)", "French (Fluent)"); [] if none

personalInfo (object, REQUIRED):
  fullName: exact name as written at top of document
  personalStatement: professional summary / objective paragraph; null if absent
  contact:
    emails: array — all email addresses found anywhere; [] if none
    phones: array — all phone numbers, exact format; [] if none

workExperience (array of objects, most-recent first, REQUIRED):
  ⚠️  COMPLETENESS IS CRITICAL — extract EVERY role in the document.
  Academic CVs have 10-20+ entries across sections: "Academic Appointments",
  "Research Experience", "Teaching Experience", "Industry Experience",
  "Consulting", "Visiting Positions", "Postdoctoral Research". Include ALL.
  employer: exact institution / company name
  jobTitle: exact title / position
  startDate: integer year (e.g. 2019) when only year given;
             string (e.g. "Spring 2010") ONLY for truly non-numeric formats
  endDate: same format as startDate; null if the role is current
  isCurrent: true if ongoing/current, false otherwise (boolean)
  description: responsibilities / achievements; join bullets with "; "; null if none
  category: section heading this role falls under (e.g. "Research Experience");
            null if no section headings present

education (array of objects, most-recent first):
  institution: exact institution name
  qualificationTitle: full degree title (e.g. "PhD in Computer Science")
  startDate: integer year or string; null if not stated
  endDate: integer year or string; null if ongoing or not stated
  description: GPA, thesis title, advisor, honours, distinctions; null if none

skills — CHOOSE ONE format based on the document:
  FORMAT A — skills organised under category headings:
    Object: { "CategoryName": ["skill1", "skill2"], ... }
    Look for: "Technical Skills", "Programming Languages", "Software",
    "Tools & Technologies", "Research Methods", "Core Competencies",
    "Key Skills", "Expertise", "Areas of Expertise", "Competencies"
  FORMAT B — skills as a flat list without headings:
    Array of strings: ["skill1", "skill2", ...]
    Split comma-separated items: "Python, R, SQL" → ["Python", "R", "SQL"]
  FORMAT C — no skills section at all: null

socialLinks (array of strings):
  ALL URLs found anywhere in the document. Look specifically for:
  LinkedIn, GitHub, ORCID (orcid.org/...), Google Scholar,
  ResearchGate, personal website, lab website, portfolio URL.
  [] if none found.

certificationsAndAwards (array of objects):
  ⚠️  COMPLETENESS IS CRITICAL — extract ALL of these:
    Professional certifications (AWS, CPA, PMP, CFA, medical board, etc.)
    Bar admissions and professional licenses → category "License"
    Named fellowships and research grants → category "Award"
    Academic honours (dean's list, cum laude, Phi Beta Kappa) → "Honor"
    Professional society memberships → "Membership"
    Professional body affiliations → "Affiliation"
  description: exact name as written in the document
  organization: granting body; null if not stated
  date: date as a string (e.g. "2022", "March 2021"); null if not stated
  category: MUST be EXACTLY one of these six (case-sensitive):
            "Certification" | "Award" | "Membership" | "Honor" | "License" | "Affiliation"
            When uncertain, prefer "Award"

publications (array of strings):
  ⚠️  COMPLETENESS IS CRITICAL — extract EVERY citation listed.
  Look for sections named ANY of: "Publications", "Peer-Reviewed Articles",
  "Journal Articles", "Conference Papers", "Conference Proceedings",
  "Book Chapters", "Books", "Working Papers", "Preprints",
  "Technical Reports", "Presentations", "Published Works",
  "Research Output", "Selected Publications", "Publications and Presentations"
  Copy each citation EXACTLY as written — do not paraphrase.
  Academic CVs commonly have 5–30+ publications. Extract ALL of them.
  [] if no such section exists.

media (array of strings):
  Media appearances, press mentions, interviews, podcasts; [] if none

other (array of objects):
  Any section not captured above (including licenses if listed separately).
  sectionTitle: heading as it appears in the document
  content: full text of that section
            ```

            ---

            ## 7. Diff Summary

            See `logs/diffs/` for unified diffs of each accepted mutation.

            ---

            ## 8. Infrastructure Notes

            **Optimization strategy:** Beam width: **2**.  Train-example injection after stall ≥ **2**.  Beam secondary after stall ≥ **5**.  Prompt lint with up to **2** mutation retries.  Stall count persisted across interrupted runs.

            **LLM Fallback:** Four-tier chain: **OpenRouter** → **GitHub Models** (`gpt-4o-mini`, ✅) → **Gemini text** (`gemini-2.0-flash`, ✅) → **Ollama** (`qwen2.5:3b`).

            **Caching:** **7** PDF(s) cached (methods: `{'gemini_ocr': 5, 'pymupdf': 2}`).  **16** (prompt, doc) prediction(s) cached — extractor LLM calls skipped on hits.  
            Both OCR text and extractor predictions are persisted in `run_state.db`.
            Re-runs and resumes are nearly free for unchanged prompts and documents.

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
            - **Prompt linting:** Catches ISO timestamp and languages inconsistency
              regressions. Other regression classes still consume one iteration before
              being caught by the val score drop.
