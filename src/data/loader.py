"""
ExtractBench dataset loader with multi-strategy PDF extraction and OCR caching.

Extraction priority order
--------------------------
1. OCR cache (SQLite)  — instant replay; avoids all API calls for known files
2. pymupdf (fitz)      — fastest, handles most digital PDFs
3. pdfplumber          — better for table-heavy / columnar layouts
4. PyPDF2              — pure-Python last resort

When all text-based methods return empty (image-only / scanned PDFs):
5. Gemini 2.5 Flash via Google AI Studio (GEMINI_API_KEY)
   — highest quality OCR, native PDF understanding, no page rendering needed
6. OpenRouter vision model (OPENROUTER_API_KEY + vision_model in config)
   — fallback if Gemini is not configured

OCR Caching
-----------
Extracted text is persisted in the SQLite StateManager (run_state.db).
The cache key is the SHA-256 of the PDF file bytes, so replacing a file
with a new version automatically bypasses the stale entry.
On a fresh run all PDFs are extracted once; subsequent runs (and resumed
interrupted runs) skip re-extraction entirely, saving time and API quota.

Gemini is prioritised over OpenRouter for vision because:
  - It accepts the raw PDF bytes (no page-by-page PNG rendering needed)
  - gemini-2.5-flash has excellent OCR quality on resume PDFs
  - The AI Studio free tier is generous for this use case
"""

from __future__ import annotations

import base64
import os
import time
from typing import Dict, List, Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# Text-layer extractors
# ---------------------------------------------------------------------------

def _try_pymupdf(pdf_path: str) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    text = "\n".join(doc[i].get_text("text") for i in range(len(doc)))
    doc.close()
    return text.strip()


def _try_pdfplumber(pdf_path: str) -> str:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()


def _try_pypdf2(pdf_path: str) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


# ---------------------------------------------------------------------------
# Gemini OCR (primary vision fallback — uses raw PDF, no page rendering)
# ---------------------------------------------------------------------------

def _try_gemini_ocr(pdf_path: str, gemini_key: str) -> str:
    """
    Send the PDF directly to Gemini 2.5 Flash for native PDF understanding.
    Passes the file path (not raw bytes) to the Google File API.
    """
    try:
        from google import genai as google_genai

        client = google_genai.Client(api_key=gemini_key)

        uploaded = client.files.upload(
            file=pdf_path,
            config={"mime_type": "application/pdf"},
        )
        time.sleep(2)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                uploaded,
                (
                    "Extract ALL text from this document exactly as it appears. "
                    "Preserve section headings, bullet points, and the document structure. "
                    "Return only the raw text — no commentary, no formatting labels."
                ),
            ],
        )

        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

        return (response.text or "").strip()

    except ImportError:
        print("    ⚠️  google-genai not installed. Run: pip install google-genai")
        return ""
    except Exception as exc:
        print(f"    ⚠️  Gemini OCR failed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# OpenRouter vision fallback (page-by-page PNG rendering)
# ---------------------------------------------------------------------------

def _try_openrouter_vision(
    pdf_path: str,
    api_key: str,
    vision_model: str,
) -> str:
    """
    Render each PDF page as a PNG and send to a vision-capable model on
    OpenRouter. Tries multiple free vision models in sequence if the
    primary model returns a 404 or rate-limit error.
    """
    FALLBACK_MODELS = [
        vision_model,
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "qwen/qwen2-vl-7b-instruct:free",
        "microsoft/phi-3.5-vision-instruct:free",
    ]
    seen: set = set()
    ordered_models = []
    for m in FALLBACK_MODELS:
        if m not in seen:
            seen.add(m)
            ordered_models.append(m)

    try:
        import fitz
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        doc = fitz.open(pdf_path)
        page_texts: List[str] = []

        for page_num, page in enumerate(doc):
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

            extracted = False
            for model in ordered_models:
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        "Extract ALL text from this document page exactly as it appears. "
                                        "Return only the raw text — no labels or commentary."
                                    ),
                                },
                            ],
                        }],
                        max_tokens=2048,
                        temperature=0.0,
                    )
                    text = resp.choices[0].message.content or ""
                    if text.strip():
                        page_texts.append(text)
                        extracted = True
                        print(f"    ✅ Vision OCR page {page_num+1} via {model}")
                        time.sleep(3)
                        break
                except Exception as exc:
                    err = str(exc)
                    if "404" in err:
                        print(f"    ⚠️  {model}: not available — trying next.")
                    elif "429" in err:
                        print(f"    ⚠️  {model}: rate limited — trying next.")
                        time.sleep(5)
                    else:
                        print(f"    ⚠️  {model}: {err[:100]}")

            if not extracted:
                print(f"    ❌ All vision models failed for page {page_num+1}.")

        doc.close()
        return "\n\n".join(page_texts).strip()

    except Exception as exc:
        print(f"    ⚠️  OpenRouter vision OCR crashed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Unified per-file extraction (no caching at this layer)
# ---------------------------------------------------------------------------

_MIN_CONTENT_LEN = 50  # Minimum chars to consider extraction successful


def extract_text_from_pdf(
    pdf_path: str,
    gemini_key: Optional[str] = None,
    openrouter_key: Optional[str] = None,
    vision_model: Optional[str] = None,
) -> tuple[str, str]:
    """
    Extract all text from a PDF using a progressive fallback chain.

    Returns
    -------
    (text, method_used) where method_used is a short label describing
    which strategy succeeded (e.g. "pymupdf", "gemini_ocr", "openrouter_vision").
    Empty string and "failed" are returned if all strategies fail.

    Text-based strategies are tried first (fast, no API cost).
    Vision/OCR strategies are only invoked if all text methods fail.
    """
    filename = os.path.basename(pdf_path)

    for name, fn in [
        ("pymupdf",    lambda: _try_pymupdf(pdf_path)),
        ("pdfplumber", lambda: _try_pdfplumber(pdf_path)),
        ("PyPDF2",     lambda: _try_pypdf2(pdf_path)),
    ]:
        try:
            text = fn()
            if text and len(text) >= _MIN_CONTENT_LEN:
                return text, name
        except Exception as exc:
            print(f"    [{name}] error on {filename}: {exc}")

    # All text methods returned empty → need OCR
    print(f"    🔍 Text extraction empty for {filename}. Trying OCR…")

    # Gemini 2.5 Flash — best quality, uses raw PDF (no page rendering)
    # NOTE: Gemini is always used as the vision/OCR path even when Ollama
    # is configured as the LLM fallback — OCR and inference are separate concerns.
    if gemini_key:
        text = _try_gemini_ocr(pdf_path, gemini_key)
        if text and len(text) >= _MIN_CONTENT_LEN:
            print(f"    ✅ Gemini OCR succeeded for {filename}.")
            return text, "gemini_ocr"

    # OpenRouter vision model — fallback
    if openrouter_key and vision_model:
        text = _try_openrouter_vision(pdf_path, openrouter_key, vision_model)
        if text and len(text) >= _MIN_CONTENT_LEN:
            return text, "openrouter_vision"

    if not gemini_key and not openrouter_key:
        print(
            f"    ⚠️  {filename}: scanned PDF detected but no OCR keys configured.\n"
            "    Set GEMINI_API_KEY (recommended) or OPENROUTER_API_KEY in your environment."
        )

    return "", "failed"


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class ExtractBenchLoader:
    """
    Loads (PDF text, gold JSON, schema) triples from an ExtractBench bundle.

    Directory layout:
      <base_path>/<schema_name>/
        ├── <name>-schema.json        ← JSON Schema with per-field evaluation_config
        └── pdf+gold/
            ├── document.pdf
            └── document.gold.json

    OCR caching
    -----------
    If a StateManager is injected, extracted PDF text is persisted in the
    run_state.db SQLite database.  On subsequent runs (including resumes)
    the loader checks the cache first and skips re-extraction for files
    whose bytes have not changed.  This saves significant time and API
    quota when iterating over multiple optimization runs.
    """

    def __init__(
        self,
        base_path: str,
        schema_name: str,
        vision_model: Optional[str] = None,
        openrouter_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        state_manager=None,  # Optional[StateManager] — avoids circular import
    ):
        self.schema_dir   = os.path.join(base_path, *schema_name.split("/"))
        self.pdf_gold_dir = os.path.join(self.schema_dir, "pdf+gold")
        self.vision_model   = vision_model
        self.openrouter_key = openrouter_key
        self.gemini_key     = gemini_key
        self.state_manager  = state_manager

    def _load_schema(self) -> str:
        if not os.path.isdir(self.schema_dir):
            raise FileNotFoundError(
                f"Schema directory not found: {self.schema_dir}\n"
                "Check base_path and schema_name in your config."
            )
        candidates = [f for f in os.listdir(self.schema_dir) if f.endswith("-schema.json")]
        if not candidates:
            raise FileNotFoundError(f"No *-schema.json found in: {self.schema_dir}")
        path = os.path.join(self.schema_dir, candidates[0])
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_with_cache(self, pdf_path: str) -> str:
        """
        Extract text from a PDF, using the SQLite OCR cache when available.

        Cache hit  → return stored text immediately (no API call).
        Cache miss → run the full extraction chain, then persist the result.
        """
        filename = os.path.basename(pdf_path)

        # --- Cache lookup ---
        if self.state_manager is not None:
            cached = self.state_manager.get_ocr_cache(pdf_path)
            if cached is not None:
                print(f"  💾 OCR cache hit for {filename} — skipping extraction.")
                return cached

        # --- Full extraction ---
        text, method = extract_text_from_pdf(
            pdf_path,
            gemini_key=self.gemini_key,
            openrouter_key=self.openrouter_key,
            vision_model=self.vision_model,
        )

        # --- Persist on success ---
        if text and self.state_manager is not None:
            self.state_manager.set_ocr_cache(pdf_path, text, method=method)
            print(f"  💾 OCR result cached for {filename} (method={method}).")

        return text

    def load_all_document_pairs(self) -> List[Dict]:
        if not os.path.isdir(self.pdf_gold_dir):
            raise FileNotFoundError(
                f"pdf+gold directory not found: {self.pdf_gold_dir}\n"
                "Run: git clone https://github.com/ContextualAI/extract-bench.git data/extract-bench"
            )

        schema    = self._load_schema()
        docs:  List[Dict] = []
        pdf_files = sorted(f for f in os.listdir(self.pdf_gold_dir) if f.endswith(".pdf"))

        # Show cache stats if we have a state manager
        if self.state_manager is not None:
            stats = self.state_manager.get_ocr_cache_stats()
            if stats["total_cached"] > 0:
                print(
                    f"  📦 OCR cache: {stats['total_cached']} file(s) cached "
                    f"({stats['by_method']})"
                )

        for pdf_file in pdf_files:
            base_name  = pdf_file[:-4]
            gold_path  = os.path.join(self.pdf_gold_dir, f"{base_name}.gold.json")
            pdf_path   = os.path.join(self.pdf_gold_dir, pdf_file)

            if not os.path.exists(gold_path):
                print(f"  ⚠️  No gold file for {pdf_file} — skipping.")
                continue

            with open(gold_path, "r", encoding="utf-8") as f:
                gold_json = f.read()

            print(f"  Loading {pdf_file}…")
            text = self._extract_with_cache(pdf_path)

            if not text:
                print(f"  ❌ {pdf_file}: could not extract text. Skipping.")
                continue

            docs.append({"id": base_name, "text": text, "gold_json": gold_json, "schema": schema})

        print(f"\n  ✅ Loaded {len(docs)}/{len(pdf_files)} document pairs from: {self.schema_dir}")
        return docs