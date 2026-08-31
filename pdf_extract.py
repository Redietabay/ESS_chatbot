"""
Shared PDF extraction helpers — text layer + OCR fallback + table extraction.

This used to live only inside index_pdfs.py. It's pulled out here so that
BOTH the bulk indexer (index_pdfs.py, for the permanent ESS corpus) and the
new ad-hoc upload feature (app.py's /upload_document route, for a document a
user attaches in chat) use the exact same extraction logic. Before this,
adding upload support would have meant copy-pasting the OCR/table code a
second time — any future fix (e.g. a Tesseract language tweak) would then
need to be made twice, and would eventually drift apart.

Importing this module has NO side effects (no ChromaDB client, no model
load, no DB connection) — safe to import from anywhere.
"""
import io
import logging
import shutil

import pytesseract
from PIL import Image

log = logging.getLogger("pdf_extract")

# ═══════════════════════════════════════
# CONFIGURATION — identical defaults to the old index_pdfs.py values
# ═══════════════════════════════════════
OCR_ENABLED  = True
OCR_LANG     = "amh+eng"
OCR_DPI      = 300
MIN_TEXT_LEN = 50

MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2


# ═══════════════════════════════════════
# OCR AVAILABILITY CHECK
# ═══════════════════════════════════════
def check_ocr_available() -> bool:
    """Verify the Tesseract binary + requested language packs are actually
    installed before relying on OCR. Cache the result at call level — the
    caller decides how often to re-check (bulk indexer: once per run;
    upload route: fine to re-check every time, it's cheap)."""
    if not OCR_ENABLED:
        return False

    if shutil.which("tesseract") is None:
        log.warning(
            "Tesseract binary not found on PATH — OCR is disabled. "
            "Windows: install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and make sure 'Add to PATH' is checked, or add the install dir manually. "
            "Linux: sudo apt-get install tesseract-ocr tesseract-ocr-amh"
        )
        return False

    try:
        available_langs = set(pytesseract.get_languages(config=""))
    except Exception as e:
        log.warning(f"Could not query installed Tesseract languages ({e}); OCR disabled.")
        return False

    requested_langs = set(OCR_LANG.split("+"))
    missing = requested_langs - available_langs
    if missing:
        log.warning(
            f"Tesseract is missing language pack(s): {sorted(missing)}. "
            f"Download the .traineddata file(s) from "
            f"https://github.com/tesseract-ocr/tessdata and place them in "
            f"your Tesseract install's 'tessdata' folder. OCR is disabled until then."
        )
        return False

    return True


# ═══════════════════════════════════════
# PER-PAGE EXTRACTION
# ═══════════════════════════════════════
def extract_text_from_page(page) -> str:
    """page is a PyMuPDF (fitz) page object."""
    return page.get_text().strip()


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Lightweight, dependency-free preprocessing that measurably helps
    Tesseract's accuracy — especially on Amharic (Ge'ez script), where the
    model has less margin for error than English to begin with. Grayscale +
    autocontrast gives Tesseract's own internal thresholding step a cleaner
    input on the faint/uneven scans that make up most of the ESS corpus.

    This does NOT close the underlying gap: Tesseract's Amharic model is
    simply less mature than its English one (see OCR_LANG comment below) —
    that's a library limitation, not something preprocessing can fix. What
    this DOES fix is the *additional* error on top of that baseline caused
    by low-contrast/low-quality scans, which in practice is the larger and
    more fixable source of garbled Amharic OCR on real ESS reports.

    For a bigger jump in Amharic accuracy than preprocessing alone can give,
    swap the installed 'amh.traineddata' for the "best" (LSTM, higher
    accuracy, slower) model from
    https://github.com/tesseract-ocr/tessdata_best — download amh.traineddata
    there and drop it into Tesseract's tessdata folder, replacing the
    default "fast" one. One-time server-side change, no code change needed.
    """
    from PIL import ImageOps
    gray = img.convert("L")
    return ImageOps.autocontrast(gray, cutoff=1)


def extract_text_with_ocr(page, ocr_available: bool, dpi: int = OCR_DPI, lang: str = OCR_LANG) -> str:
    """OCR fallback for a page with no usable embedded text layer.
    Returns "" if OCR is unavailable or fails.

    --psm 6 ("assume a single uniform block of text") is used instead of
    Tesseract's default --psm 3. This does NOT recover real table
    structure on a scanned page — see extract_tables_from_page's docstring
    and the module-level comment in index_pdfs.py: pdfplumber only sees a
    PDF's own vector/ruling lines, so a scanned table has nothing for it to
    parse, full stop. What --psm 6 DOES do is stop Tesseract from merging
    table rows into one run-on paragraph, which is its default behavior
    under --psm 3 on a busy page. The result is OCR'd text where each row
    roughly stays on its own line — not a real table, but far more usable
    to a reader (or the LLM answering from it) than one long blob of text.
    """
    if not ocr_available:
        return ""
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img = _preprocess_for_ocr(img)
        text = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
        return text.strip()
    except Exception as e:
        log.warning(f"   OCR failed on this page: {e}")
        return ""


def _clean_cell(cell) -> str:
    text = (cell or "").strip().replace("\n", " ")
    return text.replace("|", "\\|")


def table_to_markdown(table: list) -> str:
    """Convert a pdfplumber table (list of row lists) into a Markdown table."""
    rows = [[_clean_cell(c) for c in row] for row in table]
    header, body = rows[0], rows[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[:len(header)]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def extract_tables_from_page(pdfplumber_page) -> list:
    """Returns a list of Markdown-table strings for every table on this page
    that meets the MIN_TABLE_ROWS/MIN_TABLE_COLS bar."""
    markdown_tables = []
    try:
        raw_tables = pdfplumber_page.extract_tables()
    except Exception as e:
        log.warning(f"   pdfplumber table extraction failed on this page: {e}")
        return markdown_tables

    for table in raw_tables:
        if not table or len(table) < MIN_TABLE_ROWS:
            continue
        if not table[0] or len(table[0]) < MIN_TABLE_COLS:
            continue
        if not any(cell and cell.strip() for row in table for cell in row):
            continue
        markdown_tables.append(table_to_markdown(table))

    return markdown_tables


# ═══════════════════════════════════════
# WHOLE-DOCUMENT EXTRACTION (used by the /upload_document route)
# ═══════════════════════════════════════
def extract_document(file_bytes: bytes, max_pages: int = 60) -> dict:
    """
    Extracts full text + tables from an in-memory PDF, page by page, with the
    same text -> OCR fallback -> table extraction pipeline as the bulk
    indexer. Returns a dict instead of writing to ChromaDB — the caller
    decides what to do with the text (e.g. hold it in a session for Q&A).

    {
      "text": str,              # full extracted text, pages joined, tables inlined as Markdown
      "pages_total": int,
      "pages_ocrd": int,
      "pages_empty": int,       # no text layer AND OCR unavailable/failed
      "tables_found": int,
      "ocr_available": bool,
      "truncated": bool,        # True if max_pages cut the document short
    }
    """
    import fitz
    import pdfplumber

    ocr_available = check_ocr_available()

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    plumber_doc = pdfplumber.open(io.BytesIO(file_bytes))

    pages_total = len(doc)
    truncated = pages_total > max_pages
    limit = min(pages_total, max_pages)

    parts = []
    pages_ocrd = 0
    pages_empty = 0
    tables_found = 0

    try:
        for page_num in range(limit):
            text = extract_text_from_page(doc[page_num])

            if len(text) < MIN_TEXT_LEN:
                ocr_text = extract_text_with_ocr(doc[page_num], ocr_available)
                if len(ocr_text) >= MIN_TEXT_LEN:
                    text = ocr_text
                    pages_ocrd += 1
                else:
                    pages_empty += 1

            page_chunks = []
            if len(text) >= MIN_TEXT_LEN:
                page_chunks.append(text)

            if page_num < len(plumber_doc.pages):
                tables_md = extract_tables_from_page(plumber_doc.pages[page_num])
                tables_found += len(tables_md)
                page_chunks.extend(tables_md)

            if page_chunks:
                parts.append(f"[Page {page_num + 1}]\n" + "\n\n".join(page_chunks))
    finally:
        doc.close()
        plumber_doc.close()

    return {
        "text": "\n\n".join(parts).strip(),
        "pages_total": pages_total,
        "pages_ocrd": pages_ocrd,
        "pages_empty": pages_empty,
        "tables_found": tables_found,
        "ocr_available": ocr_available,
        "truncated": truncated,
    }