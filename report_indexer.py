"""
Single-report indexer for the /admin/add_report route in app.py.

Runs the EXACT same pipeline as the bulk index_pdfs.py script — text layer
-> OCR fallback -> table extraction (pdf_extract.py), same chunk size,
same category/year guessing (indexing_helpers.py) — but for ONE uploaded
PDF instead of a whole folder. This is what removes the "someone has to
copy the PDF onto the server and run index_pdfs.py by hand" step: a report
can now be added from the browser and is answerable immediately.

Deliberately does NOT create its own ChromaDB client or load its own copy
of the embedding model. The Flask app already has one running (via rag.py
-> retriever.py) to answer questions — loading a second copy of a
multilingual SentenceTransformer model would double memory use and startup
time for zero benefit. Callers MUST pass in the existing `collection` and
its lock from retriever.py (ChromaDB's local PersistentClient isn't safe
for concurrent access — see the comment above `_collection_lock` in
retriever.py — so writes from this module take the same lock reads do).
"""
import io
import logging

import fitz
import pdfplumber

from pdf_extract import (
    check_ocr_available,
    extract_text_from_page,
    extract_text_with_ocr,
    extract_tables_from_page,
    MIN_TEXT_LEN,
)
from indexing_helpers import chunk_text, guess_category, guess_year

log = logging.getLogger("report_indexer")


def index_single_pdf(
    file_bytes: bytes,
    filename: str,
    collection,
    collection_lock,
    category: str = None,
    year: str = None,
    existing_chunk_ids: set = None,
) -> dict:
    """
    Extracts, chunks, and embeds ONE in-memory PDF into `collection`.

    category/year: pass None to fall back to guess_category/guess_year
    (same rule the bulk indexer uses for any file with no metadata.csv row).

    existing_chunk_ids: optional set of chunk IDs already in the collection,
    to skip re-adding a page that (e.g.) a previous partial upload already
    wrote. Safe to omit — ChromaDB's collection.add() upserts on ID anyway,
    so a re-run just overwrites the same chunks rather than duplicating them.

    Returns:
    {
      "chunks_added": int, "tables_found": int, "pages_total": int,
      "pages_ocrd": int, "pages_empty": int, "category": str, "year": str,
      "ocr_available": bool,
    }
    """
    category = category or guess_category(filename)
    year = year or guess_year(filename)
    ocr_available = check_ocr_available()
    existing_chunk_ids = existing_chunk_ids or set()

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    plumber_doc = pdfplumber.open(io.BytesIO(file_bytes))
    pages_total = len(doc)

    all_chunks, all_ids, all_metadata = [], [], []
    pages_ocrd = pages_empty = tables_found = 0

    try:
        for page_num in range(pages_total):
            text = extract_text_from_page(doc[page_num])
            used_ocr = False

            if len(text) < MIN_TEXT_LEN:
                ocr_text = extract_text_with_ocr(doc[page_num], ocr_available)
                if len(ocr_text) >= MIN_TEXT_LEN:
                    text, used_ocr = ocr_text, True
                    pages_ocrd += 1
                else:
                    pages_empty += 1

            if len(text) >= MIN_TEXT_LEN:
                for i, chunk in enumerate(chunk_text(text)):
                    chunk_id = f"{filename}_p{page_num + 1}_c{i}"
                    if chunk_id in existing_chunk_ids:
                        continue
                    all_chunks.append(chunk)
                    all_ids.append(chunk_id)
                    all_metadata.append({
                        "source": filename,
                        "page": page_num + 1,
                        "category": category,
                        "year": year,
                        "type": "ocr_text" if used_ocr else "text",
                    })

            # Independent of the text-length check above — a page can be
            # near-empty of prose but still hold a real data table (see the
            # same note in index_pdfs.py).
            if page_num < len(plumber_doc.pages):
                tables_md = extract_tables_from_page(plumber_doc.pages[page_num])
                for t_i, table_md in enumerate(tables_md):
                    chunk_id = f"{filename}_p{page_num + 1}_table{t_i}"
                    if chunk_id in existing_chunk_ids:
                        continue
                    all_chunks.append(table_md)
                    all_ids.append(chunk_id)
                    all_metadata.append({
                        "source": filename,
                        "page": page_num + 1,
                        "category": category,
                        "year": year,
                        "type": "table",
                    })
                    tables_found += 1
    finally:
        doc.close()
        plumber_doc.close()

    if all_chunks:
        with collection_lock:
            collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadata)
        log.info(
            f"Indexed '{filename}': {len(all_chunks)} chunks "
            f"({tables_found} tables, {pages_ocrd} OCR'd pages), "
            f"category={category}, year={year}"
        )
    else:
        log.warning(f"'{filename}' produced 0 chunks — no extractable text or tables found.")

    return {
        "chunks_added": len(all_chunks),
        "tables_found": tables_found,
        "pages_total": pages_total,
        "pages_ocrd": pages_ocrd,
        "pages_empty": pages_empty,
        "category": category,
        "year": year,
        "ocr_available": ocr_available,
    }
