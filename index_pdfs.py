import fitz  # PyMuPDF
import pdfplumber
import chromadb
import os
import re
import csv
import logging
import psycopg2
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# Shared extraction logic (text layer, OCR fallback, table extraction) now
# lives in pdf_extract.py so the bulk indexer here and the /upload_document
# route in app.py can never drift apart on OCR behavior. See that file for
# the actual implementations.
from pdf_extract import (
    check_ocr_available,
    extract_text_from_page,
    extract_text_with_ocr,
    extract_tables_from_page,
    table_to_markdown,
    OCR_LANG,
    OCR_DPI,
)

# Chunking / category-guessing / metadata.csv helpers now live in
# indexing_helpers.py so this bulk script and the new /admin/add_report
# route (report_indexer.py) can never drift apart on these rules. See that
# file for the actual implementations.
from indexing_helpers import (
    chunk_text,
    guess_category,
    guess_year,
    load_metadata_csv as _load_metadata_csv,
)

load_dotenv()

# ═══════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("index_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════
PDF_FOLDER      = "data/pdf"
CHROMA_PATH     = "chroma_db"
COLLECTION_NAME = "ess_documents"
CHUNK_SIZE      = 700
OVERLAP         = 100
BATCH_SIZE      = 500
MIN_TEXT_LEN    = 50  # below this, a page is treated as empty/scanned and OCR is tried

# TABLE EXTRACTION — pdfplumber uses the PDF's own ruling lines, so this
# works well on bordered/gridded tables (ESS reports are mostly this kind).
# NOTE: pdfplumber only sees vector/text structure. On a scanned (raster)
# page it will never find a table — there is nothing to parse. Tables on
# scanned pages are NOT recovered by this script; they'd need a dedicated
# table-OCR tool (e.g. Camelot/Tabula won't help either since those also
# need a vector layer). OCR below recovers scanned TEXT, not scanned
# TABLE STRUCTURE — a scanned table will come back as OCR'd running text.
#
# OCR_LANG / OCR_DPI / check_ocr_available / extract_text_with_ocr etc. now
# live in pdf_extract.py (imported above) — see that file to tune them.
# Amharic OCR is noticeably weaker than English OCR (script/font support in
# Tesseract is less mature) — expect more garbled characters on Amharic
# pages than on English ones. This is a Tesseract limitation, not a bug here.

# ═══════════════════════════════════════
# CHROMADB SETUP
# ═══════════════════════════════════════
client = chromadb.PersistentClient(path=CHROMA_PATH)

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ef
)

# ═══════════════════════════════════════
# POSTGRES: FILE-LEVEL COMPLETION TRACKING
# Uses the existing `documents` table so already-indexed
# PDFs are skipped WITHOUT reopening/re-chunking them.
# ═══════════════════════════════════════

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def get_completed_files(conn) -> set:
    """Filenames already fully indexed, per the documents table."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT filename FROM documents;")
        rows = {r[0] for r in cur.fetchall()}
        cur.close()
        return rows
    except Exception as e:
        log.warning(f"Could not read documents table ({e}); falling back to chunk-level check only.")
        conn.rollback()
        return set()


def mark_file_complete(conn, filename: str, chunk_count: int, ocr_pages: int = 0):
    """Record a PDF as fully indexed. Commits per-file so a crash mid-run
    doesn't lose progress on files already done."""
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO documents (filename, chunk_count, indexed_at, ocr_pages)
            VALUES (%s, %s, NOW(), %s)
            ON CONFLICT (filename) DO UPDATE
                SET chunk_count = EXCLUDED.chunk_count,
                    indexed_at  = NOW(),
                    ocr_pages   = EXCLUDED.ocr_pages;
            """,
            (filename, chunk_count, ocr_pages)
        )
        conn.commit()
    except Exception as e:
        # Most likely cause: the `ocr_pages` column doesn't exist yet on an
        # older `documents` table. Fall back to the original insert so the
        # run still completes; ocr_pages just won't be recorded that time.
        log.warning(f"Insert with ocr_pages failed for '{filename}' ({e}); retrying without it.")
        conn.rollback()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO documents (filename, chunk_count, indexed_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (filename) DO UPDATE
                    SET chunk_count = EXCLUDED.chunk_count,
                        indexed_at  = NOW();
                """,
                (filename, chunk_count)
            )
            conn.commit()
        except Exception as e2:
            log.error(f"Failed to record '{filename}' in documents table: {e2}")
            conn.rollback()

# ══════════════════════════════════════
# METADATA CSV
# ══════════════════════════════════════
# data/pdf/metadata.csv columns: filename,category,year
# (load_metadata_csv itself now lives in indexing_helpers.py — this is a
# thin wrapper so the rest of this file doesn't need to change.)

def load_metadata_csv():
    metadata_map = _load_metadata_csv(PDF_FOLDER)
    if metadata_map:
        log.info(f"Loaded metadata.csv with {len(metadata_map)} entries")
    return metadata_map

# ═══════════════════════════════════════
# HELPERS
# (chunk_text / guess_category / guess_year now live in
# indexing_helpers.py — imported above)
# ═══════════════════════════════════════

def get_existing_chunk_ids() -> set:
    """Fallback dedup for any file not yet recorded in the documents table
    (e.g. a run that crashed partway through)."""
    try:
        result = collection.get()
        return set(result["ids"])
    except Exception:
        return set()


def add_in_batches(chunks, ids, metadatas):
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            documents=chunks[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end]
        )


# ═══════════════════════════════════════
# MAIN INDEXING FUNCTION
# ═══════════════════════════════════════

def index_pdfs():
    if not os.path.exists(PDF_FOLDER):
        log.error(f"Folder '{PDF_FOLDER}' not found.")
        return

    metadata_map = load_metadata_csv()
    ocr_available = check_ocr_available()

    pdf_files = sorted([
        f for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        log.error(f"No PDF files found in '{PDF_FOLDER}'")
        return

    conn = get_db_connection()
    completed_files = get_completed_files(conn)
    existing_chunk_ids = get_existing_chunk_ids()

    log.info(f"Found {len(pdf_files)} PDF files total")
    log.info(f"Already marked complete in DB: {len(completed_files)}")
    log.info(f"Existing chunks in ChromaDB   : {len(existing_chunk_ids)}")
    log.info(f"OCR fallback                  : {'enabled (' + OCR_LANG + ')' if ocr_available else 'disabled'}")
    log.info("=" * 50)

    total_new_chunks = 0
    total_ocr_pages = 0
    skipped_files = 0
    failed_files = []

    for pdf_file in pdf_files:

        # Fast skip — no file open, no re-chunking, original files untouched
        if pdf_file in completed_files:
            skipped_files += 1
            continue

        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        log.info(f"Processing: {pdf_file}")

        if pdf_file in metadata_map:
            category = metadata_map[pdf_file]["category"]
            year     = metadata_map[pdf_file]["year"]
        else:
            category = guess_category(pdf_file)
            year     = guess_year(pdf_file)
            log.info(f"   (no metadata.csv entry — guessed category={category}, year={year})")

        try:
            doc = fitz.open(pdf_path)
            plumber_doc = pdfplumber.open(pdf_path)

            all_chunks, all_ids, all_metadata = [], [], []
            skipped_pages = 0
            table_count = 0
            ocr_pages_this_file = 0

            for page_num in range(len(doc)):
                text = extract_text_from_page(doc[page_num])
                used_ocr = False

                # No usable text layer — try OCR before giving up on the page.
                if len(text) < MIN_TEXT_LEN:
                    ocr_text = extract_text_with_ocr(doc[page_num], ocr_available)
                    if len(ocr_text) >= MIN_TEXT_LEN:
                        text = ocr_text
                        used_ocr = True
                        ocr_pages_this_file += 1
                        log.info(f"   Page {page_num + 1}: recovered via OCR ({len(ocr_text)} chars)")
                    else:
                        skipped_pages += 1

                if len(text) >= MIN_TEXT_LEN:
                    for i, chunk in enumerate(chunk_text(text)):
                        chunk_id = f"{pdf_file}_p{page_num+1}_c{i}"
                        if chunk_id in existing_chunk_ids:
                            continue
                        all_chunks.append(chunk)
                        all_ids.append(chunk_id)
                        all_metadata.append({
                            "source":   pdf_file,
                            "page":     page_num + 1,
                            "category": category,
                            "year":     year,
                            "type":     "ocr_text" if used_ocr else "text"
                        })

                # Table extraction runs independently of the text-length
                # skip above — a page can be "near-empty" of prose text but
                # still hold a real data table (common in statistical annex
                # pages), so this must not be nested inside the block above.
                # NOTE: this still only finds tables on pages with a vector
                # text layer — OCR'd (scanned) pages do not produce tables
                # here, only running text (see comment near MIN_TABLE_ROWS).
                if page_num < len(plumber_doc.pages):
                    tables_md = extract_tables_from_page(plumber_doc.pages[page_num])
                    for t_i, table_md in enumerate(tables_md):
                        chunk_id = f"{pdf_file}_p{page_num+1}_table{t_i}"
                        if chunk_id in existing_chunk_ids:
                            continue
                        all_chunks.append(table_md)
                        all_ids.append(chunk_id)
                        all_metadata.append({
                            "source":   pdf_file,
                            "page":     page_num + 1,
                            "category": category,
                            "year":     year,
                            "type":     "table"
                        })
                        table_count += 1

            doc.close()
            plumber_doc.close()

            if all_chunks:
                add_in_batches(all_chunks, all_ids, all_metadata)
                total_new_chunks += len(all_chunks)
                total_ocr_pages += ocr_pages_this_file
                log.info(
                    f"   {len(all_chunks)} chunks ({table_count} tables, "
                    f"{ocr_pages_this_file} OCR'd pages) | category={category} | year={year}"
                )
                if skipped_pages > 0:
                    log.warning(f"    Skipped {skipped_pages} pages with no recoverable text (blank or OCR failed)")
            else:
                log.info("   No new chunks (already covered by existing chunk IDs)")

            # Record completion only after a successful add — keeps documents
            # table honest if a PDF fails midway
            mark_file_complete(
                conn, pdf_file,
                len(all_chunks) if all_chunks else 0,
                ocr_pages_this_file
            )

        except Exception as e:
            log.error(f"   Error: {pdf_file}: {str(e)}")
            failed_files.append(pdf_file)
            continue

    conn.close()

    log.info("=" * 50)
    log.info("✅ Indexing complete!")
    log.info(f"📈 New chunks this run     : {total_new_chunks}")
    log.info(f"🔎 Pages recovered via OCR : {total_ocr_pages}")
    log.info(f"⏭️  Files skipped (already done): {skipped_files}")
    log.info(f"📊 Total in ChromaDB       : {collection.count()}")
    if failed_files:
        log.warning(f"   Failed files ({len(failed_files)}): {failed_files}")
    log.info("=" * 50)


if __name__ == "__main__":
    index_pdfs()