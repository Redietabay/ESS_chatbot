import fitz  # PyMuPDF
import chromadb
import os
import re
import csv   
import logging
import psycopg2
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

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
MIN_TEXT_LEN    = 50  # skip pages shorter than this (likely blank/scan)

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


def mark_file_complete(conn, filename: str, chunk_count: int):
    """Record a PDF as fully indexed. Commits per-file so a crash mid-run
    doesn't lose progress on files already done."""
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
    except Exception as e:
        log.error(f"Failed to record '{filename}' in documents table: {e}")
        conn.rollback()

# ═══════════════════════════════════════
# METADATA CSV
# ═══════════════════════════════════════
# data/pdf/metadata.csv columns: filename,category,year

def load_metadata_csv():
    meta_path = os.path.join(PDF_FOLDER, "metadata.csv")
    metadata_map = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_map[row["filename"]] = {
                    "category": row.get("category", "general"),
                    "year": row.get("year", "unknown")
                }
        log.info(f"Loaded metadata.csv with {len(metadata_map)} entries")
    return metadata_map

# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    step = max(chunk_size - overlap, chunk_size // 2)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


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


def guess_category(filename):
    name = filename.lower()
    if "inflation" in name:
        return "inflation"
    elif any(x in name for x in ["population", "demographic", "edhs", "projected_population"]):
        return "population"
    elif any(x in name for x in ["agriculture", "crop", "livestock", "agss", "farm"]):
        return "agriculture"
    elif "trade" in name:
        return "trade"
    elif "manufactur" in name:
        return "manufacturing"
    elif any(x in name for x in ["labour", "labor", "migration"]):
        return "labour"
    elif "housing" in name:
        return "housing"
    elif "survey" in name:
        return "survey"
    elif "land" in name:
        return "land"
    elif "commercial" in name:
        return "commercial"
    elif any(x in name for x in ["consumption", "welfare", "household"]):
        return "household"
    else:
        return "general"


def guess_year(filename):
    match = re.search(r"(20\d{2})", filename)
    return match.group(1) if match else "unknown"


def extract_text_from_page(page):
    return page.get_text().strip()

# ═══════════════════════════════════════
# MAIN INDEXING FUNCTION
# ═══════════════════════════════════════

def index_pdfs():
    if not os.path.exists(PDF_FOLDER):
        log.error(f"Folder '{PDF_FOLDER}' not found.")
        return

    metadata_map = load_metadata_csv()

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
    log.info("=" * 50)

    total_new_chunks = 0
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

            all_chunks, all_ids, all_metadata = [], [], []
            skipped_pages = 0

            for page_num in range(len(doc)):
                text = extract_text_from_page(doc[page_num])

                if len(text) < MIN_TEXT_LEN:
                    skipped_pages += 1
                    continue

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
                        "year":     year
                    })

            doc.close()

            if all_chunks:
                add_in_batches(all_chunks, all_ids, all_metadata)
                total_new_chunks += len(all_chunks)
                log.info(f"   {len(all_chunks)} chunks | category={category} | year={year}")
                if skipped_pages > 0:
                    log.warning(f"    Skipped {skipped_pages} near-empty pages (check if scanned/image PDF)")
            else:
                log.info("   No new chunks (already covered by existing chunk IDs)")

            # Record completion only after a successful add — keeps documents
            # table honest if a PDF fails midway
            mark_file_complete(conn, pdf_file, len(all_chunks) if all_chunks else 0)

        except Exception as e:
            log.error(f"   Error: {pdf_file}: {str(e)}")
            failed_files.append(pdf_file)
            continue

    conn.close()

    log.info("=" * 50)
    log.info("✅ Indexing complete!")
    log.info(f"📈 New chunks this run :{total_new_chunks}")
    log.info(f"⏭️  Files skipped (already done): {skipped_files}")
    log.info(f"📊 Total in ChromaDB       : {collection.count()}")
    if failed_files:
        log.warning(f"   Failed files ({len(failed_files)}): {failed_files}")
    log.info("=" * 50)
    
    
if __name__ == "__main__":
    index_pdfs()