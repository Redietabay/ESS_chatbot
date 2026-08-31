"""
Shared indexing helpers used by BOTH:
  - index_pdfs.py       (bulk indexer — someone runs this by hand today)
  - report_indexer.py   (single-report indexer — used by app.py's new
                          /admin/add_report route, so a report can be added
                          from the browser instead)

Split out for the exact same reason pdf_extract.py was split out (see its
docstring): two indexing entry points must never drift apart on chunking
size, category/year guessing, or the `documents` table bookkeeping. A fix
made in one place should never need to be copy-pasted into the other.

Importing this module has NO side effects (no ChromaDB client, no model
load, no DB connection) — safe to import from anywhere, including inside
the Flask app process.
"""
import os
import re
import csv
import logging

log = logging.getLogger("indexing_helpers")

CHUNK_SIZE = 700
OVERLAP = 100


# ═══════════════════════════════════════
# CHUNKING
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


# ═══════════════════════════════════════
# CATEGORY / YEAR GUESSING (fallback when metadata.csv has no entry, and
# the /admin/add_report form was submitted with the fields left blank)
# ═══════════════════════════════════════
def guess_category(filename):
    name = filename.lower()
    # NOTE: this must stay in sync with rag.py's _PRICE_KEYWORDS — that's
    # the list rag.py uses to decide a *question* is about inflation/prices.
    # If a file's category doesn't land in "inflation" here, rag.py's
    # EFY-aware year filtering never sees it at all (it only looks at
    # chunks whose metadata category == "inflation"), so a correctly
    # uploaded report with a different naming style (e.g. "CPI_DEC_2022.pdf"
    # instead of "...inflation-report...efy-2018-final.pdf") would silently
    # fall into "general" and the chatbot would claim it isn't indexed.
    if any(x in name for x in [
        "inflation", "cpi", "consumer_price", "consumer-price",
        "price_index", "price-index", "cost_of_living", "cost-of-living",
    ]):
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


# ═══════════════════════════════════════
# metadata.csv (data/pdf/metadata.csv — columns: filename,category,year)
# ═══════════════════════════════════════
def load_metadata_csv(pdf_folder):
    meta_path = os.path.join(pdf_folder, "metadata.csv")
    metadata_map = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_map[row["filename"]] = {
                    "category": row.get("category", "general"),
                    "year": row.get("year", "unknown"),
                }
    return metadata_map


def append_metadata_csv(pdf_folder, filename, category, year):
    """Adds (or overwrites) one row in metadata.csv for `filename`. Called
    after a successful /admin/add_report upload so that a *future* full
    `python index_pdfs.py` run — e.g. after restoring from backup, or
    rebuilding the ChromaDB collection from scratch — picks up the exact
    same category/year the admin chose, instead of silently falling back
    to guess_category/guess_year for that file.

    Always writes exactly the filename/category/year columns, regardless
    of what the existing file's header says or how many columns any of its
    rows actually have. This matters because a metadata.csv edited by hand
    (stray trailing commas, an extra column someone added once) makes
    csv.DictReader stuff the overflow values into a special `None` key —
    passing that straight to DictWriter.writerows() raises ValueError.
    extrasaction="ignore" drops any such key instead of crashing.
    """
    meta_path = os.path.join(pdf_folder, "metadata.csv")
    fieldnames = ["filename", "category", "year"]
    rows = []
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("filename") == filename:
                    continue
                rows.append({k: row.get(k, "") for k in fieldnames})
    rows.append({"filename": filename, "category": category, "year": year})
    os.makedirs(pdf_folder, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"metadata.csv updated for '{filename}' (category={category}, year={year})")