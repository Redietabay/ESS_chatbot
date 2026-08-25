"""
Targeted re-index: clears ONLY the specific PDFs that lost pages because
Tesseract wasn't installed when they were indexed (confirmed via
index_log.txt — every one of these logged "Skipped N near-empty pages"
and the run-level summary showed "Pages recovered via OCR: 0" for all
of your indexing history so far).

Unlike reset_index.py (full wipe), this leaves the other ~40 already-clean
files alone, so re-running index_pdfs.py only re-embeds these ~180 pages
instead of all 10,000+ chunks.

Run once, AFTER installing Tesseract + the amh language pack:
    python reset_ocr_files.py
    python index_pdfs.py
"""
import os
import chromadb
import psycopg2
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH     = "chroma_db"
COLLECTION_NAME = "ess_documents"

# Every filename that logged "Skipped N near-empty pages" in index_log.txt
# while OCR was unavailable. Edit this list if your corpus has changed.
AFFECTED_FILES = [
    "Large-and-medium-manufacturing-industry-survey-report-2017.pdf",
    "REVISED_2013.LIVESTOCK-REPORT.FINAL-1.pdf",
    "analytical-report-on-housing-characteristics-2007.pdf",
    "area-production-and-farm-management-practices-private-peasant-holdings-belg-season-2020-21-2013-e.c.pdf",
    "edhs-2024-25-kir-01172026.pdf",
    "ethiopia-demographic-and-health-survey-edhs-2024_25-detail-report.pdf",
    "ethiopia-demographic-and-health-survey-edhs-2024_25-summary-report.pdf",
    "national-area-production.pdf",
    "national-demography.pdf",
    "national-farm-practices.pdf",
    "report-on-area-and-production-of-major-crops-202122-2014-e.c.pdf",
    "2011-ethiopia-dhs-final-report.pdf",
    "2013-MEHER-REPORT.FINAL_.pdf",
    "2013_Crop-Livestock-product-Utilization-final-report.pdf",
    "2016-ethiopian-dhs-final-report.pdf",
    "2021-22-Survey-Report.pdf",
    "6.The_6_2014_AGSS_2014.LIVESTOCK-REPORT_Final-3.pdf",
    "Ethiopia-Demographic-Health-Survey-Report-Key-Findings-2016.pdf",
    "Ethiopian-Manufacturing-Industries-Export-Earning-is-in-Deficit.pdf",
]


def main():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM documents WHERE filename = ANY(%s);",
        (AFFECTED_FILES,)
    )
    print(f"Cleared {cur.rowcount} row(s) from documents table.")
    conn.commit()
    cur.close()
    conn.close()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Delete every chunk whose metadata.source is one of the affected files
    # (covers text chunks, OCR chunks, and table chunks for that file).
    existing = collection.get(where={"source": {"$in": AFFECTED_FILES}})
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
        print(f"Deleted {len(ids)} chunk(s) from ChromaDB across {len(AFFECTED_FILES)} file(s).")
    else:
        print("No matching chunks found in ChromaDB (nothing to delete there).")

    print("\nDone. Now run: python index_pdfs.py")
    print("It will re-open just these files and OCR the pages that were skipped before.")


if __name__ == "__main__":
    main()
