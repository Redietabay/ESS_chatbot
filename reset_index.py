"""
One-time reset: clears the `documents` completion-tracking table AND the
ChromaDB collection, so the next `python index_pdfs.py` run re-processes
every PDF from scratch and applies metadata.csv (category/year) instead of
skipping files that were already indexed with guessed metadata.

Run once:  python reset_index.py
Then:      python index_pdfs.py
"""
import os
import shutil
import psycopg2
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"

# ── 1. Clear Postgres completion tracking ──
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cur = conn.cursor()
cur.execute("DELETE FROM documents;")
conn.commit()
print(f"Cleared {cur.rowcount} row(s) from documents table.")
cur.close()
conn.close()

# ── 2. Clear ChromaDB (delete the persisted collection folder) ──
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
    print(f"Deleted '{CHROMA_PATH}' folder — will be recreated fresh on next index run.")
else:
    print(f"'{CHROMA_PATH}' not found, nothing to delete.")

print("\nDone. Now run: python index_pdfs.py")
