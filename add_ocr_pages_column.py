"""
Run once. Fixes the schema drift causing this warning on every index run:

    Insert with ocr_pages failed for '<file>.pdf'
    (column "ocr_pages" of relation "documents" does not exist)
    ; retrying without it.

index_pdfs.py already tries to write an ocr_pages value on insert (that
code path is done) — the documents table was just never migrated to have
the column. This adds it. Safe to re-run any number of times.
"""
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cur = conn.cursor()

cur.execute("""
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'documents' AND column_name = 'ocr_pages'
""")

if cur.fetchone():
    print("documents.ocr_pages already exists — nothing to do.")
else:
    cur.execute("""
        ALTER TABLE documents
        ADD COLUMN ocr_pages INTEGER DEFAULT 0;
    """)
    conn.commit()
    print("Added documents.ocr_pages (INTEGER, default 0).")
    print("Re-run index_pdfs.py once — every insert will now succeed on the first try, no more 'retrying without it' warnings.")

cur.close()
conn.close()
