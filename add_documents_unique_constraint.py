"""
Run once, before the new index_pdfs.py, so ON CONFLICT (filename) works.
Safe to re-run any number of times — checks first, skips cleanly if the
constraint is already there instead of crashing.
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
    SELECT 1 FROM information_schema.table_constraints
    WHERE table_name = 'documents'
      AND constraint_name = 'documents_filename_unique'
""")

if cur.fetchone():
    print("documents.filename is already UNIQUE — nothing to do.")
else:
    cur.execute("""
        ALTER TABLE documents
        ADD CONSTRAINT documents_filename_unique UNIQUE (filename);
    """)
    conn.commit()
    print("documents.filename is now UNIQUE.")

cur.close()
conn.close()