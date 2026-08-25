"""
Full cache wipe — run after any rag.py change that alters answer generation
(prompt wording, routing logic) where you can't cleanly pattern-match which
cached rows are stale, unlike purge_bad_cache.py / purge_no_relevant_cache.py.

Destructive: deletes every row in query_cache. Asks for confirmation first.

Run:  python clear_all_cache.py          (asks to confirm)
      python clear_all_cache.py -y       (skips confirmation, e.g. in a script)
"""
import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv()

if "-y" not in sys.argv:
    confirm = input("This deletes ALL cached answers. Continue? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled — no rows deleted.")
        sys.exit(0)

conn = None
try:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cur = conn.cursor()
    cur.execute("DELETE FROM query_cache;")
    conn.commit()
    deleted = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else "an unknown number of"
    print(f"Cleared {deleted} cached row(s). Every question will regenerate fresh from now on.")
    cur.close()
except Exception as e:
    print(f"Failed to clear cache: {e}")
    if conn:
        conn.rollback()
finally:
    if conn:
        conn.close()