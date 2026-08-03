"""
One-time cleanup: is_bad_answer() previously did NOT treat "Nothing relevant
was found." as bad, so it got cached like any real answer — which is why
some questions keep returning that message forever even after the routing
fix, while an identical question asked fresh works fine.

This removes those stale cache rows so the next ask regenerates a real
answer using the current (fixed) routing logic.

Run once:  python purge_no_relevant_cache.py
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cur = conn.cursor()

cur.execute("""
    DELETE FROM query_cache
    WHERE answer ILIKE '%nothing relevant%'
       OR answer ILIKE '%no relevant%'
    RETURNING id, question_hash;
""")
removed = cur.fetchall()
conn.commit()

if removed:
    print(f"Removed {len(removed)} stale 'nothing relevant' cache row(s): ids {[r[0] for r in removed]}")
else:
    print("No stale 'nothing relevant' cache rows found.")

cur.close()
conn.close()
