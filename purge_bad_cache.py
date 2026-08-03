"""
One-time cleanup: the empty-table answer you saw in the screenshot is already
saved in query_cache from before is_bad_answer() knew to reject it. Fixing
is_bad_answer() only stops it happening again going forward — this script
removes the specific stale row(s) so the next ask regenerates a real answer.

Run once:  python purge_bad_cache.py
"""
import os
import re
import psycopg2
from dotenv import load_dotenv

load_dotenv()

TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


def has_empty_markdown_table(answer: str) -> bool:
    lines = answer.splitlines()
    for i, line in enumerate(lines):
        if TABLE_SEP_RE.match(line.strip()):
            for later in lines[i + 1:]:
                stripped = later.strip()
                if not stripped:
                    continue
                if stripped.startswith("|"):
                    inner = stripped.strip("|")
                    if re.search(r"[^\s\-:|]", inner):
                        return False
                return False
            return True
    return False


conn = psycopg2.connect(
    host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cur = conn.cursor()
cur.execute("SELECT id, question_hash, answer FROM query_cache")
rows = cur.fetchall()

bad_ids = [r[0] for r in rows if has_empty_markdown_table(r[2])]

if bad_ids:
    cur.execute("DELETE FROM query_cache WHERE id = ANY(%s)", (bad_ids,))
    conn.commit()
    print(f"Removed {len(bad_ids)} stale empty-table cache row(s): ids {bad_ids}")
else:
    print("No empty-table cache rows found.")

cur.close()
conn.close()
