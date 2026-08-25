"""
purge_stale_cache.py — unstick individual cached answers in query_cache.

Why this exists: query_cache only ever stored question_hash (a SHA-256 of
the normalized question), never the question text itself, so a stuck bad
answer couldn't be found with a keyword search — you had to already know
the exact question and recompute its hash. rag.py's ensure_cache_table()
now backfills a `question_text` column for every row saved from here on,
but existing rows (like the Dec-EFY-2018 inflation one) still only have a
hash. This script covers both cases:

  1. Recompute-and-delete: give it the exact question text(s) as they were
     typed, it hashes them the same way rag.py does and deletes those rows
     — works even on old rows with no question_text.
  2. Keyword purge: --like deletes every row whose question_text contains
     a keyword — only matches rows saved after this update.

USAGE
  # Unstick one exact question (must match what the user actually typed,
  # case/whitespace-insensitive — normalization matches rag.py's):
  python purge_stale_cache.py "What was the inflation rate in Ethiopia in 2018?"

  # Multiple at once:
  python purge_stale_cache.py "What was the inflation rate in Ethiopia in 2018?" \\
                               "What is the land utilization in Amhara region?"

  # Keyword purge (only rows saved after the question_text column exists):
  python purge_stale_cache.py --like inflation
  python purge_stale_cache.py --like "land utilization"

  # See what a purge would delete without deleting it:
  python purge_stale_cache.py --like inflation --dry-run

  # Nuclear option — wipes every cached answer. Use this when a
  # hash-guess doesn't match (most likely cause: the answer was cached
  # under an LLM-rewritten phrasing of the question, not what the user
  # actually typed — rewrite_and_route()'s rewrite is non-deterministic
  # across app restarts, so the "same" question can end up cached under
  # more than one hash over time). Cache is fully regenerable, so this
  # only costs a slower first re-answer per question, nothing is lost:
  python purge_stale_cache.py --all
  python purge_stale_cache.py --all --dry-run   # just see the row count first

Requires the same .env / DB_HOST / DATABASE_URL setup rag.py already uses
— run it from the same directory (or make sure .env is on the path).
"""
import argparse
import hashlib
import os
import re
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def _normalize_question(question: str) -> str:
    # Must stay byte-for-byte identical to rag.py's _normalize_question —
    # any drift here means the recomputed hash won't match the stored one.
    q = question.lower().strip()
    q = re.sub(r"[^\w\s\u1200-\u137F]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def get_question_hash(question: str, language: str = "en") -> str:
    key = f"{language}:{_normalize_question(question)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _connect():
    database_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    if database_url:
        return psycopg2.connect(database_url)
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def purge_by_question(conn, questions: list, languages: list, dry_run: bool):
    with conn.cursor() as cur:
        deleted = 0
        for q in questions:
            for lang in languages:
                q_hash = get_question_hash(q, lang)
                if dry_run:
                    cur.execute("SELECT id, source_doc, question_text FROM query_cache WHERE question_hash = %s", (q_hash,))
                    row = cur.fetchone()
                    if row:
                        print(f"[dry-run] would delete id={row[0]} source={row[1]!r} lang={lang} for: {q!r}")
                        deleted += 1
                else:
                    cur.execute("DELETE FROM query_cache WHERE question_hash = %s RETURNING id", (q_hash,))
                    row = cur.fetchone()
                    if row:
                        print(f"Deleted id={row[0]} lang={lang} for: {q!r}")
                        deleted += 1
        if not dry_run:
            conn.commit()
        return deleted


def purge_by_keyword(conn, keyword: str, dry_run: bool):
    with conn.cursor() as cur:
        if dry_run:
            cur.execute("SELECT id, question_text FROM query_cache WHERE question_text ILIKE %s", (f"%{keyword}%",))
            rows = cur.fetchall()
            for r in rows:
                print(f"[dry-run] would delete id={r[0]} for: {r[1]!r}")
            return len(rows)
        cur.execute("DELETE FROM query_cache WHERE question_text ILIKE %s RETURNING id", (f"%{keyword}%",))
        rows = cur.fetchall()
        conn.commit()
        for r in rows:
            print(f"Deleted id={r[0]}")
        return len(rows)


def purge_all(conn, dry_run: bool):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM query_cache")
        count = cur.fetchone()[0]
        if dry_run:
            print(f"[dry-run] would delete all {count} row(s) from query_cache.")
            return count
        cur.execute("DELETE FROM query_cache")
        conn.commit()
        return count


def main():
    parser = argparse.ArgumentParser(description="Purge stuck query_cache entries.")
    parser.add_argument("questions", nargs="*", help="Exact question text(s) to purge (hash-matched).")
    parser.add_argument("--like", help="Keyword to match against question_text (ILIKE %%kw%%). Only affects rows saved after the question_text column was added.")
    parser.add_argument("--all", action="store_true", help="Wipe every row in query_cache. Use this when a hash-guess doesn't match — e.g. because the answer was cached under an LLM-rewritten phrasing you can't easily reproduce. Cache is fully regenerable; this only costs a slower first re-answer per question.")
    parser.add_argument("--languages", default="en,am", help="Comma-separated languages to also try when hash-matching (default: en,am).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting it.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt for --all.")
    args = parser.parse_args()

    if not args.questions and not args.like and not args.all:
        parser.error("Provide question text(s), --like <keyword>, or --all.")

    conn = _connect()
    try:
        total = 0
        if args.questions:
            langs = [l.strip() for l in args.languages.split(",") if l.strip()]
            total += purge_by_question(conn, args.questions, langs, args.dry_run)
        if args.like:
            total += purge_by_keyword(conn, args.like, args.dry_run)
        if args.all:
            if not args.dry_run and not args.yes:
                confirm = input("This deletes ALL cached answers. Type 'yes' to continue: ")
                if confirm.strip().lower() != "yes":
                    print("Aborted.")
                    return
            total += purge_all(conn, args.dry_run)
        verb = "Would delete" if args.dry_run else "Deleted"
        print(f"\n{verb} {total} cache row(s).")
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())