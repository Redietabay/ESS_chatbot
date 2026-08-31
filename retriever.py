import os
import time
import threading
import chromadb
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dotenv import load_dotenv

try:
    import ftfy  # pip install ftfy — "fixes text for you": repairs mojibake
except ImportError:
    ftfy = None
    print("[retriever.py] WARNING: 'ftfy' not installed — corrupted-PDF-text "
          "cleanup is disabled. Run: pip install ftfy")

# Must run before the embedding model initializes below — it reads
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE from the environment. Safe to call
# again even if rag.py already loaded .env (load_dotenv() is a no-op if
# called twice); this just makes retriever.py correct on its own too, e.g.
# when run directly via `python retriever.py`.
load_dotenv()

from chromadb.utils import embedding_functions

# ═══════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════
from logging.handlers import RotatingFileHandler

log = logging.getLogger("retriever")
log.setLevel(logging.INFO)
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _fh = RotatingFileHandler("retriever_log.txt", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    _fh.setFormatter(_fmt)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    log.addHandler(_fh)
    log.addHandler(_sh)
    log.propagate = False  # don't leak into the root logger — keeps this file to retriever activity only

# ═══════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════
CHROMA_PATH     = "chroma_db"
COLLECTION_NAME = "ess_documents"
TOP_K           = 5
# Max distance allowed — lower = stricter/more relevant. Raised slightly
# from 1.5 -> 1.65 by default: at 1.5 some genuinely relevant chunks (e.g.
# paraphrased questions that don't lexically match the report wording) were
# being filtered out before ever reaching the LLM, which is what produced
# "the text does not provide the exact figure" even when the number existed
# in a chunk just outside the old cutoff. Tune via env if you see the
# opposite problem (irrelevant chunks getting through).
MIN_SCORE       = float(os.getenv("RETRIEVER_MIN_SCORE", "1.65"))
# Hard cap on a single query's wall-clock time. Serving under waitress with
# threads=8 (see app.py), multiple requests can hit `collection` at the same
# time — ChromaDB's local PersistentClient isn't guaranteed safe for that,
# and a stuck query previously blocked a worker thread with NO log output at
# all (no "Found chunks", no "Search error" — just silence). This timeout
# guarantees the thread always comes back and always logs something.
SEARCH_TIMEOUT_SECONDS = float(os.getenv("RETRIEVER_SEARCH_TIMEOUT", "20"))

# ═══════════════════════════════════════
# CHROMADB CONNECTION
# Same embedding function as index_pdfs.py
# ═══════════════════════════════════════
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ef
)

# Serializes access to `collection` across waitress's worker threads. Reads
# are usually fine concurrently, but this removes an entire class of
# lock-contention hangs for the cost of queries queueing briefly instead of
# running in parallel — worth it given search is already sub-second.
_collection_lock = threading.Lock()
_query_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="chroma-query")

# ═══════════════════════════════════════
# QUERY CLEANING
# ═══════════════════════════════════════

def clean_query(question: str) -> str:
    """Clean and normalize the user question."""
    if not question:
        return ""
    # Remove extra spaces, strip punctuation edges
    question = question.strip()
    question = " ".join(question.split())
    return question


# ═══════════════════════════════════════
# MAIN SEARCH FUNCTION
# ═══════════════════════════════════════

def _run_query(question: str, top_k: int, where: dict = None):
    """The actual ChromaDB call, run inside the timeout wrapper below.
    Serialized via _collection_lock — see comment at the lock's definition."""
    with _collection_lock:
        kwargs = {
            "query_texts": [question],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return collection.query(**kwargs)


def _to_output_rows(results) -> list:
    if (not results
            or "documents" not in results
            or not results["documents"]
            or not results["documents"][0]):
        return []
    rows = []
    for i in range(len(results["documents"][0])):
        score = results["distances"][0][i]
        if score > MIN_SCORE:
            continue
        text = results["documents"][0][i]
        # Some indexed PDFs have a broken/missing ToUnicode CMap on an
        # embedded font — PyMuPDF's text extraction then returns corrupted
        # codepoints (tofu boxes, e.g. "EFY 2010" then garbage instead of a
        # dash) baked right into the chunk. Clean it here, once, so it
        # never reaches the LLM's context (and therefore never gets quoted
        # into an answer) rather than trying to catch it after the fact.
        if ftfy is not None:
            text = ftfy.fix_text(text)
        rows.append({
            "text":     text,
            "source":   results["metadatas"][0][i].get("source",   "Unknown"),
            "page":     results["metadatas"][0][i].get("page",     0),
            "category": results["metadatas"][0][i].get("category", "general"),
            "year":     results["metadatas"][0][i].get("year",     "unknown"),
            "score":    round(score, 4)
        })
    return rows


def search_documents(question: str, top_k: int = TOP_K, prefer_source: str = None, source_in: list = None):
    """
    Search ChromaDB for most relevant chunks.

    Returns one of three things — callers MUST distinguish them, they mean
    different things:
      - a non-empty list of result dicts: chunks found, use them.
      - an empty list []: search completed normally and genuinely found
        nothing relevant (below MIN_SCORE, or no chunks at all).
      - None: search did NOT complete (timeout or exception) — this is
        "unknown", not "confirmed empty". Treating None the same as []
        previously caused the app to confidently tell a user "I don't have
        any data" when the true cause was an infra timeout, not an absence
        of data (Aug 25 rag_log.txt: October EFY2018 food/non-food
        follow-up — retriever_log.txt shows a 10s timeout, not a real
        empty result; the answer was sitting in the same file six minutes
        earlier). Callers should surface a "please try again" style message
        for None, not a "no data available" claim.

    prefer_source: an exact filename (matches the "source" metadata field
    set at index time, e.g. "national-land-use.pdf"). When given, this file
    is searched FIRST with a metadata filter, and its chunks are placed
    ahead of the general (unfiltered) results — instead of leaving it to
    embedding similarity alone to rank the right document above a
    similarly-worded but wrong one.

    Added after confirming (rag_log.txt, screenshots) that "land
    utilization in other regions" pulled national-area-production.pdf
    (crop production %, wrong) ahead of national-land-use.pdf (land area
    hectares, right) — guess_category() in index_pdfs.py buckets both
    under a category too generic to filter on ("general" for the
    production file, since its filename matches none of that function's
    keyword lists), so a plain category filter can't fix this; pinning by
    exact filename can.

    source_in: optional list of exact filenames to restrict the search to
    (ChromaDB "$in" filter on the "source" metadata field), ignored if
    prefer_source is also given. Added for the ESS inflation corpus, which
    is ~19 separate MONTHLY reports, not one report per year — a bare
    "inflation rate in EFY 2018" question has no single right file to
    prefer, but letting embedding similarity search all ~60+ indexed PDFs
    means a same-topic chunk from the wrong month (e.g. a historical
    comparison table on page 10 of the October report) can beat the actual
    right-month figure sitting in a different file entirely. Passing every
    EFY-2018 inflation filename here keeps the search inside just that set.
    """
    question = clean_query(question)

    if not question:
        log.warning("Empty question received")
        return []

    log.info(
        f"Searching for: '{question[:50]}'"
        + (f" (preferring {prefer_source})" if prefer_source else "")
        + (f" (restricted to {len(source_in)} file(s))" if source_in and not prefer_source else "")
    )
    start = time.time()

    try:
        if prefer_source:
            preferred_future = _query_executor.submit(_run_query, question, top_k, {"source": prefer_source})
            general_future = _query_executor.submit(_run_query, question, top_k)
            try:
                preferred_results = preferred_future.result(timeout=SEARCH_TIMEOUT_SECONDS)
                general_results = general_future.result(timeout=SEARCH_TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                log.error(f"Search TIMED OUT after {SEARCH_TIMEOUT_SECONDS}s for: '{question[:50]}'")
                # None, not [] — a timeout means "we don't know", not "we
                # checked and there's nothing". Confusing the two is what
                # made the app confidently tell a user "I don't have any
                # data" when the real cause was an infra hiccup (Aug 25
                # rag_log.txt: the October EFY2018 food/non-food follow-up).
                return None

            preferred_rows = _to_output_rows(preferred_results)
            general_rows = _to_output_rows(general_results)

            seen = {(r["source"], r["page"], r["text"][:80]) for r in preferred_rows}
            merged = preferred_rows + [
                r for r in general_rows
                if (r["source"], r["page"], r["text"][:80]) not in seen
            ]
            # Preferred-source chunks keep their place at the front
            # regardless of score — that's the whole point of "prefer" —
            # then the rest sort by score as usual.
            merged = preferred_rows + sorted(merged[len(preferred_rows):], key=lambda x: x["score"])
            output = merged[:top_k]

            if not output:
                log.warning("No results found in ChromaDB")
                return []
            log.info(f"Found {len(output)} relevant chunks ({len(preferred_rows)} from {prefer_source}) for: '{question[:50]}' ({time.time() - start:.2f}s)")
            return output

        where = {"source": {"$in": source_in}} if source_in else None
        future = _query_executor.submit(_run_query, question, top_k, where)
        try:
            results = future.result(timeout=SEARCH_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            log.error(
                f"Search TIMED OUT after {SEARCH_TIMEOUT_SECONDS}s for: "
                f"'{question[:50]}' — returning None (unknown, not confirmed-empty) "
                f"instead of hanging the request."
            )
            return None

        output = _to_output_rows(results)
        if not output:
            log.warning("No results found in ChromaDB")
            return []

        # Sort by score — lowest distance = best match
        output.sort(key=lambda x: x["score"])

        log.info(f"Found {len(output)} relevant chunks for: '{question[:50]}' ({time.time() - start:.2f}s)")
        return output

    except Exception as e:
        log.error(f"Search error after {time.time() - start:.2f}s: {str(e)}")
        # Also None, not [] — an exception mid-search is exactly the same
        # "unknown" case as a timeout, not a confirmed-empty result.
        return None


# ═══════════════════════════════════════
# METADATA INDEX (by category)
# ═══════════════════════════════════════
# BUG THIS FIXES: rag.py used to figure out "which years of inflation
# reports are indexed" and "which files match year X" by re-scanning raw
# filenames on disk for the literal substrings "inflation" and "efy" (e.g.
# only "...inflation-report-oct-efy-2018-final.pdf"-style names matched).
# Any file that was indexed correctly (category="inflation", year=<...> in
# its ChromaDB metadata) but named differently — e.g. "CPI_DEC_2022.pdf",
# uploaded through the admin form or with a metadata.csv row — was
# invisible to that logic, so the chatbot confidently claimed the data
# wasn't indexed even though it was answerable. Reading straight from the
# actual indexed metadata means ANY file, named however, is picked up as
# long as it was categorized correctly at index time.
def get_metadata_index(category: str) -> dict:
    """Returns {year_string: [source_filenames, ...]} for every chunk
    indexed under the given metadata `category`, read directly from
    ChromaDB. This reflects what was actually indexed, regardless of the
    source file's naming convention — unlike scanning PDF_FOLDER filenames
    for keyword substrings.
    """
    try:
        with _collection_lock:
            result = collection.get(where={"category": category}, include=["metadatas"])
    except Exception as e:
        log.warning(f"get_metadata_index('{category}') failed: {e}")
        return {}

    index = {}
    for meta in (result.get("metadatas") or []):
        year = str(meta.get("year", "unknown"))
        source = meta.get("source", "Unknown")
        index.setdefault(year, set()).add(source)

    return {year: sorted(sources) for year, sources in index.items()}


def get_corpus_summary() -> dict:
    """{category: {"chunks": int, "sources": int, "years": [sorted years]}}
    for the whole indexed corpus — read live from ChromaDB metadata, same
    approach as get_metadata_index(). Powers the admin observability
    dashboard's "what's actually indexed" panel."""
    try:
        with _collection_lock:
            result = collection.get(include=["metadatas"])
    except Exception as e:
        log.warning(f"get_corpus_summary() failed: {e}")
        return {}

    summary = {}
    for meta in (result.get("metadatas") or []):
        cat = meta.get("category", "general")
        entry = summary.setdefault(cat, {"chunks": 0, "sources": set(), "years": set()})
        entry["chunks"] += 1
        entry["sources"].add(meta.get("source", "Unknown"))
        entry["years"].add(str(meta.get("year", "unknown")))

    return {
        cat: {
            "chunks": v["chunks"],
            "sources": len(v["sources"]),
            "years": sorted(v["years"]),
        }
        for cat, v in summary.items()
    }


# ═══════════════════════════════════════
# FORMAT CONTEXT FOR GROQ
# ═══════════════════════════════════════

def format_context(results: list) -> tuple:
    """
    Format search results into context string for Groq.
    Returns (context_text, best_source, best_page)
    """
    if not results:
        return "", "Unknown", 0

    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[Source {i+1}: {r['source']} - "
            f"Page {r['page']} | "
            f"Category: {r['category']} | "
            f"Year: {r['year']}]\n"
            f"{r['text']}"
        )

    context     = "\n\n".join(context_parts)
    best_source = results[0]["source"]
    best_page   = results[0]["page"]

    return context, best_source, best_page


# ═══════════════════════════════════════
# TEST
# ═══════════════════════════════════════

if __name__ == "__main__":

    test_questions = [
        "What was the inflation rate in Ethiopia in 2018?",
        "What is the population of Ethiopia?",
        "Tell me about livestock production in Ethiopia",
        "What is the land utilization in Amhara region?",
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)

        results = search_documents(question)

        if not results:
            print("No relevant results found.")
            continue

        context, source, page = format_context(results)

        print(f"Top match : {source} - Page {page}")
        print(f"Score     : {results[0]['score']}")
        print(f"Category  : {results[0]['category']}")
        print(f"Year      : {results[0]['year']}")
        print(f"Preview   : {results[0]['text'][:200]}...")
        print("-" * 50)