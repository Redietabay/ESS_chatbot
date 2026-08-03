import os
import chromadb
import logging
from dotenv import load_dotenv

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

def search_documents(question: str, top_k: int = TOP_K) -> list:
    """
    Search ChromaDB for most relevant chunks.
    Returns list of results with text, source, page, score.
    """
    question = clean_query(question)

    if not question:
        log.warning("Empty question received")
        return []

    try:
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Safe structure check
        if (not results
                or "documents" not in results
                or not results["documents"]
                or not results["documents"][0]):
            log.warning("No results found in ChromaDB")
            return []

        output = []
        for i in range(len(results["documents"][0])):

            score = results["distances"][0][i]

            # Filter weak results
            if score > MIN_SCORE:
                continue

            output.append({
                "text":     results["documents"][0][i],
                "source":   results["metadatas"][0][i].get("source",   "Unknown"),
                "page":     results["metadatas"][0][i].get("page",     0),
                "category": results["metadatas"][0][i].get("category", "general"),
                "year":     results["metadatas"][0][i].get("year",     "unknown"),
                "score":    round(score, 4)
            })

        # Sort by score — lowest distance = best match
        output.sort(key=lambda x: x["score"])

        log.info(f"Found {len(output)} relevant chunks for: '{question[:50]}'")
        return output

    except Exception as e:
        log.error(f"Search error: {str(e)}")
        return []


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