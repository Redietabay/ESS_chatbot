import os
import sys
import re
import ast
import json
import time
import hashlib
import logging
import threading
from logging.handlers import RotatingFileHandler
import inspect
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import psycopg2
import psycopg2.pool
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
try:
    import ftfy  # pip install ftfy — "fixes text for you": repairs mojibake
except ImportError:
    ftfy = None
    print("[rag.py] WARNING: 'ftfy' not installed — mojibake/corrupted-PDF-text "
          "cleanup is disabled. Run: pip install ftfy")

# CRITICAL: must run before importing retriever — retriever.py initializes
# the SentenceTransformer model at import time, which reads HF_HUB_OFFLINE /
# TRANSFORMERS_OFFLINE from the environment. If .env hasn't been loaded yet,
# those flags are unset and the model init hits the network on every
# startup instead of using the local cache (this was the cause of the slow,
# repeated HuggingFace HEAD requests on startup).
load_dotenv()

from retriever import search_documents, format_context, get_metadata_index

# ═══════════════════════════════════════
# UTF-8 FIX — Defensive Check
# ═══════════════════════════════════════
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
log = logging.getLogger("rag")
log.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _fh = RotatingFileHandler("rag_log.txt", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    _fh.setFormatter(_fmt)
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    log.addHandler(_fh)
    log.addHandler(_sh)
    log.propagate = False  # don't also hand records up to the root logger

# ═══════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════
MODEL                    = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FAST_MODEL               = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
# Groq retires models with no code-side warning — this already caused a real
# outage (Aug 20 rag_log.txt: every request 400'd on "llama-3.1-70b-versatile
# has been decommissioned" until .env was hand-fixed). Mirrors the GEMINI_MODEL
# guard below: known-dead names get overridden to the current code default
# instead of failing every request until someone notices in the logs. Extend
# this set as Groq deprecates more models — check
# https://console.groq.com/docs/deprecations when a model_decommissioned
# error shows up in rag_log.txt.
_GROQ_KNOWN_DECOMMISSIONED = {
    "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192",
    "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it",
}
if MODEL.strip() in _GROQ_KNOWN_DECOMMISSIONED:
    log.warning(f"GROQ_MODEL env var is set to the decommissioned '{MODEL.strip()}' — "
                f"overriding to 'openai/gpt-oss-120b'. Fix/remove GROQ_MODEL in your .env file.")
    MODEL = "openai/gpt-oss-120b"
if FAST_MODEL.strip() in _GROQ_KNOWN_DECOMMISSIONED:
    log.warning(f"GROQ_FAST_MODEL env var is set to the decommissioned '{FAST_MODEL.strip()}' — "
                f"overriding to 'openai/gpt-oss-20b'. Fix/remove GROQ_FAST_MODEL in your .env file.")
    FAST_MODEL = "openai/gpt-oss-20b"
# Sampling temperature used ONLY for the final user-facing answer text
# (get_answer/get_answer_stream). Everything else — query rewriting, routing,
# pandas code generation, translation — stays at temperature 0.0 by default
# (call_groq/call_groq_stream default to 0.0 when this isn't passed), since
# those need deterministic, exact output. A small positive value here just
# varies phrasing/wording call-to-call; it does NOT change which facts/
# numbers get retrieved (that's still RAG retrieval, untouched by this) or
# let the model invent figures — it only affects how the same retrieved
# facts get worded. Keep this low (0.1-0.3); 0.0 reads robotic/repetitive,
# anything much above ~0.4 risks looser wording of exact figures.
ANSWER_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.2"))
# Wall-clock cap per Groq API call so a stalled connection can't hang a worker
# indefinitely (e.g. mid-stream network drop). Applies to both the normal and
# streaming call paths below.
GROQ_REQUEST_TIMEOUT     = float(os.getenv("GROQ_REQUEST_TIMEOUT", "30.0"))
# Cap on how long a single 429 retry wait is allowed to be. Groq's error body
# tells us how long IT thinks we should wait (e.g. "try again in 10.0s") —
# honoring that exactly can make one request eat 10+ seconds. Capping it
# means we fail faster and let the automatic pandas-retry / user-facing
# fallback kick in instead of a single call silently blocking for a long time.
GROQ_RATE_LIMIT_MAX_WAIT = float(os.getenv("GROQ_RATE_LIMIT_MAX_WAIT", "1.0"))
CSV_FOLDER               = os.getenv("CSV_FOLDER", "data/csv")
PDF_FOLDER               = os.getenv("PDF_FOLDER", "data/pdf")
MAX_RESULT_ROWS          = int(os.getenv("MAX_RESULT_ROWS", "20"))
MAX_SCHEMA_COLUMNS_SHOWN = int(os.getenv("MAX_SCHEMA_COLUMNS_SHOWN", "25"))
CSV_TOP_N_FILES          = int(os.getenv("CSV_TOP_N_FILES", "2"))
PDF_RETRIEVAL_TOP_K      = int(os.getenv("PDF_RETRIEVAL_TOP_K", "8"))
CACHE_TTL_DAYS           = int(os.getenv("CACHE_TTL_DAYS", "30"))
DB_POOL_MIN              = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX              = int(os.getenv("DB_POOL_MAX", "5"))
DATA_DICTIONARY_FILE     = os.getenv("DATA_DICTIONARY_FILE", "data/data_dictionary.csv")
ROUTER_MODE              = os.getenv("ROUTER_MODE", "llm")  # "llm" or "keyword"
ENABLE_QUERY_REWRITE     = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"
MAX_HISTORY_TURNS        = int(os.getenv("MAX_HISTORY_TURNS", "5"))

SOURCE_FORMATS_DESC = os.getenv(
    "SOURCE_FORMATS_DESC",
    "structured statistical databases (CSV) and official reference reports (PDF)"
)

GROQ_UNAVAILABLE_MSG = "Service temporarily unavailable. Please try again."

# Moved to module level (was a local var inside rewrite_and_route()) so
# _run_pdf_search()'s year-filter below can reuse the exact same keyword
# list instead of drifting out of sync with a second copy.
_PRICE_KEYWORDS = ("inflation", "cpi", "consumer price", "price index", "cost of living")

# ═══════════════════════════════════════
# GROQ CLIENT
# ═══════════════════════════════════════
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)

# ═══════════════════════════════════════
# FALLBACK CIRCUIT BREAKER
# Logs show every Groq-exhausted request re-attempting Ollama (not running
# on this dev machine -> instant connection-refused) and, in the past,
# Gemini (was 404ing on a retired model name) on every single request,
# each adding real latency to an already-slow rate-limited answer. Once a
# fallback tier has failed once, it's overwhelmingly likely to fail again
# within the next few seconds/minutes (dead process, bad model name, no
# network) — so skip re-attempting it for a short cooldown instead of
# re-discovering the same failure on every request. Same pattern as the
# per-model Groq cooldown above, applied to the fallback tiers themselves.
# ═══════════════════════════════════════
_fallback_unreachable_until = {}
_fallback_lock = threading.Lock()
FALLBACK_COOLDOWN_SECONDS = float(os.getenv("FALLBACK_COOLDOWN_SECONDS", "60"))


def _fallback_available(name: str) -> bool:
    with _fallback_lock:
        until = _fallback_unreachable_until.get(name, 0)
    return time.time() >= until


def _mark_fallback_unreachable(name: str, seconds: float = FALLBACK_COOLDOWN_SECONDS):
    with _fallback_lock:
        _fallback_unreachable_until[name] = time.time() + seconds
    log.warning(f"'{name}' fallback failed — skipping it for the next {seconds:.0f}s instead of "
                f"re-attempting on every subsequent request.")


# ═══════════════════════════════════════
# OLLAMA FALLBACK (local, no API)
# Only used when Groq has fully failed for a request (rate-limited on both
# models, timed out, or erroring) — NOT the primary path. CPU-only laptop
# here, so keep OLLAMA_MODEL small (e.g. llama3.2:1b) or this will be slow
# enough to defeat the purpose. Set OLLAMA_ENABLED=false in .env to disable
# entirely and fall back to the old GROQ_UNAVAILABLE_MSG behavior.
# ═══════════════════════════════════════
import requests

# Shared, reused connection pool for every Gemini/Ollama HTTP call below.
# Previously each call used a bare requests.post(), which opens a brand-new
# TCP connection + TLS handshake every time — on a slow/jittery connection
# (confirmed: 90-527ms, high jitter, plain ping to 8.8.8.8) that handshake
# overhead is paid again on every single fallback call, on top of the
# actual request. A requests.Session() keeps the underlying connection
# alive (HTTP keep-alive) and reuses it for subsequent calls to the same
# host, cutting that repeated setup cost. Groq's own SDK client (see
# groq_client above) already does this internally — this brings Gemini and
# Ollama calls up to the same behavior.
_http_session = requests.Session()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_ENABLED  = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
OLLAMA_TIMEOUT  = float(os.getenv("OLLAMA_TIMEOUT", "60"))


def _call_ollama(prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
    """Non-streaming local fallback. Returns '' on any failure (model not
    pulled, Ollama not running, timeout) so callers can detect total failure
    and fall through to GROQ_UNAVAILABLE_MSG same as before."""
    if not OLLAMA_ENABLED or not _fallback_available("ollama"):
        return ""
    try:
        resp = _http_session.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if text:
            log.info(f"Ollama fallback answered ({len(text)} chars) using '{OLLAMA_MODEL}' (Groq exhausted).")
        return text
    except Exception as e:
        log.warning(f"Ollama fallback failed ({OLLAMA_BASE_URL}, model={OLLAMA_MODEL}): {e}")
        _mark_fallback_unreachable("ollama")
        return ""


def _call_ollama_stream(prompt: str, max_tokens: int = 500, temperature: float = 0.0):
    """Streaming local fallback. Yields text chunks as they arrive; yields
    nothing at all on failure, so the caller's `yielded_any` check correctly
    falls through to GROQ_UNAVAILABLE_MSG."""
    if not OLLAMA_ENABLED or not _fallback_available("ollama"):
        return
    try:
        with _http_session.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            },
            timeout=OLLAMA_TIMEOUT,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                piece = data.get("response", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break
    except Exception as e:
        log.warning(f"Ollama streaming fallback failed ({OLLAMA_BASE_URL}, model={OLLAMA_MODEL}): {e}")
        _mark_fallback_unreachable("ollama")
        return

# ═══════════════════════════════════════
# GEMINI FALLBACK (free tier, tried BEFORE Ollama)
# Used when Groq has fully failed for a request (rate-limited on both
# models, timed out, or erroring). Gemini's free tier (Flash-Lite by
# default) is meaningfully more accurate than a small local Ollama model,
# so it sits between Groq and Ollama in the fallback chain:
#   Groq -> Gemini -> Ollama
# Get a free key (no card required) at https://aistudio.google.com/apikey
# and set GEMINI_API_KEY in .env. Set GEMINI_ENABLED=false to skip this
# tier entirely and fall straight through to Ollama (old behavior).
# NOTE: prompts/responses sent to the Gemini free tier may be used by
# Google to improve their models — same caveat you already accepted for
# Groq's free tier, just flagging it since it's a different provider.
# ═══════════════════════════════════════
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
# Verified live against this project's actual key on 2026-08-16 (see
# diagnose_gemini.py): both "gemini-2.5-flash" and "gemini-2.5-flash-lite"
# 404 with "no longer available to new users" — a Google-side retirement,
# not a config bug. "gemini-flash-latest" is Google's alias that always
# points at the current recommended flash model (resolved to
# gemini-3.7-flash when tested) and survives future retirements without
# needing another manual model-name fix.
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
if GEMINI_MODEL.strip() in ("gemini-2.5-flash-lite", "gemini-2.5-flash"):
    # Safety net: these dated models 404 on this project's key ("no longer
    # available to new users"), which means a .env value is overriding the
    # code default above. Rather than silently 404 on every fallback call
    # during a demo, force the known-good alias and say so once at startup.
    log.warning(f"GEMINI_MODEL env var is set to the retired '{GEMINI_MODEL.strip()}' — overriding to 'gemini-flash-latest'. Fix/remove GEMINI_MODEL in your .env file.")
    GEMINI_MODEL = "gemini-flash-latest"
GEMINI_ENABLED  = os.getenv("GEMINI_ENABLED", "true").lower() == "true" and bool(GEMINI_API_KEY)
GEMINI_TIMEOUT  = float(os.getenv("GEMINI_TIMEOUT", "15"))
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# "latest" aliases can resolve to a reasoning model (confirmed: resolved to
# gemini-3.7-flash) that spends part of max_tokens on invisible "thinking"
# before writing the answer — a test call with max_tokens=10 came back with
# thoughtsTokenCount=7 and an EMPTY answer. Disabling the thinking budget
# keeps the full token budget going to the actual answer text, which is all
# this fallback path needs.
GEMINI_GENERATION_CONFIG_EXTRA = {"thinkingConfig": {"thinkingBudget": 0}}



def _call_gemini(prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
    """Non-streaming Gemini fallback. Returns '' on any failure (no key,
    rate-limited, network error) so callers fall through to Ollama exactly
    like a disabled/failed Ollama call already does."""
    if not GEMINI_ENABLED or not _fallback_available("gemini"):
        return ""
    try:
        resp = _http_session.post(
            f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent",
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature, **GEMINI_GENERATION_CONFIG_EXTRA},
            },
            timeout=GEMINI_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            log.warning(f"Gemini fallback returned no candidates (possibly safety-blocked): {data}")
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if text:
            log.info(f"Gemini fallback answered ({len(text)} chars) using '{GEMINI_MODEL}' (Groq exhausted).")
        return text
    except Exception as e:
        log.warning(f"Gemini fallback failed (model={GEMINI_MODEL}): {e}")
        _mark_fallback_unreachable("gemini")
        return ""


def _call_gemini_stream(prompt: str, max_tokens: int = 500, temperature: float = 0.0):
    """Streaming Gemini fallback via SSE. Yields text chunks as they arrive;
    yields nothing at all on failure, so the caller's `yielded_any` check
    correctly falls through to Ollama, same pattern as _call_ollama_stream."""
    if not GEMINI_ENABLED or not _fallback_available("gemini"):
        return
    try:
        with _http_session.post(
            f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:streamGenerateContent",
            params={"key": GEMINI_API_KEY, "alt": "sse"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature, **GEMINI_GENERATION_CONFIG_EXTRA},
            },
            timeout=GEMINI_TIMEOUT,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            # requests only trusts a charset it finds in the Content-Type
            # header. Gemini's SSE responses come back as
            # "text/event-stream" with no charset param, so requests falls
            # back to ISO-8859-1 (the HTTP default for text/*) even though
            # the body is actually UTF-8. iter_lines(decode_unicode=True)
            # decodes using resp.encoding, so every non-ASCII byte (Amharic
            # script, Ge'ez numerals, en/em dashes, etc.) comes out as
            # mojibake, e.g. "2010–2011" -> "2010â€"2011". Pin the encoding
            # explicitly before iterating so it decodes correctly regardless
            # of what (if anything) the header says.
            resp.encoding = "utf-8"
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload in ("", "[DONE]"):
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                candidates = data.get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                piece = "".join(p.get("text", "") for p in parts)
                if piece:
                    yield piece
    except Exception as e:
        log.warning(f"Gemini streaming fallback failed (model={GEMINI_MODEL}): {e}")
        _mark_fallback_unreachable("gemini")
        return

# ═══════════════════════════════════════
# OPENROUTER FALLBACK (free tier, tried AFTER Gemini)
# One key, many free ":free"-suffixed models on a different domain/infra
# than Groq and Gemini — a DNS or outage issue that takes down one provider
# (e.g. generativelanguage.googleapis.com specifically) won't necessarily
# take this one down at the same moment.
# Get a free key (no card required) at https://openrouter.ai/keys
# Set OPENROUTER_ENABLED=false to skip this tier.
# ═══════════════════════════════════════
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_ENABLED  = os.getenv("OPENROUTER_ENABLED", "true").lower() == "true" and bool(OPENROUTER_API_KEY)
OPENROUTER_TIMEOUT  = float(os.getenv("OPENROUTER_TIMEOUT", "15"))
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def _call_openrouter(prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
    """Non-streaming OpenRouter fallback. Returns '' on any failure (no key,
    disabled, rate-limited, network error)."""
    if not OPENROUTER_ENABLED or not _fallback_available("openrouter"):
        return ""
    try:
        resp = _http_session.post(
            OPENROUTER_BASE_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=OPENROUTER_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            log.warning(f"OpenRouter fallback returned no choices: {data}")
            return ""
        text = (choices[0].get("message", {}).get("content") or "").strip()
        if text:
            log.info(f"OpenRouter fallback answered ({len(text)} chars) using '{OPENROUTER_MODEL}' (Groq/Gemini exhausted).")
        return text
    except Exception as e:
        log.warning(f"OpenRouter fallback failed (model={OPENROUTER_MODEL}): {e}")
        _mark_fallback_unreachable("openrouter")
        return ""


def _call_openrouter_stream(prompt: str, max_tokens: int = 500, temperature: float = 0.0):
    """Streaming OpenRouter fallback (OpenAI-compatible SSE). Yields nothing
    at all on failure, matching the other streaming fallbacks."""
    if not OPENROUTER_ENABLED or not _fallback_available("openrouter"):
        return
    try:
        with _http_session.post(
            OPENROUTER_BASE_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
            timeout=OPENROUTER_TIMEOUT,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            # Same fix as the Gemini stream above: pin the decode encoding
            # instead of trusting requests' ISO-8859-1 fallback, which
            # mangles Amharic text and punctuation like en-dashes.
            resp.encoding = "utf-8"
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload in ("", "[DONE]"):
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta
    except Exception as e:
        log.warning(f"OpenRouter streaming fallback failed (model={OPENROUTER_MODEL}): {e}")
        _mark_fallback_unreachable("openrouter")
        return


# ═══════════════════════════════════════
# MISTRAL FALLBACK (free tier, tried AFTER Cerebras)
# Mistral's "Experiment" tier: no card required, ~1B tokens/month, rate-
# limited to a couple requests/minute — fine for a fallback tier that only
# fires when Groq/Gemini/OpenRouter/Cerebras are all already down, since
# those requests are rare by definition. Get a free key at
# https://console.mistral.ai (phone verification required, no card).
# Set MISTRAL_ENABLED=false to skip this tier.
# ═══════════════════════════════════════
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL    = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_ENABLED  = os.getenv("MISTRAL_ENABLED", "true").lower() == "true" and bool(MISTRAL_API_KEY)
MISTRAL_TIMEOUT  = float(os.getenv("MISTRAL_TIMEOUT", "15"))
MISTRAL_BASE_URL = "https://api.mistral.ai/v1/chat/completions"


def _call_mistral(prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
    """Non-streaming Mistral fallback. Returns '' on any failure."""
    if not MISTRAL_ENABLED or not _fallback_available("mistral"):
        return ""
    try:
        resp = _http_session.post(
            MISTRAL_BASE_URL,
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=MISTRAL_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            log.warning(f"Mistral fallback returned no choices: {data}")
            return ""
        text = (choices[0].get("message", {}).get("content") or "").strip()
        if text:
            log.info(f"Mistral fallback answered ({len(text)} chars) using '{MISTRAL_MODEL}' (prior tiers exhausted).")
        return text
    except Exception as e:
        log.warning(f"Mistral fallback failed (model={MISTRAL_MODEL}): {e}")
        _mark_fallback_unreachable("mistral")
        return ""


def _call_mistral_stream(prompt: str, max_tokens: int = 500, temperature: float = 0.0):
    """Streaming Mistral fallback (OpenAI-compatible SSE)."""
    if not MISTRAL_ENABLED or not _fallback_available("mistral"):
        return
    try:
        with _http_session.post(
            MISTRAL_BASE_URL,
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
            timeout=MISTRAL_TIMEOUT,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            resp.encoding = "utf-8"
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload in ("", "[DONE]"):
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta
    except Exception as e:
        log.warning(f"Mistral streaming fallback failed (model={MISTRAL_MODEL}): {e}")
        _mark_fallback_unreachable("mistral")
# Very high daily free token allowance and extremely fast inference — a
# solid 4th tier for when Groq, Gemini, AND OpenRouter are all down at once.
# Get a free key (no card required) at https://cloud.cerebras.ai
# Set CEREBRAS_ENABLED=false to skip this tier.
# ═══════════════════════════════════════
CEREBRAS_API_KEY  = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL    = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
CEREBRAS_ENABLED  = os.getenv("CEREBRAS_ENABLED", "true").lower() == "true" and bool(CEREBRAS_API_KEY)
CEREBRAS_TIMEOUT  = float(os.getenv("CEREBRAS_TIMEOUT", "15"))
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1/chat/completions"


def _call_cerebras(prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
    """Non-streaming Cerebras fallback. Returns '' on any failure."""
    if not CEREBRAS_ENABLED or not _fallback_available("cerebras"):
        return ""
    try:
        resp = _http_session.post(
            CEREBRAS_BASE_URL,
            headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": CEREBRAS_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=CEREBRAS_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            log.warning(f"Cerebras fallback returned no choices: {data}")
            return ""
        text = (choices[0].get("message", {}).get("content") or "").strip()
        if text:
            log.info(f"Cerebras fallback answered ({len(text)} chars) using '{CEREBRAS_MODEL}' (all prior tiers exhausted).")
        return text
    except Exception as e:
        log.warning(f"Cerebras fallback failed (model={CEREBRAS_MODEL}): {e}")
        _mark_fallback_unreachable("cerebras")
        return ""


def _call_cerebras_stream(prompt: str, max_tokens: int = 500, temperature: float = 0.0):
    """Streaming Cerebras fallback (OpenAI-compatible SSE)."""
    if not CEREBRAS_ENABLED or not _fallback_available("cerebras"):
        return
    try:
        with _http_session.post(
            CEREBRAS_BASE_URL,
            headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": CEREBRAS_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
            timeout=CEREBRAS_TIMEOUT,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            # Same fix as the Gemini/OpenRouter streams above.
            resp.encoding = "utf-8"
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload in ("", "[DONE]"):
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta
    except Exception as e:
        log.warning(f"Cerebras streaming fallback failed (model={CEREBRAS_MODEL}): {e}")
        _mark_fallback_unreachable("cerebras")
        return


# ═══════════════════════════════════════
# COMBINED FALLBACK CHAIN: Groq -> (tiers below, in FALLBACK_ORDER) -> Ollama
# Each tier is skipped instantly (no network call) while it's in its own
# 60s cooldown from a recent failure — see _fallback_available(). A tier
# that's disabled (no key / *_ENABLED=false) is also skipped instantly.
#
# Order used to be hardcoded as Gemini -> OpenRouter -> Cerebras. Made
# configurable via FALLBACK_ORDER in .env (comma-separated, e.g.
# "cerebras,openrouter,gemini") so you can re-rank tiers based on which
# ones are actually reliable for you without touching code — e.g. if Groq
# is hitting its daily limit and Gemini is erroring out for you, put
# Cerebras/OpenRouter first. Ollama always stays last (it's the local,
# no-network-required option, so it doubles as the final safety net).
# Unknown/misspelled names in FALLBACK_ORDER are ignored with a warning;
# any enabled tier you forget to list is appended at the end.
# ═══════════════════════════════════════
_FALLBACK_TIERS = {
    "gemini":     ("Gemini",     _call_gemini,     _call_gemini_stream),
    "openrouter": ("OpenRouter", _call_openrouter, _call_openrouter_stream),
    "cerebras":   ("Cerebras",   _call_cerebras,   _call_cerebras_stream),
    "mistral":    ("Mistral",    _call_mistral,    _call_mistral_stream),
}
# Reordered from the old cerebras-first default: rag_log.txt (Aug 21) shows
# Cerebras returning 402 Payment Required — the free tier is exhausted, a
# billing wall, not a transient failure — so it's moved to last (kept, not
# removed, in case the account gets funded later). Mistral added as a
# genuinely free 4th option (~1B tokens/month, no card) to replace it in the
# middle of the chain. Override via FALLBACK_ORDER in .env any time.
_DEFAULT_FALLBACK_ORDER = ["gemini", "mistral", "openrouter", "cerebras"]


def _build_fallback_order() -> list:
    raw = os.getenv("FALLBACK_ORDER", "")
    requested = [t.strip().lower() for t in raw.split(",") if t.strip()] or _DEFAULT_FALLBACK_ORDER

    order = []
    for name in requested:
        if name not in _FALLBACK_TIERS:
            log.warning(f"FALLBACK_ORDER: unknown tier '{name}' — ignoring. Valid tiers: {list(_FALLBACK_TIERS)}")
            continue
        if name not in order:
            order.append(name)
    for name in _FALLBACK_TIERS:  # anything left off the list still runs, just last
        if name not in order:
            order.append(name)
    return order + ["ollama"]  # Ollama is always last


_FALLBACK_ORDER = _build_fallback_order()
log.info(f"Fallback chain order (after Groq): {' -> '.join(_FALLBACK_ORDER)}")


def _call_fallback_chain(prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
    """Tries each fallback tier in FALLBACK_ORDER, returns the first
    non-empty answer. Returns '' only if every tier is
    disabled/unreachable/empty."""
    for name in _FALLBACK_ORDER:
        fn = _call_ollama if name == "ollama" else _FALLBACK_TIERS[name][1]
        text = fn(prompt, max_tokens=max_tokens, temperature=temperature)
        if text:
            return text
    return ""


def _call_fallback_chain_stream(prompt: str, max_tokens: int = 500, temperature: float = 0.0):
    """Tries each fallback tier's streaming variant in FALLBACK_ORDER,
    yielding from the first one that produces a real answer, then stopping.

    Buffers each tier's first MIN_STREAM_FLUSH_CHARS before forwarding
    anything — a connection that dies mid-stream (confirmed in
    rag_log.txt: Gemini logged "Streamed answer via Gemini fallback" with
    no error after yielding only "* **Cow", 5 chars, then nothing —
    requests' iter_lines() just stops silently on a severed connection
    instead of raising) used to get treated as a complete, cacheable
    answer because "yielded at least 1 character" was the only success
    check. Now: if a tier dies before the buffer fills, that's treated as
    a failed attempt for THIS tier and we move to the next one instead of
    showing the user a stub. Once the buffer fills, it flushes and the
    rest streams live as before — this only changes behavior for streams
    that die very early.
    """
    MIN_STREAM_FLUSH_CHARS = 40
    for name in _FALLBACK_ORDER:
        display_name, fn = ("Ollama", _call_ollama_stream) if name == "ollama" else _FALLBACK_TIERS[name][::2]
        buffer = ""
        flushed = False
        for piece in fn(prompt, max_tokens=max_tokens, temperature=temperature):
            if flushed:
                yield piece
                continue
            buffer += piece
            if len(buffer) >= MIN_STREAM_FLUSH_CHARS:
                flushed = True
                yield buffer
        if flushed:
            log.info(f"Streamed answer via {display_name} fallback.")
            return
        if buffer:
            # Died before reaching the flush threshold — never shown to the
            # user. Log what we discarded so a real short-answer case (rare
            # but possible) is still visible in the log for debugging,
            # instead of vanishing silently.
            log.warning(
                f"{display_name} fallback stream died after only {len(buffer)} chars "
                f"({buffer!r}) — treating as failed, trying next fallback tier."
            )

# ═══════════════════════════════════════
# GROQ REQUEST THROTTLE
# Your logs show Groq itself asking for 33s+ back-off on the fast model —
# that's a real quota ceiling, not bad luck. Reacting after a 429 (the old
# capped-wait retry) just fails fast instead of avoiding the 429. This
# throttle proactively spaces requests so you stay under the per-minute
# limit in the first place. Tune GROQ_MAX_REQUESTS_PER_MIN to match your
# actual Groq dashboard quota for the model(s) you use most.
# ═══════════════════════════════════════
class _GroqThrottle:
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._times = []
        self._lock = threading.Lock()

    def wait_if_needed(self):
        with self._lock:
            now = time.time()
            self._times = [t for t in self._times if now - t < 60]
            if len(self._times) >= self.max_per_minute:
                wait = (self._times[0] + 60) - now
                if wait > 0:
                    log.info(f"Groq throttle: pacing request, waiting {wait:.1f}s "
                              f"({len(self._times)}/{self.max_per_minute} in last 60s)")
                    time.sleep(wait)
                    now = time.time()
                    self._times = [t for t in self._times if now - t < 60]
            self._times.append(now)

GROQ_MAX_REQUESTS_PER_MIN = int(os.getenv("GROQ_MAX_REQUESTS_PER_MIN", "50"))
# Per-model throttles — Groq's rate limit is a SEPARATE bucket per model, so
# one shared throttle either over-restricts the model with headroom or
# under-restricts the one that's actually near its ceiling. Each model gets
# its own pacing counter instead.
_groq_throttles = {}
_groq_throttles_lock = threading.Lock()

def _throttle_for(model: str) -> "_GroqThrottle":
    with _groq_throttles_lock:
        t = _groq_throttles.get(model)
        if t is None:
            t = _GroqThrottle(GROQ_MAX_REQUESTS_PER_MIN)
            _groq_throttles[model] = t
        return t

# ═══════════════════════════════════════
# PER-MODEL COOLDOWN (circuit breaker)
# Logs show the SAME model getting 429'd twice in a row within ~1.5s (e.g.
# two llama-3.1-8b-instant 429s back to back during one CSV lookup + its
# self-retry). Once Groq has told us a model's bucket is empty, hitting it
# again a few hundred ms later is guaranteed to fail again — it just wastes
# a network round trip and adds latency for no benefit. This tracks "model X
# is known-exhausted until timestamp Y" in memory so subsequent calls
# (including the query_csv_with_groq() self-retry) skip the dead model
# instantly instead of re-discovering the same 429 the hard way.
# ═══════════════════════════════════════
_model_cooldown = {}
_model_cooldown_lock = threading.Lock()
MODEL_COOLDOWN_CAP = float(os.getenv("MODEL_COOLDOWN_CAP", "20.0"))  # never blackout longer than this

def _model_cooldown_remaining(model: str) -> float:
    with _model_cooldown_lock:
        until = _model_cooldown.get(model, 0)
    remaining = until - time.time()
    return remaining if remaining > 0 else 0.0

def _set_model_cooldown(model: str, seconds: float):
    seconds = min(max(seconds, 0), MODEL_COOLDOWN_CAP)
    if seconds <= 0:
        return
    with _model_cooldown_lock:
        _model_cooldown[model] = time.time() + seconds
    log.info(f"Model '{model}' marked rate-limited; skipping it for the next {seconds:.1f}s.")

_groq_throttle = _throttle_for(MODEL)  # kept as a name for any external references

# ═══════════════════════════════════════
# DAILY TOKEN BUDGET GUARD (free-tier survival)
# Your Groq free tier caps tokens-per-day PER MODEL (100,000 TPD on the
# 70b model, seen in production: "Used 99835, Limit 100000"). The old code
# only reacted AFTER a 429 — by then the request that triggered it already
# failed and the user already waited out a rate-limit retry. This tracks
# real usage from every response and acts BEFORE the wall:
#   - past DOWNGRADE_RATIO of the daily budget: every call proactively uses
#     the small/fast model (separate quota bucket) instead of the 70b model
#   - past REFUSE_RATIO: stop calling Groq entirely and serve a clear,
#     cache-friendly message instead of guaranteed 429s that waste latency
#     and (with retries) waste more of the little budget that's left
# Single-process assumption (waitress, one worker) — resets at UTC midnight.
# ═══════════════════════════════════════
GROQ_DAILY_TOKEN_LIMIT  = int(os.getenv("GROQ_DAILY_TOKEN_LIMIT", "100000"))
BUDGET_DOWNGRADE_RATIO  = float(os.getenv("BUDGET_DOWNGRADE_RATIO", "0.70"))
BUDGET_REFUSE_RATIO     = float(os.getenv("BUDGET_REFUSE_RATIO", "0.96"))

BUDGET_EXHAUSTED_MSG = (
    "We've reached today's answer limit on our free service tier. "
    "Please try again after midnight (UTC), or rephrase using a question "
    "that's likely already been asked — cached answers still work."
)

# Set to False automatically the first time the installed `groq` SDK
# rejects stream_options (older versions don't have it) — see
# call_groq_stream(). Kept module-level so the detection persists for the
# rest of this process's life instead of re-failing on every request.
_stream_options_supported = True

def _estimate_tokens(text: str) -> int:
    """Rough fallback estimate (~4 chars/token) for the rare case a Groq
    response doesn't include a usage block. Only used as a fallback —
    real calls record the exact usage.total_tokens Groq reports."""
    return max(1, len(text or "") // 4)

class _DailyTokenBudget:
    """In-memory counter — reads (ratio/should_downgrade/should_refuse) stay
    lock-only, no DB round trip, so they add zero latency to the hot path.
    Writes (record) update memory immediately and fire a background thread
    to persist the increment to Postgres, so a restart/crash doesn't lose
    the day's usage (see _restore_token_budget() below, called once at
    import time after the DB pool exists)."""
    def __init__(self, daily_limit: int):
        self.daily_limit = daily_limit
        self._used = 0
        self._day = datetime.now(timezone.utc).date()
        self._lock = threading.Lock()

    def _maybe_reset(self):
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            log.info(f"Token budget reset for new day ({self._used} used yesterday).")
            self._day = today
            self._used = 0

    def restore(self, used: int):
        """One-time load of today's already-spent tokens from Postgres,
        called right after startup. Uses max() defensively in case a call
        was already recorded in-memory in the brief window before restore
        runs."""
        with self._lock:
            self._maybe_reset()
            if used > self._used:
                self._used = used
                log.info(f"Restored token budget from DB: {used}/{self.daily_limit} already used today.")

    def record(self, tokens: int):
        tokens = max(0, int(tokens))
        if not tokens:
            return
        with self._lock:
            self._maybe_reset()
            self._used += tokens
            day = self._day
        _persist_budget_increment_async(day, tokens)

    def ratio(self) -> float:
        with self._lock:
            self._maybe_reset()
            return (self._used / self.daily_limit) if self.daily_limit else 0.0

    def status(self) -> dict:
        with self._lock:
            self._maybe_reset()
            return {"used": self._used, "limit": self.daily_limit,
                    "ratio": round(self._used / self.daily_limit, 3) if self.daily_limit else 0.0}

    def should_downgrade(self) -> bool:
        return self.ratio() >= BUDGET_DOWNGRADE_RATIO

    def should_refuse(self) -> bool:
        return self.ratio() >= BUDGET_REFUSE_RATIO

_token_budget = _DailyTokenBudget(GROQ_DAILY_TOKEN_LIMIT)

def get_budget_status() -> dict:
    """Exposed for app.py's /health endpoint so you can watch remaining
    daily quota instead of finding out it's gone from a wall of 429s."""
    return _token_budget.status()

# ═══════════════════════════════════════
# SECURITY: SAFE PANDAS CODE EVALUATOR
# ═══════════════════════════════════════
ALLOWED_NAMES = {
    "dataframes", "pd", "len", "sum", "round", "str",
    "min", "max", "abs", "int", "float", "sorted"
}

ALLOWED_NODE_TYPES = (
    ast.Expression, ast.Load,
    ast.Call, ast.Attribute, ast.Subscript, ast.Index, ast.Slice,
    ast.Name, ast.Constant,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.And, ast.Or, ast.Not,
    # BitOr/BitAnd/BitXor/Invert: NOT actual bitwise math here — this is how
    # pandas combines boolean Series masks (df[(mask1) | (mask2)]). Python's
    # `and`/`or` keywords can't be used for this (they raise "truth value of
    # a Series is ambiguous"), so the LLM-generated code legitimately needs
    # `|` `&` `~` for any multi-condition filter. These operate purely on
    # already-sandboxed DataFrame/Series/bool values within this restricted
    # eval — they carry no extra risk beyond what BinOp/Compare already allow.
    ast.BitOr, ast.BitAnd, ast.BitXor, ast.Invert,
    ast.List, ast.Tuple, ast.Dict, ast.keyword,
)

# Explicitly named for defense-in-depth: even though these node types are already
# excluded by the ALLOWED_NODE_TYPES allowlist below (belt-and-suspenders — an
# allowlist alone should already reject them), we check for them by name first
# so a future change to ALLOWED_NODE_TYPES can't silently reopen this gap, and
# so the resulting error message is explicit about *why* it was rejected.
DISALLOWED_LOOP_NODE_TYPES = (
    ast.For, ast.While, ast.AsyncFor,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
)

class UnsafeCodeError(Exception):
    pass

def validate_ast(code: str) -> ast.Expression:
    try:
        # mode="eval" only accepts a single expression — statement-level loops
        # (for/while) are already a SyntaxError here and never reach ast.walk.
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        raise UnsafeCodeError(f"Not a valid expression: {e}")

    for node in ast.walk(tree):
        # No loops or comprehensions of any kind — a filter/aggregation
        # expression never needs one, and disallowing them removes both the
        # DoS risk of a user-crafted unbounded loop and the Windows-fallback
        # daemon-thread accumulation risk from a slow-but-whitelisted loop.
        if isinstance(node, DISALLOWED_LOOP_NODE_TYPES):
            raise UnsafeCodeError(f"Loops/comprehensions are not allowed: {type(node).__name__}")
        if not isinstance(node, ALLOWED_NODE_TYPES):
            raise UnsafeCodeError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            raise UnsafeCodeError(f"Disallowed name: '{node.id}'")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise UnsafeCodeError(f"Disallowed attribute: '{node.attr}'")

    return tree

class PandasTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise PandasTimeoutError("Pandas expression exceeded time limit")

# Bounded pool for the Windows (non-SIGALRM) timeout fallback. Using a fixed-size
# shared pool — instead of spawning a fresh threading.Thread per call — caps how
# many "orphaned" workers can pile up in the background if expressions keep timing
# out: at most _PANDAS_TIMEOUT_POOL_WORKERS run concurrently, further submissions
# simply queue behind them rather than adding unbounded new OS threads.
_PANDAS_TIMEOUT_POOL_WORKERS = int(os.getenv("PANDAS_TIMEOUT_POOL_WORKERS", "4"))
_pandas_timeout_pool = ThreadPoolExecutor(
    max_workers=_PANDAS_TIMEOUT_POOL_WORKERS,
    thread_name_prefix="pandas-timeout"
)

def _eval_with_thread_timeout(compiled, safe_env, timeout_seconds):
    """
    Fallback timeout mechanism for platforms without SIGALRM (e.g. Windows).
    Submits the eval to a small bounded thread pool and waits up to
    timeout_seconds for it to finish. Note: unlike signal.alarm, this can't
    forcibly kill a running eval — a timed-out expression keeps executing in
    the background until it finishes on its own — but the bounded pool size
    means that background work can never exceed _PANDAS_TIMEOUT_POOL_WORKERS
    concurrent threads, so repeated timeouts degrade to queuing delay rather
    than unbounded CPU/RAM growth.
    """
    def _target():
        return eval(compiled, {"__builtins__": {}}, safe_env)

    future = _pandas_timeout_pool.submit(_target)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        raise PandasTimeoutError("Pandas expression exceeded time limit")


def safe_eval_pandas(code: str, safe_env: dict, timeout_seconds: int = 5):
    tree = validate_ast(code)
    compiled = compile(tree, "<safe_pandas_expr>", "eval")

    # Hard wall-clock cap so a whitelisted-but-expensive expression (e.g. a
    # sort/merge over a big frame) can't hang a worker. Unix uses signal.alarm
    # (can actually interrupt the eval); on platforms without SIGALRM (Windows)
    # we fall back to a bounded thread-pool timeout so the app stays portable.
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            result = eval(compiled, {"__builtins__": {}}, safe_env)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        result = _eval_with_thread_timeout(compiled, safe_env, timeout_seconds)


    # Cap output size so a huge unfiltered frame can't blow up the prompt/response.
    if isinstance(result, (pd.Series, pd.DataFrame)) and len(result) > 500:
        log.warning(f"Pandas result truncated: {len(result)} rows -> 500")
        result = result.head(500)

    return result

# ═══════════════════════════════════════
# DATABASE — Pooled Connections
# ═══════════════════════════════════════
# Render/Railway/Heroku-style hosts hand you a single DATABASE_URL instead
# of separate DB_HOST/DB_PORT/etc. Support both so the same code runs
# locally (.env with DB_HOST etc.) and on a managed host (DATABASE_URL only).
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

def _db_connect_kwargs() -> dict:
    if DATABASE_URL:
        return {"dsn": DATABASE_URL}
    return {
        "host":     os.getenv("DB_HOST"),
        "port":     os.getenv("DB_PORT"),
        "dbname":   os.getenv("DB_NAME"),
        "user":     os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

def get_db():
    return psycopg2.connect(**_db_connect_kwargs())

_DB_POOL = None

def _init_db_pool():
    global _DB_POOL
    try:
        _DB_POOL = psycopg2.pool.ThreadedConnectionPool(
            DB_POOL_MIN, DB_POOL_MAX,
            **_db_connect_kwargs()
        )
        log.info(f"DB connection pool ready (min={DB_POOL_MIN}, max={DB_POOL_MAX})")
    except Exception as e:
        log.error(f"DB pool init failed, falling back to per-call connections: {e}")
        _DB_POOL = None

_init_db_pool()

@contextmanager
def db_cursor(commit: bool = False):
    conn = None
    from_pool = _DB_POOL is not None
    cur = None
    try:
        conn = _DB_POOL.getconn() if from_pool else get_db()
        cur = conn.cursor()
        yield cur
        if commit:
            conn.commit()
    except psycopg2.Error:
        if conn is not None:  # guard: connection may never have been established
            conn.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            if from_pool:
                _DB_POOL.putconn(conn)
            else:
                conn.close()

def ensure_cache_table():
    try:
        with db_cursor(commit=True) as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    id           SERIAL PRIMARY KEY,
                    question_hash VARCHAR(64) UNIQUE NOT NULL,
                    answer        TEXT NOT NULL,
                    source_doc    VARCHAR(255),
                    source_page   INTEGER DEFAULT 0,
                    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            # question_text: the table only ever stored the hash, not the
            # original text. That made a stuck bad cache entry (e.g. the
            # Dec-EFY-2018 inflation question, cached before the pdf-routing
            # fix landed and never regenerated since — ON CONFLICT DO
            # NOTHING below meant it could never self-heal) impossible to
            # find with a keyword search — you'd have to already know the
            # exact question text and recompute its hash by hand. Nullable
            # + backfilled going forward only; existing rows keep NULL here
            # and are still purgeable via purge_stale_cache.py (matches by
            # recomputed hash instead).
            cur.execute("""
                ALTER TABLE query_cache
                ADD COLUMN IF NOT EXISTS question_text TEXT
            """)
        log.info("Cache table ready")
    except Exception as e:
        log.error(f"Cache table error: {str(e)}")

# ═══════════════════════════════════════
# TOKEN BUDGET PERSISTENCE (survives restarts/crashes)
# Fixes: the guard above only lived in process memory, so every restart
# (24 seen in app_log.txt during dev) reset it to 0 while your real Groq
# account usage kept climbing — the guard never actually fired before the
# real 429 wall. One row per UTC day, incremented atomically.
# ═══════════════════════════════════════
def ensure_budget_table():
    try:
        with db_cursor(commit=True) as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS token_budget_daily (
                    day         DATE PRIMARY KEY,
                    tokens_used INTEGER NOT NULL DEFAULT 0,
                    updated_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
        log.info("Token budget table ready")
    except Exception as e:
        log.error(f"Token budget table error: {str(e)}")


def _load_budget_from_db(day) -> int:
    try:
        with db_cursor() as cur:
            cur.execute("SELECT tokens_used FROM token_budget_daily WHERE day = %s", (day,))
            row = cur.fetchone()
            return row[0] if row else 0
    except Exception as e:
        log.warning(f"Could not load persisted token budget ({e}); starting from 0 for this process.")
        return 0


def _persist_budget_increment_async(day, tokens: int):
    """Fire-and-forget: runs in a background thread so it never adds DB
    latency to the request that's waiting on the actual Groq answer. Worst
    case on a crash mid-write is losing a few hundred tokens of tracking —
    harmless, since Groq's own limit is still the real backstop."""
    def _write():
        try:
            with db_cursor(commit=True) as cur:
                cur.execute("""
                    INSERT INTO token_budget_daily (day, tokens_used)
                    VALUES (%s, %s)
                    ON CONFLICT (day) DO UPDATE
                        SET tokens_used = token_budget_daily.tokens_used + EXCLUDED.tokens_used,
                            updated_at  = NOW()
                """, (day, tokens))
        except Exception as e:
            log.warning(f"Failed to persist token budget increment ({e}); in-memory count for this process is still accurate.")
    threading.Thread(target=_write, daemon=True).start()


# Restore now that db_cursor/_DB_POOL exist. (_token_budget itself was
# created earlier, near GROQ_DAILY_TOKEN_LIMIT, before the DB pool was
# available — see the comment on _DailyTokenBudget.)
ensure_budget_table()
_token_budget.restore(_load_budget_from_db(_token_budget._day))

# ═══════════════════════════════════════
# CACHE FUNCTIONS
# ═══════════════════════════════════════
def _normalize_question(question: str) -> str:
    q = question.lower().strip()
    q = re.sub(r"[^\w\s\u1200-\u137F]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()

def get_question_hash(question: str, language: str = "en") -> str:
    key = f"{language}:{_normalize_question(question)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def check_cache(question: str, language: str = "en"):
    try:
        q_hash = get_question_hash(question, language)
        with db_cursor(commit=True) as cur:
            cur.execute(
                "SELECT answer, source_doc, source_page, created_at FROM query_cache WHERE question_hash = %s",
                (q_hash,)
            )
            row = cur.fetchone()

            if row and CACHE_TTL_DAYS > 0 and row[3] is not None:
                # row[3] comes back naive when the DB column/driver doesn't
                # preserve tz — datetime.now(tz) is always aware, so naive - aware
                # raised "can't subtract offset-naive and offset-aware datetimes"
                # (seen repeatedly in rag_log.txt). Normalize both sides to aware
                # UTC before subtracting instead of deriving tz from the row.
                created_at = row[3]
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - created_at
                if age > timedelta(days=CACHE_TTL_DAYS):
                    cur.execute("DELETE FROM query_cache WHERE question_hash = %s", (q_hash,))
                    return None

            if row:
                log.info(f"Cache hit: '{question[:50]}' [{language}]")
                return {"answer": row[0], "source": row[1], "page": row[2] if row[2] is not None else 0, "cached": True, "route": "cache", "language": language}
            return None
    except Exception as e:
        log.error(f"Cache check error: {str(e)}")
        return None

# ═══════════════════════════════════════
# ETHIOPIAN CALENDAR (EFY) CONTEXT
# ═══════════════════════════════════════
# ESS reports are labeled by Ethiopian Fiscal Year (EFY / Ethiopian calendar),
# not the Gregorian calendar. Ethiopian New Year falls ~Sept 11 Gregorian, and
# the Ethiopian year is 7-8 years behind. Without this, a question like
# "inflation rate in 2018" gets compared against report labels like "EFY2018"
# with no indication that EFY2018 is actually ~Sept 2025-Sept 2026 (i.e. the
# CURRENT period), not eight years ago — the model just says "not mentioned"
# with no explanation, which reads as a retrieval failure to the user when
# it's really a calendar mismatch.
def _current_efy_context() -> str:
    """One-line, computed-not-guessed EFY/Gregorian reference for prompts."""
    now = datetime.now()
    new_year_cutoff = datetime(now.year, 9, 11)
    if now >= new_year_cutoff:
        efy_year = now.year - 7
        efy_start, efy_end = now.year, now.year + 1
    else:
        efy_year = now.year - 8
        efy_start, efy_end = now.year - 1, now.year
    return (
        f"Today (Gregorian {now.strftime('%Y-%m-%d')}) falls in Ethiopian Fiscal Year "
        f"(EFY) {efy_year}, which runs roughly Sept {efy_start}-Sept {efy_end}. "
        f"General rule: EFY N ~= Gregorian (N+7) to (N+8), starting ~Sept."
    )


_EFY_RULE = (
    "When a user asks for a year in Gregorian calendar (e.g., 2018), explicitly check and map "
    "it to the corresponding Ethiopian Fiscal Year (EFY) before searching the PDF chunks, and "
    "state both calendars in the response to avoid confusion. "
    "When the user asks about a year in the Gregorian calendar (e.g. \"2018\"), explicitly "
    "check and map it to the corresponding Ethiopian Fiscal Year (EFY) BEFORE deciding the "
    "report doesn't cover it — ESS reports label figures by EFY, not Gregorian year, and the "
    "two do not share digits. " + _current_efy_context() + " Do NOT say the year isn't "
    "mentioned just because you don't see that exact number as a Gregorian year in the text. "
    "Instead: (a) convert using the rule above, (b) give the EFY figure from the text if it's "
    "there, and (c) always state BOTH calendars in the response (e.g. \"in EFY 2010 "
    "(~2017/2018)\") so the user isn't confused by the year change. Write EFY years as plain "
    "Arabic digits (\"EFY 2010\"), never as Ge'ez/Ethiopic numeral characters (፳, ፻, etc.) — "
    "those are for the reader's own calendar app, not for a mixed English-Amharic answer."
)


def _sanitize_text(text: str) -> str:
    """Repairs mojibake / corrupted Unicode (e.g. the "EFY 2010\ufffd\ufffd2011"
    tofu-box garbage seen when a PDF's embedded font has a broken/missing
    ToUnicode CMap, so PyMuPDF's extraction returns bad codepoints instead
    of the real character). resp.encoding = "utf-8" on the streaming
    fallback calls above fixes the classic "windows-1252-over-utf-8"
    mojibake pattern (â€"); this catches the other class — garbage baked
    directly into the source PDF's extracted text, which gets quoted
    verbatim into the model's context and then its answer, so pinning HTTP
    decoding can't touch it. Applied both to retrieved PDF chunks
    (retriever.py) and to the final answer text (defense in depth — covers
    anything a model itself introduces too). No-op (returns text
    unchanged) if ftfy isn't installed — never crashes the app for a
    missing optional dependency; log once so it's not a silent gap.
    """
    if not text or ftfy is None:
        return text
    return ftfy.fix_text(text)


def is_bad_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    a = answer.strip().lower()
    if answer.strip() in (GROQ_UNAVAILABLE_MSG, BUDGET_EXHAUSTED_MSG):
        return True
    if a in ("none", "n/a", "error", "null"):
        return True
    # Catches a stream that died mid-answer past the 40-char buffer in
    # _call_fallback_chain_stream (e.g. cut off at 150 chars, mid-sentence)
    # — the buffer only protects the first 40 chars, a later drop still
    # needs catching here. Heuristic, not exact: very short AND ends
    # without any sentence-ending punctuation or a closed markdown
    # bold/table marker is a strong truncation signal for this app's
    # answer style (every real answer is at least a full sentence).
    stripped = answer.strip()
    if len(stripped) < 80 and not stripped.endswith((".", "!", "?", "%", ":", ")")):
        return True
    # The model's own "I looked, nothing was in the retrieved text" sentinel
    # (see the PDF/hybrid prompts' rule 5). This is a legitimate answer to
    # show the user once, but must never be cached — the next question on
    # the same topic deserves a fresh retrieval attempt, not a permanently
    # stuck "not found" from one weak chunk pull.
    # NOTE: the prompts never mandate one exact phrase, so the model varies
    # its wording ("No specific figure...is provided", "isn't in the
    # retrieved sources", "not mentioned in the report", etc.). A narrow
    # 2-phrase check let most of these slip through and get cached as if
    # they were real answers — this is why some questions kept returning
    # the same stale miss for weeks. Cast a wider net instead.
    _NOT_FOUND_PATTERNS = (
        "nothing relevant", "no relevant", "not relevant",
        "not provided", "isn't provided", "is not provided",
        "not available", "isn't available", "is not available",
        "no specific figure", "no specific data", "no specific number",
        "not mentioned", "isn't mentioned", "is not mentioned",
        "not found in", "not present in", "isn't present in",
        "doesn't contain", "does not contain",
        "isn't in the", "is not in the", "not in the retrieved",
        "no data", "not specified", "isn't specified",
        # Widened after finding the Dec-EFY-2018 inflation question stuck
        # on a cached miss for weeks (rag_log.txt, repeated "Cache hit"
        # from Aug 14 through Aug 21): the original answer used one of
        # these equivalent phrasings, none of which the old list caught,
        # so it slipped past this check, got cached, and — because
        # save_cache's INSERT used ON CONFLICT DO NOTHING — could never be
        # overwritten by a later, correct regeneration.
        "does not include", "doesn't include", "does not detail",
        "doesn't detail", "does not specify", "doesn't specify",
        "couldn't find", "could not find", "unable to find",
        "unable to locate", "no mention of", "no information about",
        "no information on", "lacks information", "not indicated",
        "not stated", "wasn't stated", "was not stated",
    )
    if any(p in a for p in _NOT_FOUND_PATTERNS):
        return True
    return False

def save_cache(question: str, answer: str, source: str, page: int, language: str = "en"):
    if is_bad_answer(answer):
        log.warning(f"Skipped caching bad/sentinel answer for: '{question[:50]}'")
        return
    answer = _sanitize_text(answer)
    try:
        safe_page = int(page) if (page is not None and str(page).isdigit()) else 0
        q_hash = get_question_hash(question, language)
        with db_cursor(commit=True) as cur:
            # Was ON CONFLICT DO NOTHING: once a hash had ANY row (even one
            # that slipped past the old, narrower is_bad_answer() net — see
            # the widened pattern list above), it was permanently frozen —
            # a later, correct regeneration for the same question could
            # never overwrite it. This is what stuck "What was the
            # inflation rate in Ethiopia in 2018?" on a stale cache entry
            # from before the pdf-routing fix, for a week+. DO UPDATE lets
            # a good answer heal a bad one the next time this exact
            # question is asked and the cache TTL check (or a manual purge)
            # forces a fresh generation.
            cur.execute("""
                INSERT INTO query_cache (question_hash, answer, source_doc, source_page, question_text, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (question_hash) DO UPDATE
                    SET answer        = EXCLUDED.answer,
                        source_doc    = EXCLUDED.source_doc,
                        source_page   = EXCLUDED.source_page,
                        question_text = EXCLUDED.question_text,
                        created_at    = NOW()
            """, (q_hash, answer, source, safe_page, question[:500]))
    except Exception as e:
        log.error(f"Cache save error: {str(e)}")

# ═══════════════════════════════════════
# GROQ CALL
# ═══════════════════════════════════════
# Matches Groq's rate-limit error body in either unit it uses:
# "...Please try again in 8.0s." or "...Please try again in 370ms."
_RETRY_AFTER_RE = re.compile(r"try again in\s+([\d.]+)\s*(ms|s)\b", re.IGNORECASE)

def call_groq(prompt: str, retries: int = 2, max_tokens: int = 500, model: str = None, temperature: float = 0.0) -> str:
    use_model = model or MODEL

    if _token_budget.should_refuse():
        log.warning(f"Daily token budget exhausted ({_token_budget.ratio():.0%}) — skipping Groq, trying fallback chain.")
        return _call_fallback_chain(prompt, max_tokens=max_tokens, temperature=temperature) or BUDGET_EXHAUSTED_MSG
    if _token_budget.should_downgrade() and use_model != FAST_MODEL:
        log.info(f"Token budget at {_token_budget.ratio():.0%} — proactively using {FAST_MODEL} instead of {use_model}.")
        use_model = FAST_MODEL

    for attempt in range(retries):
        # Circuit breaker: if we already know this model is rate-limited
        # (from a 429 seen moments ago, possibly by another concurrent
        # request), don't pay for another network round trip that will just
        # 429 again. Switch to FAST_MODEL immediately if possible, otherwise
        # skip straight to the next attempt / give up.
        cooldown = _model_cooldown_remaining(use_model)
        if cooldown > 0:
            if use_model != FAST_MODEL and _model_cooldown_remaining(FAST_MODEL) == 0:
                log.info(f"Skipping known-rate-limited model '{use_model}' ({cooldown:.1f}s left), using {FAST_MODEL} instead.")
                use_model = FAST_MODEL
            else:
                log.warning(f"Model '{use_model}' still cooling down ({cooldown:.1f}s left) — skipping call.")
                if attempt < retries - 1:
                    continue
                return _call_fallback_chain(prompt, max_tokens=max_tokens, temperature=temperature) or GROQ_UNAVAILABLE_MSG

        _throttle_for(use_model).wait_if_needed()
        try:
            response = groq_client.chat.completions.create(
                model             = use_model,
                messages          = [{"role": "user", "content": prompt}],
                max_tokens        = max_tokens,
                temperature       = temperature,
                timeout           = GROQ_REQUEST_TIMEOUT,
                reasoning_effort  = "low",
                include_reasoning = False,
            )
            usage = getattr(response, "usage", None)
            total = getattr(usage, "total_tokens", None) if usage else None
            answer = response.choices[0].message.content.strip()
            _token_budget.record(total if total else _estimate_tokens(prompt) + _estimate_tokens(answer))
            return answer
        except Exception as e:
            is_rate_limit = getattr(e, "status_code", None) == 429 or "429" in str(e) or "rate_limit_exceeded" in str(e)
            if is_rate_limit:
                # Honor Groq's own suggested wait time (parsed from the error
                # body) rather than blind exponential backoff — it's telling
                # us exactly when the token bucket refills. Record the FULL
                # suggested duration as this model's cooldown (so other
                # callers skip it too), but still cap our OWN wait so one
                # slow-to-refill window doesn't block this single request for
                # the full suggested duration; a capped-out wait means we
                # give up faster and let the caller's own retry/fallback path
                # (e.g. the pandas-correction retry, or the PDF-only fallback
                # in _gather_sources) take over instead of stalling.
                m = _RETRY_AFTER_RE.search(str(e))
                if m:
                    value, unit = float(m.group(1)), m.group(2).lower()
                    suggested = value / 1000.0 if unit == "ms" else value
                else:
                    suggested = float(2 ** attempt)
                _set_model_cooldown(use_model, suggested)
                wait = min(suggested, GROQ_RATE_LIMIT_MAX_WAIT)
                log.warning(
                    f"Groq rate limit (429) asked for {suggested:.1f}s, capping wait to {wait:.1f}s "
                    f"(attempt {attempt+1}/{retries}, model={use_model})"
                )
                # Different models draw from different Groq quota buckets —
                # if the big model is out of quota, switch to FAST_MODEL for
                # the remaining attempts instead of retrying the same
                # exhausted bucket. Answer quality may drop slightly, but
                # this is the difference between an actual answer and
                # "Service temporarily unavailable."
                if use_model != FAST_MODEL:
                    log.warning(f"Falling back to {FAST_MODEL} for remaining attempts (rate-limited on {use_model}).")
                    use_model = FAST_MODEL
                    wait = 0  # new bucket, no need to also sleep before trying it
            else:
                wait = float(2 ** attempt)
                log.warning(f"Groq call failed (attempt {attempt+1}/{retries}, model={use_model}): {e}")
            if attempt < retries - 1 and wait > 0:
                time.sleep(wait)
    # All Groq retries exhausted (rate-limited on every model, or erroring) —
    # walk the rest of the fallback chain (Gemini -> OpenRouter -> Cerebras
    # -> Ollama) before giving up entirely.
    return _call_fallback_chain(prompt, max_tokens=max_tokens, temperature=temperature) or GROQ_UNAVAILABLE_MSG

def call_groq_stream(prompt: str, max_tokens: int = 500, model: str = None, temperature: float = 0.0):
    use_model = model or MODEL

    if _token_budget.should_refuse():
        log.warning(f"Daily token budget exhausted ({_token_budget.ratio():.0%}) — skipping Groq stream, trying fallback chain.")
        yielded_any = False
        for piece in _call_fallback_chain_stream(prompt, max_tokens=max_tokens, temperature=temperature):
            yielded_any = True
            yield piece
        if not yielded_any:
            yield BUDGET_EXHAUSTED_MSG
        return
    if _token_budget.should_downgrade() and use_model != FAST_MODEL:
        log.info(f"Token budget at {_token_budget.ratio():.0%} — proactively streaming {FAST_MODEL} instead of {use_model}.")
        use_model = FAST_MODEL

    # Single retry, and ONLY when the failure happens before any content has
    # been yielded to the caller. Once even one delta has reached the client,
    # retrying would mean starting a second stream from scratch and duplicating
    # (or garbling) what's already been shown — so a mid-stream failure still
    # falls straight through to GROQ_UNAVAILABLE_MSG via the caller, same as
    # before. This only covers the common case: the 429 fires on connect,
    # before a single token comes back.
    for attempt in range(2):
        yielded_any = False
        streamed_chars = 0

        cooldown = _model_cooldown_remaining(use_model)
        if cooldown > 0:
            if use_model != FAST_MODEL and _model_cooldown_remaining(FAST_MODEL) == 0:
                log.info(f"Skipping known-rate-limited model '{use_model}' ({cooldown:.1f}s left), streaming {FAST_MODEL} instead.")
                use_model = FAST_MODEL
            elif use_model == FAST_MODEL and _model_cooldown_remaining(MODEL) == 0:
                # Mirror of the block above: the token-budget downgrade above
                # picked FAST_MODEL, but FAST_MODEL is the one cooling down
                # right now (e.g. it was just rate-limited by an earlier CSV
                # pandas-generation call moments ago) while the big MODEL is
                # completely free. Falling straight to GROQ_UNAVAILABLE_MSG
                # here would be wrong when a working model is available —
                # budget-saving is a nice-to-have, answering the user isn't
                # optional.
                log.info(f"Skipping known-rate-limited model '{use_model}' ({cooldown:.1f}s left), streaming {MODEL} instead.")
                use_model = MODEL
            elif attempt == 0:
                log.warning(f"Model '{use_model}' still cooling down ({cooldown:.1f}s left) — skipping stream attempt.")
                continue
            else:
                # Both attempts blocked by cooldown, nothing streamed yet —
                # walk the fallback chain (Gemini -> OpenRouter -> Cerebras
                # -> Ollama); _call_fallback_chain_stream logs which one hit.
                for piece in _call_fallback_chain_stream(prompt, max_tokens=max_tokens, temperature=temperature):
                    yielded_any = True
                    yield piece
                return

        _throttle_for(use_model).wait_if_needed()
        try:
            create_kwargs = dict(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                timeout=GROQ_REQUEST_TIMEOUT,
                reasoning_effort="low",
                include_reasoning=False,
            )
            if _stream_options_supported:
                create_kwargs["stream_options"] = {"include_usage": True}
            try:
                stream = groq_client.chat.completions.create(**create_kwargs)
            except TypeError as te:
                # Older `groq` SDK versions don't accept stream_options —
                # detected once per process, then skipped from here on so
                # every request doesn't pay for the same failed call.
                # Token usage falls back to the char-based estimate below
                # (already the existing fallback path) — accuracy dips
                # slightly but the budget guard keeps working either way.
                # `pip install -U groq` restores exact usage tracking.
                if "stream_options" in str(te):
                    globals()["_stream_options_supported"] = False
                    log.warning(f"Installed groq SDK doesn't support stream_options — using estimated token counts from now on. Run 'pip install -U groq' for exact tracking. ({te})")
                    create_kwargs.pop("stream_options", None)
                    stream = groq_client.chat.completions.create(**create_kwargs)
                else:
                    raise
            usage_recorded = False
            for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage and getattr(chunk_usage, "total_tokens", None):
                    _token_budget.record(chunk_usage.total_tokens)
                    usage_recorded = True
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yielded_any = True
                        streamed_chars += len(delta)
                        yield delta
            if not usage_recorded:
                # Groq build/version didn't return a usage chunk — fall back
                # to a char-based estimate so the budget still tracks reality
                # closely enough to trigger the downgrade/refuse thresholds.
                _token_budget.record(_estimate_tokens(prompt) + streamed_chars // 4)
            return
        except Exception as e:
            is_rate_limit = getattr(e, "status_code", None) == 429 or "429" in str(e) or "rate_limit_exceeded" in str(e)
            if is_rate_limit:
                log.warning(f"Groq streaming call hit rate limit (429) (attempt {attempt+1}/2, model={use_model}): {e}")
            else:
                log.warning(f"Groq streaming call failed (attempt {attempt+1}/2, model={use_model}): {e}")

            if yielded_any:
                # Already streamed partial content from Groq — can't safely
                # restart on a different backend without duplicating/garbling
                # what the user already saw. Give up here as before.
                return

            if attempt == 1:
                # Final Groq attempt failed and nothing was streamed yet —
                # try the local fallback before giving up entirely. The
                # caller (get_answer_stream) treats a totally empty result as
                # failure and substitutes GROQ_UNAVAILABLE_MSG, so an empty
                # fallback chain (nothing configured / all down) degrades to
                # the exact old behavior.
                for piece in _call_fallback_chain_stream(prompt, max_tokens=max_tokens, temperature=temperature):
                    yield piece
                return

            # Rate limit on the big model, first failure, haven't already
            # switched: retry against FAST_MODEL instead. Different models
            # draw from different quota buckets on Groq, so this is a real
            # fallback (not just hoping the same exhausted bucket refills),
            # and it's the difference between a full outage and a slightly
            # lower-quality-but-working answer.
            if is_rate_limit and use_model != FAST_MODEL:
                log.warning(f"Falling back to {FAST_MODEL} for this request (primary model rate-limited).")
                use_model = FAST_MODEL
                continue

            if is_rate_limit:
                m = _RETRY_AFTER_RE.search(str(e))
                if m:
                    value, unit = float(m.group(1)), m.group(2).lower()
                    suggested = value / 1000.0 if unit == "ms" else value
                else:
                    suggested = 1.0
                _set_model_cooldown(use_model, suggested)
                wait = min(suggested, GROQ_RATE_LIMIT_MAX_WAIT)
            else:
                wait = 0.5
            time.sleep(wait)
            # loop around for the single retry

# ═══════════════════════════════════════
# LANGUAGE DETECTOR & SYSTEM INSTRUCTIONS
# ═══════════════════════════════════════
_AMHARIC_RE = re.compile(r"[\u1200-\u137F]")

def _contains_amharic_script(text: str) -> bool:
    return bool(_AMHARIC_RE.search(text or ""))

_ENGLISH_STOPWORDS = {
    "the", "is", "are", "what", "how", "many", "much", "population", "rate",
    "when", "where", "which", "average", "total", "percent", "percentage",
    "in", "of", "for", "was", "were", "and", "report", "data"
}

def detect_language(question: str) -> str:
    """Only 'am' (Amharic) and 'en' (English) are supported answer
    languages. Amharic script is the only reliable signal — anything else
    (including questions phrased in Afaan Oromoo) is answered in English
    rather than attempting an Oromo translation the model can't do well."""
    if _contains_amharic_script(question):
        return "am"
    return "en"


def detect_language_llm(question: str) -> str:
    """Fallback classifier for the rare ambiguous case (e.g. Amharic typed
    phonetically in Latin script). Cheap/fast model, tiny output."""
    prompt = f"""Identify the language of this question. It is ESS (Ethiopian Statistical Service)
chatbot input, so it will be English or Amharic (possibly typed phonetically in Latin letters).
Respond with exactly one lowercase code: en or am. No explanation.

Question: "{question}\""""
    raw = call_groq(prompt, retries=1, max_tokens=20, model=FAST_MODEL)
    code = re.sub(r"[^a-z]", "", (raw or "").lower().strip())
    return code if code in ("en", "am") else "en"


def resolve_language(question: str) -> str:
    """detect_language() is now fully deterministic (am or en), so this is
    just a thin wrapper kept for call-site compatibility."""
    return detect_language(question)

def get_language_instructions(lang_code: str) -> str:
    if lang_code == "am":
        return (
            "\nCRITICAL: Respond in Clear, Formal Amharic (አማርኛ) ONLY.\n"
            "- Extract the factual values from the English source materials and write your output in standard Amharic script.\n"
            "- Keep system terms, table titles, or specific document names in English if no direct Amharic translation exists."
        )
    return "\nCRITICAL: Respond in Clear, Formal English ONLY."

# ═══════════════════════════════════════
# TERMINOLOGY GLOSSARY (fill this in with ESS staff — native speakers only,
# never guess these yourself and never trust the LLM's own guess here).
# Lives in data/amharic_glossary.json (NOT hardcoded here) specifically so
# ESS staff can add/edit "English term": "Amharic translation" rows directly
# in a plain JSON file — no code changes, no redeploy needed for a wording
# fix. This file is auto-created with the two known-good defaults below on
# first run if it doesn't exist yet.
# Every term listed here is now FORCED (via placeholder substitution — see
# _apply_glossary_placeholders below) into BOTH translation paths: the
# Google/MyMemory NMT engines AND the Groq LLM fallback. Previously it was
# only injected into the Groq prompt, so any answer that succeeded via the
# NMT path (now the common case, post-chunking-fix) never got the forced
# wording at all.
# ═══════════════════════════════════════
GLOSSARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "amharic_glossary.json")

_DEFAULT_GLOSSARY_TERMS = {
    "Ethiopian Statistical Service": "የኢትዮጵያ ስታቲስቲክስ አገልግሎት",
    "Meher season": "መኸር",
    "Belg season": "በልግ",
}


def _load_glossary_terms() -> dict:
    """Loads {English term: Amharic translation} from GLOSSARY_FILE. Creates
    the file with the built-in defaults on first run if missing, so ESS
    staff have a real file to open and edit right away. Falls back to the
    in-memory defaults (never crashes the app) if the file is missing,
    unreadable, or malformed."""
    try:
        os.makedirs(os.path.dirname(GLOSSARY_FILE), exist_ok=True)
        if not os.path.exists(GLOSSARY_FILE):
            with open(GLOSSARY_FILE, "w", encoding="utf-8") as f:
                json.dump(_DEFAULT_GLOSSARY_TERMS, f, ensure_ascii=False, indent=2)
            log.info(f"Created {GLOSSARY_FILE} with {len(_DEFAULT_GLOSSARY_TERMS)} default glossary term(s). "
                      f"ESS staff can add more rows directly to this file.")
            return dict(_DEFAULT_GLOSSARY_TERMS)
        with open(GLOSSARY_FILE, "r", encoding="utf-8") as f:
            terms = json.load(f)
        if not isinstance(terms, dict):
            raise ValueError("glossary file must be a JSON object of {\"English term\": \"Amharic translation\"}")
        log.info(f"Loaded {len(terms)} glossary term(s) from {GLOSSARY_FILE}.")
        return terms
    except Exception as e:
        log.warning(f"Could not load {GLOSSARY_FILE} ({e}) — using built-in defaults only.")
        return dict(_DEFAULT_GLOSSARY_TERMS)


# Hot-reload cache: ESS staff will keep editing GLOSSARY_FILE after terms
# get confirmed, and this is a live public widget — requiring an app
# restart for every wording tweak isn't realistic. _get_glossary_terms()
# below re-reads the file only when its mtime changes (one cheap stat()
# call per use, not a re-parse every time), so an edit takes effect on the
# very next question with no deploy/restart needed. If the file is
# mid-write or briefly invalid, the last known-good version keeps serving
# until the next successful read — never a crash, never a gap.
_glossary_cache = {"terms": _load_glossary_terms(), "mtime": None}
try:
    _glossary_cache["mtime"] = os.path.getmtime(GLOSSARY_FILE) if os.path.exists(GLOSSARY_FILE) else None
except Exception:
    pass


def _get_glossary_terms() -> dict:
    """Returns the current glossary, auto-reloading from GLOSSARY_FILE when
    its mtime changes. Use this everywhere instead of a static dict."""
    try:
        mtime = os.path.getmtime(GLOSSARY_FILE) if os.path.exists(GLOSSARY_FILE) else None
    except Exception:
        mtime = _glossary_cache["mtime"]
    if mtime != _glossary_cache["mtime"]:
        reloaded = _load_glossary_terms()
        if reloaded:  # guards against a transient empty/mid-write read
            _glossary_cache["terms"] = reloaded
            _glossary_cache["mtime"] = mtime
            log.info(f"Glossary file changed — hot-reloaded {len(reloaded)} term(s), no restart needed.")
    return _glossary_cache["terms"]

# ═══════════════════════════════════════
# GOOGLE TRANSLATE (Amharic quality fix)
# ═══════════════════════════════════════
# Root cause of weaker Amharic answers: general LLMs (Groq's models
# included) are trained on far less Amharic than English, so no amount of
# prompt tuning closes the gap. Google's Translate models are dedicated NMT
# systems trained on much more Amharic parallel text, so they're used as the
# FIRST attempt now, with the existing Groq LLM translation kept as fallback.
#
# Two backends, pick with GOOGLE_TRANSLATE_MODE:
#   "free"  -> deep-translator (scrapes translate.google.com). No API key,
#              no cost, but no SLA — can get rate-limited/blocked under real
#              traffic. Fine for testing/low-traffic, NOT for a public
#              government-facing deploy long-term.
#   "cloud" -> official Google Cloud Translation API. Needs GOOGLE_TRANSLATE_API_KEY
#              (free tier: 500k chars/month, no charge until you exceed it).
#              pip install google-cloud-translate
#   "off"   -> skip Google Translate entirely, go straight to Groq (old behavior).
GOOGLE_TRANSLATE_MODE = os.getenv("GOOGLE_TRANSLATE_MODE", "free").lower()
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY", "")

_gt_cloud_client = None
if GOOGLE_TRANSLATE_MODE == "cloud" and GOOGLE_TRANSLATE_API_KEY:
    try:
        from google.cloud import translate_v2 as _gt_v2
        _gt_cloud_client = _gt_v2.Client(client_options={"api_key": GOOGLE_TRANSLATE_API_KEY})
    except Exception as e:
        log.warning(f"Could not init Google Cloud Translate client, falling back to 'free' mode: {e}")
        GOOGLE_TRANSLATE_MODE = "free"


# MyMemory (fallback free engine, below) expects locale-region codes for
# some languages rather than the bare ISO code the rest of this app uses
# internally.
_MYMEMORY_LANG_MAP = {"am": "am-ET"}


def _google_translate_single(text: str, target_lang: str) -> str:
    """One raw call to the free/cloud NMT backends — no chunking. Callers
    should go through _google_translate() below instead, which chunks long
    text automatically; call this directly only when you already know
    `text` is short (< ~450 chars)."""
    if GOOGLE_TRANSLATE_MODE == "off" or not text or not text.strip():
        return ""

    if GOOGLE_TRANSLATE_MODE == "cloud":
        if not _gt_cloud_client:
            return ""
        try:
            result = _gt_cloud_client.translate(text, target_language=target_lang, source_language="en")
            out = (result.get("translatedText") or "").strip()
            if out:
                log.info(f"Google Cloud Translate succeeded ({target_lang}, {len(out)} chars).")
            return out
        except Exception as e:
            log.warning(f"Google Cloud Translate failed ({target_lang}): {e}")
            return ""

    # "free" mode — deep-translator, Google engine first
    try:
        from deep_translator import GoogleTranslator
        out = (GoogleTranslator(source="en", target=target_lang).translate(text) or "").strip()
        if out:
            log.info(f"Google Translate (free) succeeded ({target_lang}, {len(out)} chars).")
            return out
    except ImportError:
        log.warning("deep-translator not installed — run 'pip install deep-translator'. Falling back to Groq for translation.")
        return ""
    except Exception as e:
        log.warning(f"Free Google Translate failed ({target_lang}): {e}")

    # Second free engine, tried before giving up on non-Groq translation.
    # Root cause this covers: on the livestock-question incident
    # (rag_log.txt), "Free Google Translate failed" and the Groq-based
    # fallback below BOTH failed in the same turn — Groq was mid-429 from
    # the CSV retry that had just burned its per-minute budget on the same
    # request. A translation path that depends on Groq being healthy at
    # the exact moment Groq is rate-limited is a single point of failure.
    # MyMemory is a second free, keyless NMT backend with an independent
    # rate limit/outage profile from both Google's scrape endpoint and Groq,
    # so it can absorb exactly this "both providers down together" case.
    try:
        from deep_translator import MyMemoryTranslator
        out = (MyMemoryTranslator(source="en-GB", target=_MYMEMORY_LANG_MAP.get(target_lang, target_lang)).translate(text) or "").strip()
        if out:
            log.info(f"MyMemory Translate (free, fallback engine) succeeded ({target_lang}, {len(out)} chars).")
            return out
    except ImportError:
        pass  # deep-translator already confirmed importable above; nothing more to log
    except Exception as e:
        log.warning(f"MyMemory Translate fallback failed ({target_lang}): {e}")

    return ""


# ═══════════════════════════════════════
# SMART CHUNKING + QUALITY GUARDRAILS (Amharic long-answer fix)
# ═══════════════════════════════════════
# Confirmed root cause in rag_log.txt (2026-08-26 16:10): a ~900-char urban
# population answer was handed to _google_translate_single() whole. The free
# Google endpoint returned "No translation was found using the current
# translator" and MyMemory returned "Text length need to be between 0 and
# 500 characters" — both backends were fed the ENTIRE answer in one call,
# and MyMemory hard-caps at 500 chars. Fix: split into <450-char pieces on
# paragraph -> sentence -> (last resort) word boundaries, translate each
# piece, and reassemble. A quality check on the final result stops a
# garbled/partial/echoed-error translation from ever reaching the user —
# on any failure this returns "" so the caller falls through to the
# existing Groq LLM translation path.
_MYMEMORY_SAFE_CHARS = 450  # under MyMemory's hard 500-char cap, with headroom

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?።])\s+")
_AMHARIC_CHAR_RE = re.compile(r"[\u1200-\u137F]")
_TRANSLATION_FAILURE_MARKERS = (
    "no translation was found",
    "text length need to be between",
    "try another translator",
    "invalid source language",
)


def _smart_chunk_text(text: str, max_chars: int = _MYMEMORY_SAFE_CHARS) -> list:
    """Splits `text` into pieces <= max_chars, breaking on paragraph, then
    sentence, then (last resort) word boundaries — never mid-number,
    mid-percentage, or mid-word. Returns [text] unchanged if it already
    fits in one piece."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in text.split("\n"):
        candidate = (current + "\n" + para) if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue

        flush()
        sentences = _SENTENCE_SPLIT_RE.split(para) if para.strip() else [para]
        for sent in sentences:
            candidate = (current + " " + sent) if current else sent
            if len(candidate) <= max_chars:
                current = candidate
                continue
            flush()
            if len(sent) <= max_chars:
                current = sent
                continue
            # Single sentence still too long (rare) — hard-split on words.
            piece = ""
            for w in sent.split(" "):
                cand = (piece + " " + w) if piece else w
                if len(cand) <= max_chars:
                    piece = cand
                else:
                    if piece:
                        chunks.append(piece.strip())
                    piece = w
            current = piece
    flush()
    return chunks if chunks else [text]


def _translation_quality_ok(english_source: str, translated: str) -> bool:
    """Cheap sanity check so a garbled/partial/failed translation never gets
    served as if it were good Amharic. Rejects (False) when the result is
    empty, echoes one of the translator's own error strings, contains no
    Ge'ez script at all, or is wildly disproportionate in length to the
    source (a sign of truncation or a stuck retry)."""
    if not translated or not translated.strip():
        return False
    low = translated.lower()
    if any(marker in low for marker in _TRANSLATION_FAILURE_MARKERS):
        return False
    if not _AMHARIC_CHAR_RE.search(translated):
        return False
    if english_source:
        ratio = len(translated) / max(len(english_source), 1)
        if ratio < 0.25 or ratio > 4.0:
            return False
    return True


# ═══════════════════════════════════════
# GLOSSARY ENFORCEMENT (placeholder substitution)
# ═══════════════════════════════════════
# Previously GLOSSARY_TERMS was only injected as instructions into the Groq
# LLM prompt — an NMT engine (Google/MyMemory) never saw it at all, and an
# LLM prompt is a request, not a guarantee (it can still ignore/reword it).
# Fix: before ANY translation call, swap each glossary term's English text
# for a placeholder token that no translator will "helpfully" retranslate
# or reword (the ⟦...⟧ brackets don't occur in normal ESS text), run the
# translation, then swap the placeholder back for the EXACT approved
# Amharic wording. This makes glossary terms deterministic — forced,
# not requested — across both the NMT and LLM translation paths.
_GLOSSARY_PLACEHOLDER_FMT = "\u27e6G{idx}\u27e7"  # ⟦G0⟧, ⟦G1⟧, ...


def _apply_glossary_placeholders(text: str, target_lang: str) -> tuple:
    """Returns (protected_text, mapping). mapping is {placeholder: approved
    Amharic text}, applied in longest-term-first order so a shorter term
    that's a substring of a longer one (e.g. a future 'Meher' vs 'Meher
    season') never gets swapped first and breaks the longer match."""
    glossary_terms = _get_glossary_terms()
    if target_lang != "am" or not glossary_terms or not text:
        return text, {}

    mapping = {}
    protected = text
    terms = sorted((t for t in glossary_terms.items() if t[1]), key=lambda kv: -len(kv[0]))
    for idx, (en_term, am_value) in enumerate(terms):
        placeholder = _GLOSSARY_PLACEHOLDER_FMT.format(idx=idx)
        pattern = re.compile(r"\b" + re.escape(en_term) + r"\b", re.IGNORECASE)
        new_protected, n = pattern.subn(placeholder, protected)
        if n:
            protected = new_protected
            mapping[placeholder] = am_value
    return protected, mapping


def _restore_glossary_placeholders(text: str, mapping: dict) -> str:
    if not mapping or not text:
        return text
    for placeholder, am_value in mapping.items():
        text = text.replace(placeholder, am_value)
    return text


def _google_translate(text: str, target_lang: str) -> str:
    """Public entry point used everywhere else in this file. Forces glossary
    terms via placeholder substitution, chunks long text automatically (see
    _smart_chunk_text — this is what the 500-char MyMemory failures were
    missing), translates each chunk, reassembles, restores glossary terms,
    and quality-checks the final result before returning it. Returns '' on
    any failure so the caller falls through to the Groq LLM translation
    path — never blocks the user on a translation-provider error."""
    if GOOGLE_TRANSLATE_MODE == "off" or not text or not text.strip():
        return ""

    protected_text, glossary_map = _apply_glossary_placeholders(text, target_lang)
    chunks = _smart_chunk_text(protected_text)

    if len(chunks) == 1:
        out = _google_translate_single(protected_text, target_lang)
        out = _restore_glossary_placeholders(out, glossary_map)
        return out if _translation_quality_ok(text, out) else ""

    translated_chunks = []
    for chunk in chunks:
        t = _google_translate_single(chunk, target_lang)
        if not t or not _AMHARIC_CHAR_RE.search(t):
            # One failed/English-only chunk would poison the whole
            # reassembled answer (half-Amharic, half-English reads as
            # broken, not partially working) — abandon the chunked path
            # entirely and let the caller fall through to Groq for the
            # FULL text instead.
            log.warning(f"Chunked Google Translate: chunk {len(translated_chunks) + 1}/{len(chunks)} "
                        f"failed quality check — abandoning chunked path for this answer.")
            return ""
        translated_chunks.append(t)

    result = " ".join(translated_chunks).strip()
    result = _restore_glossary_placeholders(result, glossary_map)
    if _translation_quality_ok(text, result):
        log.info(f"Chunked Google Translate succeeded ({target_lang}, {len(chunks)} chunks, {len(result)} chars).")
        return result

    log.warning("Chunked Google Translate: reassembled result failed the quality check — falling through to Groq.")
    return ""


# ─── Table-aware translation ───────────────────────────────────────
# Root cause of "table form" working in English but not Amharic (confirmed
# in a screenshot: same question, English gives a clean table, Amharic
# gives a wall of prose with no table at all, and takes 80s+): both
# translation backends (_google_translate's whole-blob NMT call, and the
# Groq LLM fallback below) were given the ENTIRE answer — including
# "| Region | Area (ha) | ... |" rows — as one plain-text string to
# translate. Machine translation reflows text; it doesn't preserve a
# pipe-delimited grid, so the table structure was destroyed before it ever
# reached the frontend's Markdown-table renderer. Fix: split the answer
# into table / non-table segments BEFORE translating, translate each
# separately, and for table segments translate cell-by-cell so the exact
# same "|...|...|" shape survives — numeric/percentage cells pass through
# completely untouched (translating "3,396,871" or "87.0%" is not just
# unnecessary, it risks the translator "helpfully" reformatting the number).
def _split_markdown_table_segments(text: str) -> list:
    """Splits text into ('prose', str) / ('table', str) segments in
    original order. A 'table' segment is a contiguous run of lines that
    each start and end with '|' (covers the header row, the |---|---|
    separator row, and every data row)."""
    lines = text.split("\n")
    segments = []
    buf = []
    in_table = False

    def flush(kind):
        if buf:
            segments.append((kind, "\n".join(buf)))
            buf.clear()

    for line in lines:
        looks_like_row = line.strip().startswith("|") and line.strip().endswith("|") and len(line.strip()) > 1
        if looks_like_row != in_table:
            flush("table" if in_table else "prose")
            in_table = looks_like_row
        buf.append(line)
    flush("table" if in_table else "prose")
    return segments


_NUMERIC_CELL_RE = re.compile(r"^[\d.,%\-–—+\s]*$")


def _translate_table_segment(table_text: str, target_lang: str, translate_cell) -> str:
    """translate_cell: a function(str) -> str used for each non-numeric
    cell's text (region names, header words, etc.). Numeric-only cells
    (areas, percentages, counts) are passed through byte-for-byte."""
    out_lines = []
    for line in table_text.split("\n"):
        stripped = line.strip()
        # Separator row ("|---|---|" or "|:--|--:|") — never translate, it
        # has no human-language content and any edit breaks the table.
        if re.match(r"^\|?\s*:?-{2,}", stripped.strip("|")):
            out_lines.append(line)
            continue
        if not (stripped.startswith("|") and stripped.endswith("|")):
            out_lines.append(line)
            continue
        cells = stripped[1:-1].split("|")
        translated_cells = []
        for cell in cells:
            c = cell.strip()
            if not c or _NUMERIC_CELL_RE.match(c):
                translated_cells.append(f" {c} ")
            else:
                t = (translate_cell(c) or c).strip()
                translated_cells.append(f" {t} ")
        out_lines.append("|" + "|".join(translated_cells) + "|")
    return "\n".join(out_lines)


def _build_glossary_block(target_lang: str) -> str:
    """Turns the glossary into a forced-wording block for the translation
    prompt. Only used for 'am' — terms without an Amharic translation are
    skipped."""
    if target_lang != "am":
        return ""
    rows = [f'  "{en_term}" -> "{val}"' for en_term, val in _get_glossary_terms().items() if val]
    if not rows:
        return ""
    return "Mandatory glossary — use this EXACT wording whenever these terms appear, do not translate them any other way:\n" + "\n".join(rows) + "\n\n"

# ═══════════════════════════════════════
# FINAL-ANSWER TRANSLATION (Amharic quality fix)
# ═══════════════════════════════════════
# Root cause of weaker Amharic answers: asking the model to REASON and
# WRITE directly in a low-resource language in one shot is noticeably less
# reliable than asking it to reason in English (where it's strongest) and
# then translate the finished answer. This is a dedicated, narrow
# translation call — no reasoning required, so a small/fast model handles
# it well, and numbers/sources are far less likely to drift.
_TRANSLATE_FEW_SHOT = {
    "am": (
        "Example:\n"
        "English: \"The projected population of Ethiopia for 2025 is approximately 110 million, "
        "according to the Ethiopian Statistical Service.\"\n"
        "Amharic: \"በኢትዮጵያ ስታቲስቲክስ አገልግሎት መሠረት፣ የኢትዮጵያ የ2025 ዓ.ም ግምታዊ የሕዝብ ብዛት 110 ሚሊዮን ገደማ ነው።\"\n\n"
    ),
}

# ═══════════════════════════════════════
# ACTUAL-LANGUAGE TAGGING (fallback-chain visibility fix)
# ═══════════════════════════════════════
# Gap found while tracing the full translation fallback chain: every result
# dict below sets "language": target_lang — the REQUESTED language, not
# what was actually served. If Amharic translation exhausts its entire
# fallback chain (Google free -> MyMemory -> Groq -> Gemini -> Mistral ->
# OpenRouter -> Cerebras -> Ollama — see translate_answer()/call_groq()) and
# every tier fails, translate_answer() already degrades gracefully by
# returning the English answer unchanged. But the response still claimed
# "language": "am" — so chat.js/widget.js's Listen button (see
# bubble.dataset.answerLang in widget.js) would fetch the 'am' voice from
# /tts and read plain English text with Amharic pronunciation. Cheap fix:
# tag the response with what the text actually IS, not what was asked for.
def _actual_answer_lang(answer_text: str, requested_lang: str) -> str:
    """For 'am': if the served text has no Ge'ez script at all, the whole
    translation fallback chain was exhausted and English was served instead
    — report 'en' so the frontend (esp. TTS) reacts to reality, not the
    request. Unchanged for every other case."""
    if requested_lang == "am" and answer_text and not _AMHARIC_CHAR_RE.search(answer_text):
        return "en"
    return requested_lang


def translate_answer(english_answer: str, target_lang: str) -> str:
    """Translate a finished English answer into Amharic.
    Returns the English answer unchanged for 'en' or on translation failure
    (never block the user on a translation error)."""
    if target_lang != "am" or not english_answer or not english_answer.strip():
        return english_answer

    segments = _split_markdown_table_segments(english_answer)
    has_table = any(kind == "table" for kind, _ in segments)

    if has_table:
        # Translate each segment on its own terms instead of handing the
        # whole answer (table markup included) to a translator in one
        # shot — see the comment on _split_markdown_table_segments for why.
        def _translate_prose(seg: str) -> str:
            if not seg.strip():
                return seg
            t = _google_translate(seg, target_lang)
            if t:
                return t
            # Groq fallback for this prose chunk specifically.
            prompt = (
                f"Translate the following English text into Amharic (አማርኛ). "
                f"Translate ONLY — preserve every number and percentage exactly, "
                f"output ONLY the translation:\n\n\"{seg}\""
            )
            t = call_groq(prompt, retries=1, max_tokens=300, model=MODEL)
            return seg if is_bad_answer(t) else t.strip()

        def _translate_cell(cell_text: str) -> str:
            t = _google_translate(cell_text, target_lang)
            return t if t else cell_text

        out_parts = []
        for kind, seg in segments:
            if kind == "table":
                out_parts.append(_translate_table_segment(seg, target_lang, _translate_cell))
            else:
                out_parts.append(_translate_prose(seg))
        result = "\n".join(out_parts).strip()
        if result:
            log.info(f"Table-aware Amharic translation succeeded ({len(segments)} segments, "
                      f"{sum(1 for k, _ in segments if k == 'table')} table).")
            return result
        log.warning("Table-aware Amharic translation produced nothing usable — falling back to English answer.")
        return english_answer

    # No table in the answer — original whole-text path, unchanged.
    # 1. Try Google Translate first — dedicated NMT beats a general LLM on
    #    low-resource languages like Amharic. Falls through to Groq on failure.
    gt = _google_translate(english_answer, target_lang)
    if gt:
        return gt

    # Groq fallback. Glossary terms are forced via the SAME placeholder
    # substitution used in _google_translate (deterministic — the LLM can't
    # reword a placeholder token), with the prompt's glossary block kept as
    # a second, belt-and-suspenders instruction for any term the pattern
    # match happened to miss (e.g. slightly different phrasing).
    protected_answer, glossary_map = _apply_glossary_placeholders(english_answer, target_lang)

    lang_name = "Amharic (አማርኛ)"
    few_shot = _TRANSLATE_FEW_SHOT.get(target_lang, "")
    glossary = _build_glossary_block(target_lang)

    prompt = f"""Translate the following English answer into {lang_name}.

{glossary}{few_shot}Rules:
- Translate ONLY — do not add, remove, or reinterpret any information.
- Preserve every number, statistic, date, and percentage EXACTLY as written.
- Keep proper nouns, region names, and document/report titles in their recognizable form.
- Tokens that look like \u27e6G0\u27e7, \u27e6G1\u27e7 etc. are placeholders — copy them into your output EXACTLY as written, unchanged, do not translate or remove them.
- Output ONLY the translation, nothing else — no notes, no English text alongside it.

English answer:
\"{protected_answer}\"

{lang_name} translation:"""

    translated = call_groq(prompt, retries=2, max_tokens=400, model=MODEL)
    translated = _restore_glossary_placeholders(translated, glossary_map) if translated else translated
    if is_bad_answer(translated) or not _translation_quality_ok(english_answer, translated):
        log.warning(f"Translation to '{target_lang}' failed quality check, falling back to English answer.")
        return english_answer
    return translated.strip()

# ═══════════════════════════════════════
# QUERY REWRITE & ROUTING
# ═══════════════════════════════════════
def _format_history(chat_history: list) -> str:
    if not chat_history:
        return ""
    turns = chat_history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in turns:
        u = (turn.get("question") or turn.get("user") or "").strip()
        a = (turn.get("answer") or turn.get("assistant") or "").strip()
        if u:
            lines.append(f"User: {u}")
        if a:
            lines.append(f"Assistant: {a[:200]}")
    return "\n".join(lines)

def rewrite_query(question: str, chat_history: list = None) -> str:
    if not ENABLE_QUERY_REWRITE:
        return question

    history_block = _format_history(chat_history)
    has_amharic = _contains_amharic_script(question)
    target_lang = resolve_language(question)
    rewrite_model = MODEL if target_lang == "am" else FAST_MODEL

    prompt = f"""You clean up user questions for the Ethiopian Statistical Service (ESS) AI Assistant
before they are routed to a database or document search. The document/CSV corpus is English-only.

Do ALL of the following that apply:
1. Fix typos, misspellings, or phonetically written Amharic-in-Latin-script text.
2. CRITICAL: If written in native Amharic script (Fidel), TRANSLATE into a clear,
   concise English query. Do not leave any non-English script in the output.
3. If a chat history is given and the new question is a vague follow-up, rewrite it into a
   standalone question using the history. Otherwise do not invent new topics.

CRITICAL TRANSLATION RULES:
- Translate LITERALLY. Preserve every named topic, region, season, year, and report type exactly.
- NEVER substitute a different topic. "Meher" is an Ethiopian farming season — keep "Meher season".
- NEVER drop key nouns: trade, export, investment, livestock, manufacturing, housing, labour,
  population, inflation, agriculture, survey, report.
- Glossary: Meher=main rainy-season crops; Belg=short rains; AGSS=Agricultural Sample Survey;
  EDHS=Ethiopia Demographic and Health Survey; ESS=Ethiopian Statistical Service.
- If unsure about a word, keep the original word rather than guessing a different topic.

Never answer the question. Never add information that isn't implied by the input.

{"Recent conversation:" if history_block else ""}
{history_block}

Question: "{question}"

Output ONLY the corrected/translated question text, in English. No quotes, no explanation."""

    cleaned = call_groq(prompt, retries=2, max_tokens=300, model=rewrite_model)

    if is_bad_answer(cleaned):
        return question
    cleaned = cleaned.strip().strip('"').strip()
    if not cleaned:
        return question
    max_ratio = 8 if has_amharic else 4
    if len(cleaned) > max(40, len(question) * max_ratio):
        log.warning(f"Rewrite discarded (too long vs input): '{question[:60]}' -> '{cleaned[:60]}'")
        return question
    if has_amharic and _contains_amharic_script(cleaned):
        log.warning(f"Rewrite still contains Amharic script, treating as failed translation: '{question[:60]}'")
        return question

    if cleaned.lower() != question.lower():
        log.info(f"Query rewritten: '{question[:60]}' -> '{cleaned[:60]}'")
    return cleaned

def route_question_keywords(question: str) -> str:
    q = question.lower().strip()

    unambiguous_meta = [
        "what kind of data", "what kind of question", "what kind of questions",
        "what questions can", "what can i ask", "what can you answer",
        "what can you help", "what can this", "what topics can", "which topics",
        "who are you", "what are you"
    ]
    if any(m in q for m in unambiguous_meta):
        return "meta"

    strong_pdf = [
        "explain", "describe", "definition", "what does", "methodology",
        "chapter", "according to", "findings", "overview", "purpose",
        "background", "tell me about", "why", "how does",
        "report", "survey", "statistics", "characteristics", "analysis",
        "projected", "projection", "trade", "export", "manufacturing",
        "livestock", "housing", "labour", "labor", "migration",
        "key findings", "summary of", "what does the",
        # Price/inflation must always route to "pdf": every CSV in this
        # corpus is household consumption/welfare survey data — none carry
        # CPI/price-index columns. Routing these to "csv" made the LLM
        # invent pandas code against unrelated columns and silently return
        # a fabricated number as "the inflation rate" (seen in rag_log.txt,
        # Aug 14 15:04-15:05: routed to sect4_hh_w5.csv / s4q00 / pw_w5).
        "inflation", "cpi", "consumer price", "price index", "cost of living",
        "land utilization", "land utilisation", "land use", "land-use",
        "land area", "hectares",
        "ማብራሪያ", "ትርጉም", "ጋባሳ", "ሪፖርት",
    ]
    strong_csv = [
        "how many", "how much", "total number", "what is the total",
        "average", "percentage of", "count of", "sum of", "maximum",
        "minimum", "what percentage",
        "how many households", "household count", "number of households",
    ]

    has_pdf_trigger = any(p in q for p in strong_pdf)
    has_csv_trigger = any(c in q for c in strong_csv)

    if has_pdf_trigger and has_csv_trigger:
        return "hybrid"
    if has_csv_trigger:
        return "csv"
    if has_pdf_trigger:
        return "pdf"
    # No strong keyword match either way — this only runs when the LLM
    # router call failed/was skipped (rate limit, network hiccup), so we
    # can't ask the model for its best guess. Rather than gamble on a
    # single PDF-only search (which can miss on bare factual questions
    # like "population of Ethiopia" and return "Nothing relevant was
    # found"), try both CSV and PDF and let whichever succeeds answer.
    return "hybrid"

_VALID_ROUTES = {"meta", "csv", "pdf", "hybrid"}

# Shared with _run_pdf_search()'s prefer_source logic below — kept as one
# list so the routing override and the retrieval boost can never drift
# apart (previously _LAND_KEYWORDS was local to rewrite_and_route() only).
_LAND_KEYWORDS_MODULE = (
    "land utilization", "land utilisation", "land use", "land-use",
    "land area", "hectares", "arable land", "agricultural land",
)

# When a question's keywords match a key here, _run_pdf_search() searches
# this exact indexed filename FIRST (via retriever.search_documents's
# prefer_source) instead of relying purely on embedding similarity to rank
# it above a wrong-but-similarly-worded document. Added after confirming
# (rag_log.txt + screenshots, Aug 22) that "land utilization in other
# regions" pulled national-area-production.pdf (crop production %) ahead
# of national-land-use.pdf (land area in hectares — the actually-correct
# source) — guess_category() in index_pdfs.py buckets the production file
# under the too-generic "general" category, so a category filter can't
# separate them; pinning by exact filename can. Extend this dict for other
# topic/file collisions as they turn up in the logs — don't guess ahead of
# evidence.
_PREFERRED_SOURCE_MAP = {
    _LAND_KEYWORDS_MODULE: "national-land-use.pdf",
}

# Same class of bug as land-utilization above, confirmed in rag_log.txt for
# "በኢትዮጵያ የእንስሳት ሃብት ምርት መጠን ምን ያህል ነው?" ("what is the livestock production
# quantity in Ethiopia?"): this general/aggregate question got routed to
# "csv" (both in Amharic AND in an earlier English run of the same intent),
# which sends it to query_csv_with_groq(). The model then had to guess a
# column in sect11_hh_w5.csv (a household-survey file with no
# national-aggregate livestock figure at all) and picked a nonexistent one
# ('cs11q06') -> KeyError -> retry -> Groq 429 -> Mistral fallback -> still
# flagged as a bad/sentinel answer. The correct source for a general
# livestock question is the livestock survey PDF report, which the keyword
# router (route_question_keywords, above) already sends "livestock"
# questions to via strong_pdf — but that keyword list is only consulted
# when the LLM router call fails; the normal path is the combined
# rewrite+route LLM call below, which has no equivalent hard override for
# livestock the way price/inflation and land-utilization already do.
#
# Only force "pdf" for GENERAL livestock questions. A genuine
# household-level computation (e.g. "how many households in Oromia own
# cattle") legitimately needs the CSV, so don't override those — detect
# them via _HOUSEHOLD_LEVEL_KEYWORDS and skip the override when present.
_LIVESTOCK_KEYWORDS_MODULE = (
    "livestock", "cattle", "oxen", "cows", "cow ", "bulls", "goats", "sheep",
    "poultry", "chickens", "chicken ", "camels", "beehives", "animal husbandry",
    "livestock production", "livestock population", "livestock resource",
)

_HOUSEHOLD_LEVEL_KEYWORDS = (
    "household", "households", "per household", "families own", "family owns",
    "how many households", "number of households", "households that own",
    "households who own", "own livestock", "owning livestock",
)


def _preferred_source_for(question: str) -> str:
    q = question.lower()
    for keywords, source in _PREFERRED_SOURCE_MAP.items():
        if any(k in q for k in keywords):
            return source
    return None


# ═══════════════════════════════════════
# REWRITE+ROUTE DECISION CACHE
# ═══════════════════════════════════════
_REWRITE_ROUTE_CACHE = {}  # {normalized_q: ((cleaned_q, route), timestamp)}
_REWRITE_CACHE_TTL_SECONDS = 3600  # 1 hour

def _get_cached_rewrite_route(question: str):
    """Return (cleaned_q, route) if we've seen this question before (within 1 hour)."""
    normalized = _normalize_question(question)
    if normalized in _REWRITE_ROUTE_CACHE:
        entry, timestamp = _REWRITE_ROUTE_CACHE[normalized]
        if time.time() - timestamp < _REWRITE_CACHE_TTL_SECONDS:
            return entry, True  # (cleaned_q, route), is_cached=True
    return None, False

def _cache_rewrite_route(question: str, cleaned_q: str, route: str):
    """Store rewrite decision for future similar questions."""
    normalized = _normalize_question(question)
    _REWRITE_ROUTE_CACHE[normalized] = ((cleaned_q, route), time.time())
    # Limit cache size to prevent memory bloat
    if len(_REWRITE_ROUTE_CACHE) > 10000:
        oldest_key = min(
            _REWRITE_ROUTE_CACHE.keys(),
            key=lambda k: _REWRITE_ROUTE_CACHE[k][1]
        )
        del _REWRITE_ROUTE_CACHE[oldest_key]

def route_question_llm(question: str) -> str:
    prompt = f"""You are a routing classifier for the Ethiopian Statistical Service (ESS) AI Assistant.
Categorize the user's question into exactly one route. Base the decision on INTENT, not on
surface keywords.

Routes:
- meta: greetings, identity questions, or questions about what data/topics the assistant can answer.
- csv: the user wants ONLY an exact number, statistic, count, average, or raw tabular value.
- pdf: the user wants ONLY a text explanation, definition, methodology, or background summary from a report.
- hybrid: the user's question combines a numeric/tabular need AND a request for explanation.

Question: "{question}"

Respond with exactly one lowercase word: meta, csv, pdf, or hybrid. No punctuation, no explanation."""

    raw = call_groq(prompt, retries=2, max_tokens=20, model=FAST_MODEL)
    cleaned = re.sub(r"[^a-z]", "", raw.lower().strip()) if raw else ""
    route = cleaned
    if route not in _VALID_ROUTES:
        for candidate in _VALID_ROUTES:
            if candidate in raw.lower():
                route = candidate
                break
    if route in _VALID_ROUTES:
        return route

    log.warning(f"LLM router returned invalid output '{raw}', falling back to keyword router.")
    return route_question_keywords(question)

# Patterns unambiguous enough to skip the LLM router call entirely — these
# essentially never need a PDF explanation alongside them, so paying for a
# ~1-2s Groq round trip just to confirm "yes, this is a csv question" is
# pure latency with no accuracy upside.
_STRONG_CSV_ONLY_PATTERNS = (
    "how many", "how much", "total number of", "what is the total",
    "average of", "sum of", "count of", "maximum of", "minimum of",
)
# If any of these appear, the question likely wants explanation too (or
# instead) — don't fast-path, let the LLM router make the judgment call.
_EXPLANATION_SIGNALS = (
    "explain", "describe", "why", "how does", "methodology", "report", "meaning",
    "survey", "findings", "statistics", "characteristics", "analysis", "projected",
    "trade", "manufacturing", "livestock", "housing", "labour", "labor", "summary",
)

def _fast_path_route(question: str):
    """Conservative keyword short-circuit: only fires for short, clearly
    numeric-only questions. Anything longer or with any explanation signal
    falls through to the normal LLM/keyword routing — this only trims
    latency off the unambiguous common case, it never guesses on hard ones."""
    q = question.lower().strip()
    if len(q.split()) > 12:
        return None
    if any(p in q for p in _EXPLANATION_SIGNALS):
        return None
    if any(p in q for p in _STRONG_CSV_ONLY_PATTERNS):
        return "csv"
    return None

def route_question(question: str) -> str:
    if ROUTER_MODE == "keyword":
        return route_question_keywords(question)
    fast = _fast_path_route(question)
    if fast:
        log.info(f"Fast-path keyword route (skipped LLM router call): '{question[:50]}' -> {fast}")
        return fast
    try:
        return route_question_llm(question)
    except Exception as e:
        log.error(f"LLM router failed entirely, falling back to keyword router: {e}")
        return route_question_keywords(question)

def rewrite_and_route(question: str, chat_history: list = None) -> tuple:
    if not ENABLE_QUERY_REWRITE:
        return question, route_question(question)
    
    # Check cache first — skip expensive rewrite+route LLM call if we've seen this before
    if not chat_history:  # Only use cache for standalone questions, not conversation context
        cached_result, is_cached = _get_cached_rewrite_route(question)
        if is_cached:
            log.info(f"Cached rewrite+route hit: '{question[:50]}'")
            return cached_result
    
    if ROUTER_MODE == "keyword":
        cleaned = rewrite_query(question, chat_history)
        return cleaned, route_question_keywords(cleaned)

    history_block = _format_history(chat_history)
    has_amharic = _contains_amharic_script(question)
    # Prefer larger model for AM/OM — translation quality matters more than the ~1s extra
    target_lang = resolve_language(question)
    rewrite_model = MODEL if target_lang == "am" else FAST_MODEL

    prompt = f"""You prepare user questions for the Ethiopian Statistical Service (ESS) AI Assistant in ONE step.
Do BOTH tasks and return a single JSON object, nothing else.

TASK 1 - Clean / translate the question into clear English:
- Fix typos or phonetically written Amharic/Oromo-in-Latin-script.
- If written in native Amharic script (Fidel), TRANSLATE it fully into clear English. No Fidel characters may remain.
- If written in Afaan Oromoo, TRANSLATE it fully into clear English.
- If it's a vague follow-up ("what do you mean", "and in 2020?") and conversation history is given,
  rewrite it into a standalone question using that history.

CRITICAL TRANSLATION RULES (do not violate):
- Translate LITERALLY. Preserve every named topic, region, season, year, and report type exactly.
- NEVER substitute a different topic. Example: "Meher" is an Ethiopian farming season — keep it as "Meher season", do NOT turn it into "cotton industry" or anything else.
- NEVER drop key nouns such as trade, export, investment, livestock, manufacturing, housing, labour, population, inflation, agriculture, survey, report.
- Domain glossary (keep these terms):
  Meher = main rainy-season crop period (not cotton)
  Belg = short rainy season
  AGSS = Agricultural Sample Survey
  EDHS = Ethiopia Demographic and Health Survey
  ESS = Ethiopian Statistical Service
  Oromia / Amhara / Tigray / SNNPR / Addis Ababa = region names (keep as-is)
- Afaan Oromoo vocabulary (translate these correctly, do not substitute a
  different topic — e.g. "ummata" is population, NEVER translate a
  population question into a households/survey-count question):
  ummata / uummata = population / people
  baay'ina = number / quantity / amount ("baay'ina ... meeqa?" = "what is
    the number of ...?" / "how many ...?")
  meeqa = how many / how much
  Itoophiyaa = Ethiopia
  qonna = agriculture
  horii = livestock
  gabaa = market
  Oromiyaa = Oromia
  Example: "Baay'ina ummata Itoophiyaa meeqa?" = "What is the population of
  Ethiopia?" — NOT a question about households, surveys, or counts of
  anything else.
- If unsure about a word, keep the original word rather than guessing a different topic.

TASK 2 - Classify the CLEANED English question's intent into exactly one route:
- "meta": greetings, identity questions, or "what can you answer" type questions.
- "csv": wants ONLY an exact number/statistic/count/average from structured tabular data (household counts, consumption values, etc.).
- "pdf": wants explanation, findings, methodology, survey/report summary, trends, or background (most report-style questions).
- "hybrid": explicitly needs both a number AND explanation.

Prefer "pdf" when the question mentions: report, survey, findings, statistics, characteristics, analysis, methodology, projected, trade, manufacturing, livestock, housing, labour/labor, inflation report, land utilization/land use/hectares.

{"Recent conversation:" if history_block else ""}
{history_block}

Original question: "{question}"

Respond with ONLY this JSON, no markdown fences, no commentary:
{{"question": "<cleaned English question>", "route": "<meta|csv|pdf|hybrid>"}}"""

    raw = call_groq(prompt, retries=2, max_tokens=350, model=rewrite_model)

    if is_bad_answer(raw):
        return question, route_question_keywords(question)

    cleaned_question, route = question, None
    try:
        match = re.search(r"\{.*\}", raw, re.S)
        payload = json.loads(match.group(0) if match else raw)
        candidate_q = (payload.get("question") or "").strip().strip('"').strip()
        candidate_r = (payload.get("route") or "").strip().lower()

        if candidate_q:
            max_ratio = 8 if has_amharic else 4
            if len(candidate_q) <= max(40, len(question) * max_ratio) and not (
                has_amharic and _contains_amharic_script(candidate_q)
            ):
                cleaned_question = candidate_q

        if candidate_r in _VALID_ROUTES:
            route = candidate_r
    except Exception as e:
        log.warning(f"Combined rewrite+route parse failed: {e} | raw: {raw[:120]}")

    if route is None:
        route = route_question_keywords(cleaned_question)

    # Hard override, runs regardless of whether the LLM or the keyword
    # fallback picked the route: no CSV in this corpus has CPI/price data,
    # so "csv"/"hybrid" for a price/inflation question is always wrong and
    # produces a fabricated number instead of an honest "not found". This
    # is what actually mis-routed on Aug 14 (LLM router returned "csv"
    # directly — the keyword list above never even ran).
    if route in ("csv", "hybrid") and any(k in cleaned_question.lower() for k in _PRICE_KEYWORDS):
        log.info(f"Forcing route 'pdf' (was '{route}') for price/inflation question: '{cleaned_question[:50]}'")
        route = "pdf"

    # Same problem, different topic: "land utilization" reads to the LLM
    # router as "wants a number" about as often as it reads as "wants a
    # report summary" — rag_log.txt shows the SAME question ("What is the
    # land utilization in Amhara region?") routed to "pdf" once and to
    # "csv"/"hybrid" ~1h40m later (past the 1h rewrite/route cache TTL, so
    # the LLM router ran fresh and flipped). Both the "csv" AND "hybrid"
    # branches call query_csv_with_groq(), which generates pandas code
    # against sect11_com_w5.csv's cs11q04a column — a community-survey
    # field, not the actual national-land-use.pdf report — which is the
    # wrong source even when it "succeeds", or returns nothing and still
    # gets blended into the answer via _gather_sources()'s hybrid path.
    #
    # This override originally only checked `route == "csv"`, unlike the
    # price/inflation override above it which already covers ("csv",
    # "hybrid"). That asymmetry is why land questions kept failing even
    # after prefer_source pinning + PDF_RETRIEVAL_TOP_K were added: retriever_log.txt
    # never shows a single "(preferring national-land-use.pdf)" line for
    # any Amhara land query, confirming route never actually reached "pdf"
    # (_preferred_source_for is only consulted inside _run_pdf_search,
    # which "hybrid" also calls — but blending in a wrong CSV number
    # alongside the correct PDF chunk is exactly what produced the
    # "not found" / vague answers). Force these to pure "pdf", mirroring
    # the price/inflation override exactly.
    _LAND_KEYWORDS = _LAND_KEYWORDS_MODULE
    if route in ("csv", "hybrid") and any(k in cleaned_question.lower() for k in _LAND_KEYWORDS):
        log.info(f"Forcing route 'pdf' (was '{route}') for land-utilization question: '{cleaned_question[:50]}'")
        route = "pdf"

    # Livestock override — see _LIVESTOCK_KEYWORDS_MODULE comment above for
    # the incident this fixes. Skip the override (let csv/hybrid stand) when
    # the question is actually asking for a household-level computation.
    if (
        route in ("csv", "hybrid")
        and any(k in cleaned_question.lower() for k in _LIVESTOCK_KEYWORDS_MODULE)
        and not any(k in cleaned_question.lower() for k in _HOUSEHOLD_LEVEL_KEYWORDS)
    ):
        log.info(f"Forcing route 'pdf' (was '{route}') for general livestock question: '{cleaned_question[:50]}'")
        route = "pdf"

    if cleaned_question.lower() != question.lower():
        log.info(f"Query rewritten: '{question[:60]}' -> '{cleaned_question[:60]}'")

    # Cache this rewrite+route decision for future similar questions
    _cache_rewrite_route(question, cleaned_question, route)
    
    return cleaned_question, route

# ═══════════════════════════════════════
# DATA DICTIONARY & SCHEMAS
# ═══════════════════════════════════════
_BASE_DATA_DICTIONARY = {
    "cons_agg_w5.csv": {
        "household_id":   "Unique household identifier",
        "total_cons_ann": "Total annual consumption value",
        "region":         "Region name (e.g., Amhara, Oromia, Tigray)",
        "urban":          "1=Urban, 0=Rural"
    }
}

def load_csv_files() -> dict:
    dataframes = {}
    if not os.path.exists(CSV_FOLDER):
        return dataframes
    for f in sorted(os.listdir(CSV_FOLDER)):
        if f.lower().endswith(".csv"):
            try:
                path = os.path.join(CSV_FOLDER, f)
                dataframes[f] = pd.read_csv(path, low_memory=False)
            except Exception as e:
                log.error(f"CSV load error {f}: {e}")
    return dataframes

def _infer_column_notes(df: pd.DataFrame) -> str:
    lines = []
    cols = list(df.columns)[:MAX_SCHEMA_COLUMNS_SHOWN]
    for col in cols:
        dtype = str(df[col].dtype)
        try:
            sample_vals = df[col].dropna().unique()[:3]
            sample_str = ", ".join(str(v) for v in sample_vals)
        except Exception:
            sample_str = ""
        lines.append(f"  {col} ({dtype}): e.g. {sample_str}")
    if len(df.columns) > MAX_SCHEMA_COLUMNS_SHOWN:
        lines.append(f"  ... and {len(df.columns) - MAX_SCHEMA_COLUMNS_SHOWN} more columns")
    return "\n".join(lines)

def _build_single_file_schema(name: str, df: pd.DataFrame) -> str:
    schema  = f"\nFile: '{name}'\nRows: {len(df)}\n"
    schema += f"Columns: {list(df.columns)}\n"
    if name in _BASE_DATA_DICTIONARY:
        schema += "Column meanings:\n"
        for col, m in _BASE_DATA_DICTIONARY[name].items():
            schema += f"  {col}: {m}\n"
    else:
        schema += "Column meanings (inferred from data, no manual dictionary entry):\n"
        schema += _infer_column_notes(df) + "\n"
    return schema

def build_per_file_schema(dataframes: dict) -> dict:
    return {name: _build_single_file_schema(name, df) for name, df in dataframes.items()}

_WORD_RE = re.compile(r"[a-z0-9]+")

def _select_relevant_schemas(question: str, per_file_schema: dict, top_n: int = CSV_TOP_N_FILES) -> dict:
    """Only send the schema of the files most likely to answer this question,
    instead of every CSV file on every call — this is the main token cost of
    query_csv_with_groq() and directly eats into the Groq per-minute budget.
    Scores by how many question words appear in the filename or that file's
    schema text (column names/meanings). Falls back to ALL files if there
    are few files to begin with, or if nothing scores above zero (never
    silently hide a file that might actually be the right one)."""
    if len(per_file_schema) <= top_n:
        return per_file_schema

    q_words = set(_WORD_RE.findall(question.lower()))
    if not q_words:
        return per_file_schema

    scored = []
    for name, schema_text in per_file_schema.items():
        name_words = set(_WORD_RE.findall(name.lower()))
        schema_words = set(_WORD_RE.findall(schema_text.lower()))
        score = len(q_words & name_words) * 3 + len(q_words & schema_words)
        scored.append((score, name))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored[0][0] == 0:
        # No keyword signal at all — safer to send everything than guess wrong.
        return per_file_schema

    top_names = {name for score, name in scored[:top_n] if score > 0}
    return {name: per_file_schema[name] for name in per_file_schema if name in top_names}

# ═══════════════════════════════════════

def _fix_dataframe_keys(code: str, dataframes: dict) -> str:
    """
    Safety net: the model sometimes emits dataframes['sect11_hh_w5'] instead of
    dataframes['sect11_hh_w5.csv']. If a quoted key is missing but key+'.csv'
    exists, rewrite it before eval. Only touches keys that appear in the
    dataframes dict — never invents new files.
    """
    def _repl(m):
        key = m.group(1)
        if key in dataframes:
            return m.group(0)
        with_csv = key if key.lower().endswith('.csv') else key + '.csv'
        if with_csv in dataframes:
            quote = m.group(0)[len('dataframes['):len('dataframes[')+1]
            return f"dataframes[{quote}{with_csv}{quote}]"
        return m.group(0)
    return re.sub(r"dataframes\[\s*['\"]([^'\"]+)['\"]\s*\]", _repl, code)


# Matches dataframes['file.csv']['col_name'] / dataframes["file.csv"]["col_name"]
# references in generated pandas code, so every column the model claims
# exists can be checked against the real DataFrame BEFORE eval runs.
_DF_COL_ACCESS_RE = re.compile(
    r"dataframes\[\s*['\"]([^'\"]+)['\"]\s*\]\s*\[\s*['\"]([^'\"]+)['\"]\s*\]"
)


def _validate_pandas_columns(code: str, dataframes: dict) -> str:
    """Catches the 'guessed a column that doesn't exist' failure mode BEFORE
    burning an eval() + exception + retry cycle on it (confirmed root cause
    of the livestock/cs11q06 incident in rag_log.txt: the model guessed a
    nonexistent column, safe_eval_pandas() raised KeyError, and the existing
    except-block retry consumed a second Groq call that then hit the 429
    rate limit and fell through to a flagged bad answer).

    Returns a human-readable error string describing the first bad
    file/column reference found, or None if every dataframes[...][...]
    reference in the code resolves to a real column. Only checks the
    bracket-access form the codegen prompt asks for (df['col']) — doesn't
    attempt to parse dot-attribute access or dynamic column names, so it
    can't produce false positives on code it doesn't fully understand."""
    for file_key, col in _DF_COL_ACCESS_RE.findall(code):
        resolved = None
        if file_key in dataframes:
            resolved = file_key
        elif (file_key + ".csv") in dataframes:
            resolved = file_key + ".csv"

        if resolved is None:
            return f"'{file_key}' is not one of the loaded CSV files."
        if col not in dataframes[resolved].columns:
            return (
                f"Column '{col}' does not exist in '{resolved}'. "
                f"Available columns: {list(dataframes[resolved].columns)}"
            )
    return None


# CSV PYTHON CODE GENERATION
# ═══════════════════════════════════════
def query_csv_with_groq(question: str, dataframes: dict, per_file_schema: dict,
                         _retry: bool = False, _last_error: str = None) -> str:
    if not dataframes:
        return None

    relevant_schema = _select_relevant_schemas(question, per_file_schema)
    schema = "".join(list(relevant_schema.values()))
    if len(relevant_schema) < len(per_file_schema):
        log.info(f"CSV schema trimmed to {len(relevant_schema)}/{len(per_file_schema)} file(s) for: '{question[:50]}'")

    retry_hint = ""
    if _retry:
        if _last_error:
            # Thread the ACTUAL failure back to the model instead of a generic
            # guess — a syntax/KeyError message is far more useful for a
            # self-correction than a canned "maybe it's case mismatch" note.
            retry_hint = (
                f"\nNOTE: Your previous attempt failed with this error:\n{_last_error}\n"
                "Fix the specific problem above. Common causes: (1) a column name that "
                "doesn't exist — re-check the exact column names listed in the schema "
                "above, don't guess or invent one; (2) code too long/got cut off — for "
                "multi-value filters use .isin([...]) instead of chaining many == "
                "conditions with |; (3) unbalanced brackets/quotes from an overly long "
                "expression — keep the whole thing under ~120 words; (4) if the error "
                "says \"Can only use .str accessor with string values\", the column is "
                "already numeric — drop the .str.replace()/.str.strip() call entirely "
                "and use the column as-is; (5) if the error mentions invalid syntax and "
                "your previous code contained the word \"for\", that is not valid here — "
                "rewrite it as a single .loc[mask] filter expression instead."
            )
        else:
            retry_hint = (
                "\nNOTE: A previous attempt at this returned no rows — almost always caused by an exact-case "
                "string filter (e.g. == 'Amhara' failing against 'amhara' in the data). This time, filter with "
                ".str.lower().str.strip() == 'value'.lower() on every text/region/category comparison, no exceptions."
            )

    prompt = f"""You have pandas DataFrames in a dict called 'dataframes'. Write ONE LINE of Python code to answer: "{question}"
{schema}
Rules:
- Use only: dataframes, pd, len, sum, round, str, min, max, abs
- CRITICAL: use the EXACT file keys shown in the schema above, INCLUDING the .csv
  extension. Write dataframes['sect11_hh_w5.csv'] — NEVER dataframes['sect11_hh_w5']
  without .csv. The dict keys are the full filenames.
- CRITICAL: use the EXACT column names shown in the schema above. Never guess or
  invent a column name (e.g. don't write ['region'] if the schema calls it ['saq01']).
- CRITICAL: Always use string normalization like .str.lower() or .str.strip() when filtering regional or text columns (e.g. filter by 'amhara' instead of exact match) to prevent case mismatch errors.
- CRITICAL: Only call .str.xxx() (.str.replace, .str.strip, .str.lower, etc.) on a
  column whose dtype shown above is (object) — i.e. already text. If a column's
  dtype is (int64) or (float64), it is ALREADY numeric: use it directly, do NOT
  call .str.replace(',', '') or any other .str method on it — that raises
  "Can only use .str accessor with string values" and fails immediately.
- CRITICAL: Write ONLY a single valid Python expression using .loc[], .isin(),
  boolean masks with | & ~, and simple method calls (.sum(), .mean(), .round()).
  NEVER write `for`, a list/generator comprehension, or any construct with the
  word "for" in it (e.g. "X for column == value" is not valid Python at all and
  will always fail to parse) — filter with .loc[mask] instead, always.
- CRITICAL — COUNT vs SUM, do not confuse these:
  * "how many X" / "total number of X" / "count of X" / "number of households/
    people/records" → COUNT ROWS. Use len(df) / df.shape[0] / df['id_col'].nunique()
    on an ID/identifier column. NEVER .sum() a value column for a "how many"
    question — summing total_cons_ann, income, or any money/quantity column
    when asked "how many households" produces a nonsense figure (e.g. hundreds
    of millions) that is NOT a count, it's a sum of money mislabeled as a count.
  * "total <amount/spending/consumption/income>" → SUM a value column, e.g.
    df['total_cons_ann'].sum() — this is correct ONLY when the question asks
    for a total quantity/money amount, not a total count of rows.
  * If in doubt which the question wants, prefer counting rows for anything
    containing the words "how many", "number of", or "count".
- For "multiple regions/categories" questions, prefer ONE .isin(['a','b','c']) call
  over chaining many == comparisons with | — it's shorter (won't get cut off) and
  clearer. Example: df.loc[df['saq01'].str.lower().isin(['amhara','oromia','tigray']), 'total_cons_ann'].sum()
- If you do need to combine boolean conditions manually, use pandas' `|` / `&` /
  `~` with each condition in parentheses — e.g. (cond1) | (cond2) — never Python's
  `or`/`and` keywords (they don't work element-wise on a Series/DataFrame).
- Keep the whole expression concise — long chains of repeated .loc[...] calls are
  more likely to get cut off mid-generation. Prefer .isin(), .groupby(), or a
  single boolean mask built with | / & over many separate .loc lookups.
- Worked example — filtering by a year column then aggregating a numeric column
  (adjust file/column names to match the schema above, never copy these verbatim):
  dataframes['some_file.csv'].loc[dataframes['some_file.csv']['year'] == 2023, 'value_col'].sum()
- Return ONLY the code line, no markdown backticks, no comments.{retry_hint}
"""
    # max_tokens=700 (vs the 500 default): isin()-based code is short, but a
    # borderline case that still needs a few chained conditions has headroom
    # to finish instead of getting truncated mid-expression.
    pandas_code = call_groq(prompt, model=FAST_MODEL, max_tokens=700).strip().replace("```python", "").replace("```", "").strip()

    if is_bad_answer(pandas_code):
        log.warning("Groq failed to produce pandas code; skipping CSV lookup.")
        return None

    log.info(f"Generated Pandas Code{' (retry)' if _retry else ''}: {pandas_code}")

    fixed_code = _fix_dataframe_keys(pandas_code, dataframes)
    if fixed_code != pandas_code:
        log.info(f"Auto-fixed dataframe keys: {pandas_code} -> {fixed_code}")
        pandas_code = fixed_code

    # Column-existence check BEFORE eval — a guessed/invented column is a
    # doomed query no matter how many times we retry it against the same
    # bad guess, so don't spend a retry's Groq call (and rate-limit budget)
    # on it. Skip straight to returning None so the caller (_gather_sources)
    # falls through to PDF retrieval immediately, exactly like an "empty
    # CSV result" already does today.
    validation_error = _validate_pandas_columns(pandas_code, dataframes)
    if validation_error:
        log.warning(
            f"Skipping pandas execution — invalid column reference in generated code "
            f"(no retry, falling through to PDF): {validation_error} | code: {pandas_code}"
        )
        return None

    safe_env = {"dataframes": dataframes, "pd": pd, "len": len, "sum": sum, "round": round, "str": str, "min": min, "max": max, "abs": abs, "int": int, "float": float, "sorted": sorted}
    try:
        result = safe_eval_pandas(pandas_code, safe_env)
        empty = (
            result is None
            or (hasattr(result, '__len__') and len(result) == 0)
        )
        if not empty:
            if isinstance(result, (pd.Series, pd.DataFrame)) and not result.empty:
                text = result.head(5).to_string()
            else:
                text = str(result)
            empty = text is None or text.strip().lower() in ("", "none", "nan", "empty", "series([], dtype: object)")
            # A bare 0/0.0 scalar is almost always the wrong column/filter (e.g.
            # summing a sampling-weight column instead of the actual metric),
            # not a genuine "the answer is zero" — so treat it the same as an
            # empty result: retry once, then let the caller fall back to PDF
            # search instead of confidently stating a meaningless zero.
            if not empty and re.fullmatch(r"0(\.0+)?", text.strip()):
                empty = True
            if not empty:
                return text

        # Likely a case-mismatch on a string filter — the column exists, the row just didn't match.
        # One cheap retry with an explicit correction before giving up.
        if not _retry and re.search(r"==\s*['\"]", pandas_code) and ".str.lower()" not in pandas_code:
            return query_csv_with_groq(question, dataframes, per_file_schema, _retry=True)
        return None
    except Exception as e:
        log.warning(f"Pandas runtime failed: {e}")
        if not _retry:
            return query_csv_with_groq(question, dataframes, per_file_schema, _retry=True, _last_error=str(e))
        return None

def list_pdf_titles(folder: str) -> list:
    if not os.path.exists(folder):
        return []
    return [re.sub(r"[_\-]+", " ", os.path.splitext(f)[0]).strip() for f in sorted(os.listdir(folder)) if f.lower().endswith(".pdf")]

PDF_TITLES = list_pdf_titles(PDF_FOLDER)
# NOTE (thread-safety, preventive only): CSV_DATA is loaded once at import time
# and treated as read-only for the rest of the process's life — every request
# thread only reads from it (e.g. via query_csv_with_groq / safe_eval_pandas),
# so no lock is needed today. If this ever changes — e.g. a future feature lets
# CSV files be hot-reloaded or refreshed while the app is serving traffic — this
# module-level dict/DataFrame reassignment must be protected with a
# threading.Lock (or swapped atomically via a new dict reference) to avoid a
# reader seeing a partially-updated dataset. Not implemented now to avoid
# adding complexity for a mutation path that doesn't exist yet.
CSV_DATA = load_csv_files()

# ═══════════════════════════════════════
# EFY-AWARE YEAR FILTERING (inflation reports)
# ═══════════════════════════════════════
# BUG THIS FIXES: index_pdfs.py's guess_year() pulls the first "20xx" digit
# group out of the filename and stores it as plain metadata "year" — e.g.
# "1.inflation-report-oct-efy-2018-final.pdf" gets year="2018". But that
# "2018" IS an Ethiopian Fiscal Year (EFY) label, not a Gregorian year —
# EFY 2018 runs ~Sept 2025-Sept 2026. Retrieval never filtered on that
# field at all, so a question about Gregorian 2018 (-> EFY 2010, per
# _EFY_RULE) could still retrieve and get answered from an EFY2018 chunk
# just because it's semantically about "inflation", producing a confusing
# reply that mixes up which calendar the "2018" in the answer refers to
# (see rag_log.txt, Aug 25 12:40 — "does not provide... for EFY 2010"
# sourced from "...efy-2018-final.pdf").
#
# Only inflation reports are named this way today; extend this set if
# other EFY-labeled report types (e.g. future CPI bulletins) are added.
_EFY_LABELED_CATEGORIES = {"inflation"}


def _gregorian_to_efy_candidates(g_year: int) -> list:
    """Mirrors the 'EFY N ~= Gregorian (N+7)-(N+8)' rule already used in
    _current_efy_context()/_EFY_RULE above. A bare Gregorian year with no
    month given can fall in either of two EFY years depending on whether
    the real month is before or after the ~Sept 11 Ethiopian New Year, so
    return both candidates rather than guessing one."""
    return [g_year - 8, g_year - 7]


def _detect_requested_year(question: str):
    """Best-effort extraction of a year the user is asking about.
    Returns (year:int, is_explicit_efy:bool), or (None, False) if no
    4-digit year appears in the question at all.
    An explicit EFY mention is trusted as-is; a bare "2018" is assumed
    Gregorian, since that's what people naturally type.

    Matches "efy 2018", "EFY2018", AND "Ethiopian Fiscal Year 2018" /
    "fiscal year 2018" — the query-rewrite step upstream (rewrite_and_route)
    spells "EFY" out as "Ethiopian Fiscal Year" in the cleaned question it
    hands back, so checking only for the literal "efy" here missed every
    rewritten question and silently mis-detected real EFY questions as
    bare Gregorian ones (confirmed via rag_log.txt: "EFY 2018?" ->
    rewritten to "...Ethiopian Fiscal Year 2018?" -> wrongly treated as
    Gregorian 2018 -> false "not indexed" answer for data that IS indexed)."""
    q = question.lower()
    efy_match = re.search(
        r"(?:ethiopian\s+fiscal\s+year|ethiopian\s+calendar(?:\s+year)?|ethiopian\s+year|"
        r"fiscal\s+year|e\.?c\.?|efy)\s*(\d{4})",
        q
    )
    if efy_match:
        return int(efy_match.group(1)), True
    # Reversed order: "2013 E.C." / "2018 EC" — how your own report titles
    # write it (e.g. "2020/21 [2013 E.C.]"), year first then the suffix.
    efy_suffix_match = re.search(r"(\d{4})\s*e\.?c\.?\b", q)
    if efy_suffix_match:
        return int(efy_suffix_match.group(1)), True
    year_match = re.search(r"\b(19|20)\d{2}\b", q)
    if year_match:
        return int(year_match.group(0)), False
    return None, False


def _inflation_years_indexed() -> dict:
    """{year:int -> [source filenames]} for every chunk actually indexed
    under metadata category="inflation", read live from ChromaDB.

    Replaces the old approach of regex-scanning filenames on disk for the
    literal substrings "inflation" and "efy" (e.g. only names like
    "...inflation-report-oct-efy-2018-final.pdf" ever matched). A file
    indexed correctly — via metadata.csv, the bulk indexer's category
    guess, or the /admin/add_report upload — but named differently (e.g.
    "CPI_DEC_2022.pdf") was invisible to that scan, so the chatbot
    confidently claimed data wasn't indexed even though it was. Metadata is
    the source of truth; filenames are not.

    Called fresh on every request (not cached at import time) because a
    report can be added through /admin/add_report while the app is
    running — the whole point of that route (see report_indexer.py) is
    that it's answerable immediately, without an app restart.

    The stored "year" value isn't guaranteed to already be an EFY year —
    it's whatever metadata.csv / guess_year() / the admin form set it to,
    which for an EFY-named legacy file happens to BE the EFY year, but for
    a plainly Gregorian-named file (e.g. "CPI_DEC_2022.pdf" -> "2022") is a
    Gregorian year instead. This function just returns what's indexed, as
    indexed; _run_pdf_search below checks a requested year against both
    conventions rather than assuming one or the other.
    """
    years = {}
    for year_str, sources in get_metadata_index("inflation").items():
        if year_str.isdigit():
            years.setdefault(int(year_str), []).extend(sources)
    return years


def _inflation_efy_years_indexed() -> set:
    """Back-compat convenience: just the set of indexed year numbers."""
    return set(_inflation_years_indexed().keys())

# ═══════════════════════════════════════
# MONTH-AWARE INFLATION FILE MATCHING
# ═══════════════════════════════════════
# BUG THIS FIXES: your inflation corpus is ~19 separate MONTHLY reports per
# EFY year, not one report per year. A question like "inflation rate in EFY
# 2018" (no month named) or even "inflation rate in October EFY 2018" was
# left entirely to embedding similarity across ALL indexed PDFs — which can
# (and did: rag_log.txt, Aug 25 13:40) land on a chunk from the WRONG month
# of the RIGHT year, e.g. a historical year-over-year comparison table on
# page 10 of the October report, instead of the actual figure the question
# asked about. The fix below narrows the search to only the filenames that
# actually match the requested year (and month, if named) before searching,
# instead of hoping similarity ranking sorts it out.
_MONTH_ALIASES = {
    "january": "jan", "jan": "jan",
    "february": "feb", "feb": "feb",
    "march": "mar", "mar": "mar",
    "april": "apr", "apr": "apr",
    "may": "may",
    "june": "jun", "jun": "jun",
    "july": "jul", "jul": "jul",
    "august": "aug", "aug": "aug",
    "september": "sep", "sept": "sep", "sep": "sep",
    "october": "oct", "oct": "oct",
    "november": "nov", "nov": "nov",
    "december": "dec", "dec": "dec",
}


def _detect_requested_month(question: str) -> str:
    """Returns a 3-letter canonical month token ('jan'..'dec') if the
    question names a month, else None. Longest aliases checked first so
    'september' matches before a shorter accidental substring would."""
    q = question.lower()
    for alias, token in sorted(_MONTH_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{alias}\b", q):
            return token
    return None


def _inflation_filenames_for(year: int, month_token: str = None) -> list:
    """Raw filenames (exactly as stored in ChromaDB's 'source' metadata —
    see index_pdfs.py, which writes pdf_file, the unmodified os.listdir()
    name) of indexed inflation reports matching the given year.

    Looked up from live ChromaDB metadata (category="inflation", year=
    <year>) instead of re-scanning PDF_FOLDER filenames for the literal
    substring "inflation" — that scan missed any correctly-indexed report
    named differently (e.g. "CPI_DEC_2022.pdf"). Month narrowing is still
    done against the filename, since month isn't currently stored as its
    own metadata field at index time; if none of the matched files' names
    happen to mention the requested month, all year-matches are returned
    rather than filtering down to zero — better to let embedding
    similarity pick the right month among real candidates than to discard
    them all over a naming quirk.
    """
    sources = _inflation_years_indexed().get(year, [])
    if not sources:
        return []
    if not month_token:
        return sources
    narrowed = [f for f in sources if month_token in f.lower()]
    return narrowed if narrowed else sources

PER_FILE_SCHEMA = build_per_file_schema(CSV_DATA)
META_CONTEXT = "CSV DATASETS:\n" + "".join(list(PER_FILE_SCHEMA.values())) + "\nPDF REPORTS:\n" + "\n".join(PDF_TITLES)

def _gather_sources(question: str, route: str, original_question: str = None):
    """Fetch CSV + PDF context. For 'hybrid', both branches run unconditionally
    and don't depend on each other, so run them concurrently instead of back
    to back — the CSV codegen Groq call and the PDF vector search overlap."""
    csv_result = None
    pdf_context, best_source, best_page = "", "ESS Reference Document", 0

    if route == "hybrid" and CSV_DATA:
        with ThreadPoolExecutor(max_workers=2) as ex:
            csv_future = ex.submit(query_csv_with_groq, question, CSV_DATA, PER_FILE_SCHEMA)
            pdf_future = ex.submit(_run_pdf_search, question, original_question)
            csv_result = csv_future.result()
            pdf_context, source, page = pdf_future.result()
            if pdf_context:
                best_source, best_page = source, page
        return csv_result, pdf_context, best_source, best_page

    if route == "hybrid":
        # hybrid but no CSV_DATA loaded at all
        pdf_context, source, page = _run_pdf_search(question, original_question)
        if pdf_context:
            best_source, best_page = source, page
        return csv_result, pdf_context, best_source, best_page

    if route == "csv" and CSV_DATA:
        csv_result = query_csv_with_groq(question, CSV_DATA, PER_FILE_SCHEMA)

    if route == "pdf" or (route == "csv" and not csv_result):
        pdf_context, source, page = _run_pdf_search(question, original_question)
        if pdf_context:
            best_source, best_page = source, page

    if route == "pdf" and not pdf_context and not csv_result and CSV_DATA:
        csv_result = query_csv_with_groq(question, CSV_DATA, PER_FILE_SCHEMA)

    return csv_result, pdf_context, best_source, best_page


# ═══════════════════════════════════════
# GENERIC METADATA-FIRST PREFILTER (non-inflation categories)
# ═══════════════════════════════════════
# Same idea as the inflation-specific block above — prefer an exact
# (category, year) metadata match over letting embedding similarity alone
# sort out same-topic-different-year chunks — generalized to every other
# category. Deliberately SOFT, unlike the inflation path: if no metadata
# match is found here, we fall through to ordinary similarity search
# rather than refusing outright. Other categories don't have inflation's
# rigid "one file per month" structure or its well-tested EFY messaging,
# so a hard refusal here risks false negatives on report types this
# hasn't been validated against yet.
_CATEGORY_KEYWORDS = {
    "population":    ["population", "demographic", "census"],
    "agriculture":   ["agricultur", "crop", "livestock", "farm"],
    "trade":         ["trade", "import", "export"],
    "manufacturing": ["manufactur"],
    "labour":        ["labour", "labor", "migration", "employment", "unemployment"],
    "housing":       ["housing"],
    "land":          ["land utilization", "land use", "land area"],
    "household":     ["consumption", "welfare", "household"],
}


def _detect_requested_category(question: str) -> str:
    q = question.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(k in q for k in keywords):
            return category
    return None


def _generic_metadata_source_in(question: str, year_detection_text: str) -> list:
    """Best-effort (category, year) -> filenames lookup for any non-price
    question. Returns None (not []) whenever it can't confidently narrow
    the search, so the caller knows to fall back to plain similarity
    rather than treating "no match" as "restrict to nothing"."""
    category = _detect_requested_category(question)
    if not category:
        return None
    requested_year, is_explicit_efy = _detect_requested_year(year_detection_text)
    if requested_year is None:
        return None
    candidate_years = (
        [requested_year] if is_explicit_efy
        else _gregorian_to_efy_candidates(requested_year) + [requested_year]
    )
    metadata_index = get_metadata_index(category)
    for y in candidate_years:
        sources = metadata_index.get(str(y))
        if sources:
            log.info(f"Metadata prefilter: restricting to {len(sources)} file(s) for category={category}, year={y}")
            return sources
    return None


def _run_pdf_search(question: str, original_question: str = None):
    prefer_source = _preferred_source_for(question)
    inflation_source_in = None

    # Year/EFY detection runs against BOTH the rewritten `question` and the
    # raw `original_question` (whatever the user actually typed), not just
    # the rewritten one. rewrite_and_route()'s paraphrase is non-
    # deterministic — most of the time it keeps "EFY 2018" or expands it to
    # "Ethiopian Fiscal Year 2018" (both already handled by the regex
    # below), but it can just as easily drop the EFY marker entirely (e.g.
    # rewriting "October EFY 2018 inflation..." down to "October 2018
    # inflation..."). Once that word is gone, no regex downstream can
    # recover it — _detect_requested_year() then falls back to "bare year
    # -> assume Gregorian" and confidently reports the WRONG year as
    # missing, even though the user was explicit. Checking the original
    # text too means an explicit "EFY" the user typed can never be lost
    # to an unlucky paraphrase.
    year_detection_text = f"{original_question} {question}" if original_question else question

    # See _EFY_LABELED_CATEGORIES comment above: for inflation/price
    # questions that name a year, check up front whether ANY indexed
    # inflation report could possibly cover it. If not, don't bother
    # asking ChromaDB — embedding similarity will still return the
    # closest-topic chunk (a real but wrong-year report) and leave the
    # model to notice and explain the year mismatch buried in prose,
    # which is exactly what produced confusing non-answers before. An
    # explicit, specific notice is clearer for both the model and the user.
    indexed_inflation_years = _inflation_efy_years_indexed()
    if any(k in question.lower() for k in _PRICE_KEYWORDS) and indexed_inflation_years:
        requested_year, is_explicit_efy = _detect_requested_year(year_detection_text)
        if requested_year is not None:
            # Indexed reports aren't guaranteed to all use the same
            # calendar in their "year" metadata — a legacy
            # "...efy-2018-final.pdf" file has an EFY year stored, but a
            # newly uploaded "CPI_DEC_2022.pdf" has a plain Gregorian year
            # stored. So for a bare (non-explicit-EFY) requested year,
            # check it BOTH ways: as a literal year (matches
            # Gregorian-labeled files) and converted to its EFY candidates
            # (matches EFY-labeled files) — rather than assuming every
            # indexed report follows one convention.
            candidate_efys = (
                [requested_year] if is_explicit_efy
                else _gregorian_to_efy_candidates(requested_year) + [requested_year]
            )
            matched_efys = [y for y in candidate_efys if y in indexed_inflation_years]
            if not matched_efys:
                available = ", ".join(str(y) for y in sorted(indexed_inflation_years))
                if is_explicit_efy:
                    year_note = f"EFY {requested_year}"
                else:
                    year_note = f"Gregorian {requested_year} (-> EFY {candidate_efys[0]} or EFY {candidate_efys[1]})"
                log.info(
                    f"No indexed inflation report covers requested year ({year_note}) — "
                    f"short-circuiting instead of retrieving a same-topic wrong-year chunk."
                )
                notice = (
                    f"NO MATCHING REPORT INDEXED. The user asked about {year_note}, but no "
                    f"inflation report for that period is indexed. The inflation report years "
                    f"that ARE indexed: {available} (each as labeled on the source file — may "
                    f"be EFY or Gregorian depending on the report). State this plainly, "
                    f"including both the year the user asked about and which years are "
                    f"actually available — do not guess or substitute a figure from a "
                    f"different year."
                )
                return notice, "ESS Inflation Reports Index", 0

            # Year IS indexed — now narrow to the exact monthly report(s)
            # instead of letting similarity search wander the whole
            # corpus. See _MONTH_ALIASES/_inflation_filenames_for comment
            # above for why: same topic + same year but wrong month can
            # still outrank the actually-requested figure.
            month_token = _detect_requested_month(year_detection_text)
            year_for_filenames = matched_efys[0]
            matches = _inflation_filenames_for(year_for_filenames, month_token)
            if matches:
                inflation_source_in = matches
                log.info(
                    f"Restricting inflation search to {len(matches)} file(s) for "
                    f"EFY {year_for_filenames}" + (f" ({month_token})" if month_token else " (all months)")
                )
    else:
        # Not a price/inflation question — try the generic, soft
        # metadata-first prefilter instead (see comment above its
        # definition). No-op (None) if the question doesn't name a
        # recognizable category+year combination.
        inflation_source_in = _generic_metadata_source_in(question, year_detection_text)

    try:
        results = search_documents(
            question,
            top_k=PDF_RETRIEVAL_TOP_K if not inflation_source_in else max(PDF_RETRIEVAL_TOP_K, len(inflation_source_in)),
            prefer_source=prefer_source,
            source_in=inflation_source_in,
        )
    except Exception as e:
        log.warning(f"search_documents failed: {e}")
        results = None

    if results is None:
        # search_documents() returning None means the search did NOT
        # complete (timeout or exception) — this is "unknown", not
        # "confirmed nothing here". Previously this was indistinguishable
        # from a genuinely empty [] result, so the app told the user
        # "I don't have any data" when the real cause was a 10s retriever
        # timeout (Aug 25 rag_log.txt/retriever_log.txt: the October
        # EFY2018 food/non-food follow-up — the data was in the same file
        # answered correctly six minutes earlier). Be honest instead: this
        # is a transient failure, not a data gap.
        log.warning(f"PDF search did not complete (timeout/error) for: '{question[:50]}'")
        notice = (
            "SEARCH DID NOT COMPLETE. The document search timed out or hit a temporary "
            "error — this is NOT a confirmed absence of data, do not claim the report "
            "lacks this information. Tell the user the search took too long and ask them "
            "to please try the question again."
        )
        return notice, "Search Timeout", 0

    if not results:
        return None, None, 0

    context, source, page = format_context(results)
    return context, source, page

# ═══════════════════════════════════════
# MAIN HYBRID PIPELINE
# ═══════════════════════════════════════
UPLOAD_MAX_CHARS = 12000  # ~3k tokens — lowered from 24000, was blowing Groq's free-tier TPM on large uploads

def _build_upload_prompt(question: str, uploaded_context: str, uploaded_filename: str) -> str:
    """Shared prompt for both get_answer() and get_answer_stream() when the
    user has an uploaded document attached. Deliberately answers from the
    uploaded text ONLY — not the ESS corpus — since a user who just attached
    a specific file almost always means "answer from THIS document", not
    "search everything you have"."""
    context = uploaded_context[:UPLOAD_MAX_CHARS]
    return f"""You are the ESS (Ethiopian Statistical Service) AI Assistant.
The user has attached a document called "{uploaded_filename}". Answer their
question using ONLY the document text below, IN ENGLISH.

DOCUMENT TEXT:
{context}

STRICT RULES:
1. Extract every number, percentage, total, rate, or year that helps answer the question.
2. Lead with the key figure(s) if the question asks for one.
   (If a source above is a Markdown table — rows separated by | with a --- header line — match the question to the exact row and column; don't just describe the table.)
3. If the document does not contain the answer, say so plainly in one short sentence — do not guess or pull in outside knowledge.
4. Max 180 words. Be factual and direct.

Question: {question}
Answer:"""


def get_answer(question: str, chat_history: list = None,
                uploaded_context: str = None, uploaded_filename: str = None,
                force_lang: str = None) -> dict:
    original_question = question
    log.info(f"Processing Question: '{question[:80]}'")

    # force_lang lets the caller (e.g. the widget's EN/አማ toggle) override
    # auto-detection so the ANSWER always comes back in the language the
    # user selected in the UI, regardless of what language the question
    # itself was typed in. Falls back to detection when not given/invalid.
    target_lang = force_lang if force_lang in ("en", "am") else resolve_language(original_question)

    # An uploaded document takes priority over the ESS corpus and bypasses
    # the cache entirely — the same question text can mean something
    # completely different depending on which file is attached, so caching
    # it under the question alone would return another user's (or another
    # session's) stale answer.
    if uploaded_context:
        final_prompt = _build_upload_prompt(question, uploaded_context, uploaded_filename or "the uploaded document")
        answer_en = call_groq(final_prompt, max_tokens=500, temperature=ANSWER_TEMPERATURE)
        if is_bad_answer(answer_en):
            # Distinguish "couldn't reach any LLM provider" from "the doc
            # doesn't answer this" — the generic GROQ_UNAVAILABLE_MSG reads
            # like the file was ignored, when the real cause is Groq/Gemini/
            # Ollama all being unreachable (see call_groq's fallback chain).
            answer_en = (f'Could not reach the AI service to read "{uploaded_filename or "your document"}" '
                         f'right now. Check your internet connection (Groq/Gemini need it) or make sure '
                         f'Ollama is running locally, then try again.')
        answer = translate_answer(answer_en, target_lang)
        return {
            "answer": answer,
            "source": uploaded_filename or "Uploaded document",
            "page": 0,
            "route": "upload",
            "cached": False,
            "language": _actual_answer_lang(answer, target_lang),
            "resolved_question": None,
        }

    # Fast path: try the cache with the RAW question first. Most repeat
    # questions are typed the same way twice — this skips a whole rewrite+
    # route LLM round trip (and the token spend/latency that goes with it)
    # on what will turn out to be a cache hit anyway. If this misses, we
    # fall through to rewrite_and_route + a second cache check on the
    # cleaned question, same as before (covers paraphrases).
    early_cached = check_cache(original_question, target_lang)
    if early_cached:
        return early_cached

    question, route = rewrite_and_route(question, chat_history)

    cached = check_cache(question, target_lang)
    if cached:
        return cached

    log.info(f"Route Selected: {route} | Target Language: {target_lang}")

    if route == "meta":
        prompt = f"""You are the official ESS (Ethiopian Statistical Service) AI Assistant.
Respond warmly, professionally, and concisely to this identity/system question, IN ENGLISH.
State that you can answer questions using {SOURCE_FORMATS_DESC}.
Available Data Summary:
{META_CONTEXT}
Question: {question}
Max 100 words."""
        answer_en = call_groq(prompt, max_tokens=400, temperature=ANSWER_TEMPERATURE)
        answer = translate_answer(answer_en, target_lang)
        result = {"answer": answer, "source": "ESS Asset Index", "page": 0, "route": "meta", "cached": False, "language": _actual_answer_lang(answer, target_lang)}
        save_cache(question, answer, result["source"], result["page"], target_lang)
        return result

    effective_route = route
    csv_result, pdf_context, best_source, best_page = _gather_sources(question, route, original_question)

    # NOTE: all generation prompts below now instruct ENGLISH output only.
    # The target-language version is produced afterward by translate_answer(),
    # a dedicated translation call — this is more reliable for Amharic than
    # asking the model to reason-and-write directly in that
    # language in one shot. See translate_answer() above for why.
    if csv_result and pdf_context:
        effective_route = "hybrid"
        final_prompt = f"""You are the ESS (Ethiopian Statistical Service) AI Assistant.
Answer the user's question using BOTH sources below, IN ENGLISH.

DATABASE LOOKUP RESULT:
{csv_result}

REPORT TEXT:
{pdf_context}

STRICT RULES:
1. State every relevant number, percentage, total, or year that answers the question.
2. Prefer concrete figures over vague statements — e.g. "was 12.3% in 2023", not "the report mentions trends".
3. Before answering, check: does a number in the sources above actually match the specific thing
   asked (same region/category/year)? If yes, state it. If the sources only cover a different
   region/year/category, say plainly that the exact figure isn't in the retrieved sources rather
   than reporting the closest number as if it answered the question.
   (If a source above is a Markdown table — rows separated by | with a --- header line — match the question to the exact row and column; don't just describe the table.)
4. If both sources give numbers, lead with the database result and add brief context from the report.
5. {_EFY_RULE}
6. Max 150 words. Be direct and factual. Source: {best_source} (Page {best_page})

Question: {question}
Answer:"""
        answer_en = call_groq(final_prompt, max_tokens=500, temperature=ANSWER_TEMPERATURE)
        best_source = f"ESS Database & {best_source}"

    elif csv_result:
        effective_route = "csv"
        final_prompt = f"""You are the ESS (Ethiopian Statistical Service) AI Assistant.
Answer the question using the database result below, IN ENGLISH.

DATABASE RESULT: {csv_result}

RULES:
1. State the number(s) clearly and directly — lead with the figure.
2. Do not invent extra figures not present above.
3. Do not say the data is missing if a result is present above.
4. Max 80 words.

Question: {question}
Answer:"""
        answer_en = call_groq(final_prompt, max_tokens=400, temperature=ANSWER_TEMPERATURE)
        best_source = "ESS Statistical Database"

    elif pdf_context:
        effective_route = "pdf"
        final_prompt = f"""You are the ESS (Ethiopian Statistical Service) AI Assistant.
Answer the question using ONLY the report text below, IN ENGLISH.

REPORT TEXT:
{pdf_context}

Source document: {best_source} (Page {best_page})

STRICT RULES:
1. Extract every number, percentage, total, rate, or year that helps answer the question.
2. Lead with the key figure(s), e.g. "According to the report, the inflation rate was X% in YEAR."
3. Before answering, check: does a number in the text above actually match the specific thing asked
   (same region/category/year)? If yes, state it clearly. If the text only covers a different
   region/year/category, say the exact figure isn't in this report rather than reporting the
   closest number as if it answered the question.
   (If a source above is a Markdown table — rows separated by | with a --- header line — match the question to the exact row and column; don't just describe the table.)
4. If multiple years or regions appear, list the most relevant ones clearly.
5. Only say nothing relevant was found if that is genuinely true — then say so in one short sentence.
6. Max 150 words. Be factual and direct.
7. The text above may contain chunks from MORE THAN ONE source document, each tagged
   "[Source N: filename - Page X ...]". NEVER combine numbers from two different [Source N] tags
   into a single answer, even if they look related (e.g. a population figure from [Source 1] and a
   household count from [Source 2]) — that misattributes facts to the wrong document. Pick the
   single [Source N] that most directly answers the question and answer from that one only. If you
   used a source other than the top one ({best_source}), say which document/page you actually used
   instead of the one listed above.
8. {_EFY_RULE}
9. Always finish every sentence. Never stop mid-sentence or mid-list. If you start a number or region, complete the full figure.
10. EXCEPTION to rule 7: if the sources are monthly inflation reports (filenames containing
    "inflation") and more than one month appears in the text above, that is expected — these
    reports are monthly, not annual — so list EACH month's figure with its own month/EFY year
    (e.g. "EFY 2018: Jan 4.2%, Feb 3.9%, ..."), don't pick just one and don't say the year's rate
    is unavailable just because no single "annual" number exists.

Question: {question}
Answer:"""
        answer_en = call_groq(final_prompt, max_tokens=700, temperature=ANSWER_TEMPERATURE)
    else:
        effective_route = "fallback"
        final_prompt = f"""You are the ESS (Ethiopian Statistical Service) AI Assistant.
No matching database result or report text was found for this question.
Reply in clear English, in one or two short sentences. Do not invent statistics or sources.
Question: {question}
Answer:"""
        answer_en = call_groq(final_prompt, max_tokens=350, temperature=ANSWER_TEMPERATURE)

    answer = translate_answer(answer_en, target_lang)

    save_cache(question, answer, best_source, best_page, target_lang)
    return {
        "answer": answer,
        "source": best_source,
        "page": best_page,
        "route": effective_route,
        "cached": False,
        "language": _actual_answer_lang(answer, target_lang),
        "resolved_question": question if question != original_question else None,
    }

# ═══════════════════════════════════════
# STREAMING VARIANT
# ═══════════════════════════════════════
def get_answer_stream(question: str, chat_history: list = None,
                       uploaded_context: str = None, uploaded_filename: str = None,
                       force_lang: str = None):
    original_question = question
    log.info(f"Processing Question (stream): '{question[:80]}'")

    target_lang = force_lang if force_lang in ("en", "am") else resolve_language(original_question)

    if uploaded_context:
        final_prompt = _build_upload_prompt(question, uploaded_context, uploaded_filename or "the uploaded document")
        upload_unavailable_msg = (f'Could not reach the AI service to read "{uploaded_filename or "your document"}" '
                                   f'right now. Check your internet connection (Groq/Gemini need it) or make sure '
                                   f'Ollama is running locally, then try again.')
        full_answer = None
        for kind, out in _stream_or_translate(final_prompt, 260, target_lang, unavailable_msg=upload_unavailable_msg):
            yield (kind, out)
            if kind == "done":
                full_answer = out["answer"]
        result = {
            "answer": full_answer,
            "source": uploaded_filename or "Uploaded document",
            "page": 0,
            "route": "upload",
            "cached": False,
            "language": _actual_answer_lang(full_answer, target_lang),
            "resolved_question": None,
        }
        yield ("done", result)
        return

    # Fast path: same reasoning as get_answer() above — try the raw question
    # against the cache before paying for a rewrite+route LLM call.
    early_cached = check_cache(original_question, target_lang)
    if early_cached:
        yield ("chunk", early_cached["answer"])
        yield ("done", early_cached)
        return

    question, route = rewrite_and_route(question, chat_history)

    cached = check_cache(question, target_lang)
    if cached:
        yield ("chunk", cached["answer"])
        yield ("done", cached)
        return

    log.info(f"Route Selected: {route} | Target Language: {target_lang}")

    if route == "meta":
        prompt = f"""You are the official ESS (Ethiopian Statistical Service) AI Assistant.
Respond warmly, professionally, and concisely to this identity/system question, IN ENGLISH.
State that you can answer questions using {SOURCE_FORMATS_DESC}.
Available Data Summary:
{META_CONTEXT}
Question: {question}
Max 100 words."""
        for kind, out in _stream_or_translate(prompt, 180, target_lang):
            yield (kind, out)
            if kind == "done":
                meta = out
        result = {"answer": meta["answer"], "source": "ESS Asset Index", "page": 0, "route": "meta", "cached": False, "language": _actual_answer_lang(meta["answer"], target_lang)}
        save_cache(question, meta["answer"], result["source"], result["page"], target_lang)
        yield ("done", result)
        return

    effective_route = route
    csv_result, pdf_context, best_source, best_page = _gather_sources(question, route, original_question)

    if csv_result and pdf_context:
        effective_route = "hybrid"
        final_prompt = f"""You are the ESS (Ethiopian Statistical Service) AI Assistant.
Answer the user's question using BOTH sources below, IN ENGLISH.

DATABASE LOOKUP RESULT:
{csv_result}

REPORT TEXT:
{pdf_context}

STRICT RULES:
1. State every relevant number, percentage, total, or year that answers the question.
2. Prefer concrete figures over vague statements — e.g. "was 12.3% in 2023", not "the report mentions trends".
3. Before answering, check: does a number in the sources above actually match the specific thing
   asked (same region/category/year)? If yes, state it. If the sources only cover a different
   region/year/category, say plainly that the exact figure isn't in the retrieved sources rather
   than reporting the closest number as if it answered the question.
   (If a source above is a Markdown table — rows separated by | with a --- header line — match the question to the exact row and column; don't just describe the table.)
4. If both sources give numbers, lead with the database result and add brief context from the report.
5. {_EFY_RULE}
6. Max 150 words. Be direct and factual. Source: {best_source} (Page {best_page})

Question: {question}
Answer:"""
        max_tok = 500
        best_source = f"ESS Database & {best_source}"

    elif csv_result:
        effective_route = "csv"
        final_prompt = f"""You are the ESS (Ethiopian Statistical Service) AI Assistant.
Answer the question using the database result below, IN ENGLISH.

DATABASE RESULT: {csv_result}

RULES:
1. State the number(s) clearly and directly — lead with the figure.
2. Do not invent extra figures not present above.
3. Do not say the data is missing if a result is present above.
4. Max 80 words.

Question: {question}
Answer:"""
        max_tok = 400
        best_source = "ESS Statistical Database"

    elif pdf_context:
        effective_route = "pdf"
        final_prompt = f"""You are the ESS (Ethiopian Statistical Service) AI Assistant.
Answer the question using ONLY the report text below, IN ENGLISH.

REPORT TEXT:
{pdf_context}

Source document: {best_source} (Page {best_page})

STRICT RULES:
1. Extract every number, percentage, total, rate, or year that helps answer the question.
2. Lead with the key figure(s), e.g. "According to the report, the inflation rate was X% in YEAR."
3. Before answering, check: does a number in the text above actually match the specific thing asked
   (same region/category/year)? If yes, state it clearly. If the text only covers a different
   region/year/category, say the exact figure isn't in this report rather than reporting the
   closest number as if it answered the question.
   (If a source above is a Markdown table — rows separated by | with a --- header line — match the question to the exact row and column; don't just describe the table.)
4. If multiple years or regions appear, list the most relevant ones clearly.
5. Only say nothing relevant was found if that is genuinely true — then say so in one short sentence.
6. Max 150 words. Be factual and direct.
7. The text above may contain chunks from MORE THAN ONE source document, each tagged
   "[Source N: filename - Page X ...]". NEVER combine numbers from two different [Source N] tags
   into a single answer, even if they look related (e.g. a population figure from [Source 1] and a
   household count from [Source 2]) — that misattributes facts to the wrong document. Pick the
   single [Source N] that most directly answers the question and answer from that one only. If you
   used a source other than the top one ({best_source}), say which document/page you actually used
   instead of the one listed above.
8. {_EFY_RULE}
9. EXCEPTION to rule 7: if the sources are monthly inflation reports (filenames containing
   "inflation") and more than one month appears in the text above, that is expected — these
   reports are monthly, not annual — so list EACH month's figure with its own month/EFY year
   (e.g. "EFY 2018: Jan 4.2%, Feb 3.9%, ..."), don't pick just one and don't say the year's rate
   is unavailable just because no single "annual" number exists.

Question: {question}
Answer:"""
        max_tok = 500

    else:
        effective_route = "fallback"
        final_prompt = f"""You are the ESS (Ethiopian Statistical Service) AI Assistant.
No matching database result or report text was found for this question.
Reply in clear English, in one or two short sentences. Do not invent statistics or sources.
Question: {question}
Answer:"""
        max_tok = 350

    full_answer = None
    for kind, out in _stream_or_translate(final_prompt, max_tok, target_lang):
        yield (kind, out)
        if kind == "done":
            full_answer = out["answer"]

    save_cache(question, full_answer, best_source, best_page, target_lang)
    result = {
        "answer": full_answer,
        "source": best_source,
        "page": best_page,
        "route": effective_route,
        "cached": False,
        "language": _actual_answer_lang(full_answer, target_lang),
        "resolved_question": question if question != original_question else None,
    }
    yield ("done", result)


def _stream_or_translate(prompt: str, max_tokens: int, target_lang: str, unavailable_msg: str = None):
    """Shared streaming helper for get_answer_stream():
    - English: stream tokens live as they're generated (unchanged behavior).
    - Amharic: generate the English answer first (not streamed, since
      it isn't the final text), then STREAM the translation — user still
      sees live token-by-token output, it's just the translation pass.
    Yields ("chunk", text) any number of times, then exactly one
    ("done", {"answer": full_final_text}).

    unavailable_msg: override for the "couldn't generate an answer" text
    (default GROQ_UNAVAILABLE_MSG) — callers with more context (e.g. an
    uploaded document) can pass a more specific message."""
    unavailable_msg = unavailable_msg or GROQ_UNAVAILABLE_MSG
    if target_lang != "am":
        full = ""
        for piece in call_groq_stream(prompt, max_tokens=max_tokens, temperature=ANSWER_TEMPERATURE):
            full += piece
            yield ("chunk", piece)
        if not full.strip():
            full = unavailable_msg
            yield ("chunk", full)
        yield ("done", {"answer": full})
        return

    answer_en = call_groq(prompt, max_tokens=max_tokens, temperature=ANSWER_TEMPERATURE)
    if is_bad_answer(answer_en):
        yield ("chunk", unavailable_msg)
        yield ("done", {"answer": unavailable_msg})
        return

    # Try Google Translate first (not streaming — sent as one chunk, same
    # pattern already used above for the non-streamed English-answer step).
    gt = _google_translate(answer_en, target_lang)
    if gt:
        yield ("chunk", gt)
        yield ("done", {"answer": gt})
        return

    lang_name = "Amharic (አማርኛ)"
    few_shot = _TRANSLATE_FEW_SHOT.get(target_lang, "")
    glossary = _build_glossary_block(target_lang)
    translate_prompt = f"""Translate the following English answer into {lang_name}.

{glossary}{few_shot}Rules:
- Translate ONLY — do not add, remove, or reinterpret any information.
- Preserve every number, statistic, date, and percentage EXACTLY as written.
- Keep proper nouns, region names, and document/report titles in their recognizable form.
- Output ONLY the translation, nothing else — no notes, no English text alongside it.

English answer:
\"{answer_en}\"

{lang_name} translation:"""

    full = ""
    # Amharic script tokenizes far less efficiently than English — the
    # same content needs roughly 1.5-2x the tokens, so reusing the English
    # max_tokens here was cutting translations off mid-sentence. Bumped from
    # 1.8x/900 cap to 2.2x/1100: dense report answers (multiple regions,
    # percentages, years) sit close to their own English token ceiling
    # already, so 1.8x wasn't leaving enough room and translations were
    # truncating mid-word (e.g. "...ተጠ" instead of "...ተጠቀሙ").
    translate_max_tokens = min(int(max_tokens * 2.2), 1100)
    for piece in call_groq_stream(translate_prompt, max_tokens=translate_max_tokens):
        full += piece
        yield ("chunk", piece)
    if not full.strip() or not _translation_quality_ok(answer_en, full):
        # Translation stream failed or produced garbage/echoed-English —
        # fall back to the English answer rather than showing a broken or
        # empty result. Already-streamed chunks can't be un-sent, but the
        # final saved/cached "done" answer will be the trustworthy one.
        full = answer_en
        yield ("chunk", full)
    yield ("done", {"answer": full})


if __name__ == "__main__":
    ensure_cache_table()
    print("Multilingual RAG Engine initialized.")