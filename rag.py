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

# CRITICAL: must run before importing retriever — retriever.py initializes
# the SentenceTransformer model at import time, which reads HF_HUB_OFFLINE /
# TRANSFORMERS_OFFLINE from the environment. If .env hasn't been loaded yet,
# those flags are unset and the model init hits the network on every
# startup instead of using the local cache (this was the cause of the slow,
# repeated HuggingFace HEAD requests on startup).
load_dotenv()

from retriever import search_documents, format_context

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
# Wall-clock cap per Groq API call so a stalled connection can't hang a worker
# indefinitely (e.g. mid-stream network drop). Applies to both the normal and
# streaming call paths below.
GROQ_REQUEST_TIMEOUT     = float(os.getenv("GROQ_REQUEST_TIMEOUT", "5.0"))
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
PDF_RETRIEVAL_TOP_K      = int(os.getenv("PDF_RETRIEVAL_TOP_K", "5"))
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

# ═══════════════════════════════════════
# GROQ CLIENT
# ═══════════════════════════════════════
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)

# ═══════════════════════════════════════
# OLLAMA FALLBACK (local, no API)
# Only used when Groq has fully failed for a request (rate-limited on both
# models, timed out, or erroring) — NOT the primary path. CPU-only laptop
# here, so keep OLLAMA_MODEL small (e.g. llama3.2:1b) or this will be slow
# enough to defeat the purpose. Set OLLAMA_ENABLED=false in .env to disable
# entirely and fall back to the old GROQ_UNAVAILABLE_MSG behavior.
# ═══════════════════════════════════════
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_ENABLED  = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
OLLAMA_TIMEOUT  = float(os.getenv("OLLAMA_TIMEOUT", "60"))


def _call_ollama(prompt: str, max_tokens: int = 500) -> str:
    """Non-streaming local fallback. Returns '' on any failure (model not
    pulled, Ollama not running, timeout) so callers can detect total failure
    and fall through to GROQ_UNAVAILABLE_MSG same as before."""
    if not OLLAMA_ENABLED:
        return ""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.0},
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
        return ""


def _call_ollama_stream(prompt: str, max_tokens: int = 500):
    """Streaming local fallback. Yields text chunks as they arrive; yields
    nothing at all on failure, so the caller's `yielded_any` check correctly
    falls through to GROQ_UNAVAILABLE_MSG."""
    if not OLLAMA_ENABLED:
        return
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": max_tokens, "temperature": 0.0},
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
        return

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
                tz = row[3].tzinfo if row[3].tzinfo else timezone.utc
                age = datetime.now(tz) - row[3]
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

def is_bad_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    a = answer.strip().lower()
    if answer.strip() in (GROQ_UNAVAILABLE_MSG, BUDGET_EXHAUSTED_MSG):
        return True
    if a in ("none", "n/a", "error", "null"):
        return True
    # The model's own "I looked, nothing was in the retrieved text" sentinel
    # (see the PDF/hybrid prompts' rule 5). This is a legitimate answer to
    # show the user once, but must never be cached — the next question on
    # the same topic deserves a fresh retrieval attempt, not a permanently
    # stuck "not found" from one weak chunk pull.
    if "nothing relevant" in a or "no relevant" in a:
        return True
    return False

def save_cache(question: str, answer: str, source: str, page: int, language: str = "en"):
    if is_bad_answer(answer):
        log.warning(f"Skipped caching bad/sentinel answer for: '{question[:50]}'")
        return
    try:
        safe_page = int(page) if (page is not None and str(page).isdigit()) else 0
        q_hash = get_question_hash(question, language)
        with db_cursor(commit=True) as cur:
            cur.execute("""
                INSERT INTO query_cache (question_hash, answer, source_doc, source_page)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (question_hash) DO NOTHING
            """, (q_hash, answer, source, safe_page))
    except Exception as e:
        log.error(f"Cache save error: {str(e)}")

# ═══════════════════════════════════════
# GROQ CALL
# ═══════════════════════════════════════
# Matches Groq's rate-limit error body in either unit it uses:
# "...Please try again in 8.0s." or "...Please try again in 370ms."
_RETRY_AFTER_RE = re.compile(r"try again in\s+([\d.]+)\s*(ms|s)\b", re.IGNORECASE)

def call_groq(prompt: str, retries: int = 2, max_tokens: int = 500, model: str = None) -> str:
    use_model = model or MODEL

    if _token_budget.should_refuse():
        log.warning(f"Daily token budget exhausted ({_token_budget.ratio():.0%}) — refusing Groq call.")
        return BUDGET_EXHAUSTED_MSG
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
                return _call_ollama(prompt, max_tokens=max_tokens) or GROQ_UNAVAILABLE_MSG

        _throttle_for(use_model).wait_if_needed()
        try:
            response = groq_client.chat.completions.create(
                model       = use_model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = max_tokens,
                temperature = 0.0,
                timeout     = GROQ_REQUEST_TIMEOUT
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
    # try the local Ollama fallback before giving up entirely.
    return _call_ollama(prompt, max_tokens=max_tokens) or GROQ_UNAVAILABLE_MSG

def call_groq_stream(prompt: str, max_tokens: int = 500, model: str = None):
    use_model = model or MODEL

    if _token_budget.should_refuse():
        log.warning(f"Daily token budget exhausted ({_token_budget.ratio():.0%}) — refusing Groq stream call.")
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
                # Both attempts blocked by cooldown, nothing streamed yet — try local fallback.
                for piece in _call_ollama_stream(prompt, max_tokens=max_tokens):
                    yielded_any = True
                    yield piece
                if yielded_any:
                    log.info(f"Streamed answer via Ollama fallback ('{OLLAMA_MODEL}') — Groq models all cooling down.")
                return

        _throttle_for(use_model).wait_if_needed()
        try:
            create_kwargs = dict(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                stream=True,
                timeout=GROQ_REQUEST_TIMEOUT,
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
                # Ollama fallback (not running / model not pulled) degrades
                # to the exact old behavior.
                fell_back = False
                for piece in _call_ollama_stream(prompt, max_tokens=max_tokens):
                    fell_back = True
                    yield piece
                if fell_back:
                    log.info(f"Streamed answer via Ollama fallback ('{OLLAMA_MODEL}') after Groq exhausted.")
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

_OROMO_KEYWORDS = [
    "maal", "kam", "eessa", "yoom", "maaliif", "akkam", "meeqa", "haala", "gabaasa",
    "hangam", "eenyu", "maalif", "danda", "argam", "jira", "jiru", "kessa", "keessa",
    "biyya", "godina", "bara", "lakkoofsa", "baay", "xiqqaa", "guddaa", "dhiyeessi",
    "oromiyaa", "qonna", "horsiisa", "beela", "gabaa", "sadarkaa", "ummata", "ummanni",
    "naannoo", "waggaa", "ji'a", "kkf", "fi ", "ni ", "hin ", "irratti"
]

_ENGLISH_STOPWORDS = {
    "the", "is", "are", "what", "how", "many", "much", "population", "rate",
    "when", "where", "which", "average", "total", "percent", "percentage",
    "in", "of", "for", "was", "were", "and", "report", "data"
}

def detect_language(question: str) -> str:
    q_lower = question.lower()
    if _contains_amharic_script(question):
        return "am"

    words = re.findall(r"[a-z']+", q_lower)
    if not words:
        return "en"

    oromo_hits = sum(1 for w in words if w in _OROMO_KEYWORDS)
    english_hits = sum(1 for w in words if w in _ENGLISH_STOPWORDS)

    # Require actual signal either way; if neither language shows evidence,
    # DON'T silently default to English — let the caller fall back to an
    # LLM-based classification (see detect_language_llm) instead of guessing.
    if oromo_hits > 0 and oromo_hits >= english_hits:
        return "om"
    if english_hits > 0:
        return "en"

    return "uncertain"


def detect_language_llm(question: str) -> str:
    """Fallback classifier for questions the heuristic can't confidently tag
    (e.g. Afaan Oromoo phrased without any of the whitelisted words, or
    Amharic typed phonetically in Latin script). Cheap/fast model, tiny output."""
    prompt = f"""Identify the language of this question. It is ESS (Ethiopian Statistical Service)
chatbot input, so it will be English, Amharic (possibly typed phonetically in Latin letters),
or Afaan Oromoo. Respond with exactly one lowercase code: en, am, or om. No explanation.

Question: "{question}\""""
    raw = call_groq(prompt, retries=1, max_tokens=3, model=FAST_MODEL)
    code = re.sub(r"[^a-z]", "", (raw or "").lower().strip())
    return code if code in ("en", "am", "om") else "en"


def resolve_language(question: str) -> str:
    """Fast path for the common cases (Fidel script / clear English), LLM
    fallback only for the ambiguous minority — so this doesn't add latency
    to most requests."""
    lang = detect_language(question)
    if lang == "uncertain":
        lang = detect_language_llm(question)
        log.info(f"Language heuristic uncertain, LLM classified as '{lang}': '{question[:50]}'")
    return lang

def get_language_instructions(lang_code: str) -> str:
    if lang_code == "am":
        return (
            "\nCRITICAL: Respond in Clear, Formal Amharic (አማርኛ) ONLY.\n"
            "- Extract the factual values from the English source materials and write your output in standard Amharic script.\n"
            "- Keep system terms, table titles, or specific document names in English if no direct Amharic translation exists."
        )
    elif lang_code == "om":
        return (
            "\nCRITICAL: Respond in Clear, Grammatical Afaan Oromoo ONLY.\n"
            "- Translate technical facts and statistics from the English sources into Afaan Oromoo.\n"
            "- Keep proper nouns or precise document references in English where translation might confuse the source."
        )
    return "\nCRITICAL: Respond in Clear, Formal English ONLY."

# ═══════════════════════════════════════
# TERMINOLOGY GLOSSARY (fill this in with ESS staff — native speakers only,
# never guess these yourself and never trust the LLM's own guess here).
# Add one entry per row: "English term": {"am": "Amharic", "om": "Afaan Oromoo"}
# Every term listed here gets forced into the translation prompt below, so
# the model uses YOUR approved wording instead of inventing its own.
# ═══════════════════════════════════════
GLOSSARY_TERMS = {
    # "Household Consumption Expenditure Survey": {"am": "", "om": ""},
    # "Meher season": {"am": "", "om": ""},
    # "Belg season": {"am": "", "om": ""},
    # "Ethiopian Statistical Service": {"am": "", "om": ""},
    # add real entries here — leave empty dict {} until you have them
}

def _build_glossary_block(target_lang: str) -> str:
    """Turns GLOSSARY_TERMS into a forced-wording block for the translation
    prompt. Terms with an empty translation for this language are skipped."""
    rows = []
    for en_term, translations in GLOSSARY_TERMS.items():
        val = (translations or {}).get(target_lang, "").strip()
        if val:
            rows.append(f'  "{en_term}" -> "{val}"')
    if not rows:
        return ""
    return "Mandatory glossary — use this EXACT wording whenever these terms appear, do not translate them any other way:\n" + "\n".join(rows) + "\n\n"

# ═══════════════════════════════════════
# FINAL-ANSWER TRANSLATION (AM/OM quality fix)
# ═══════════════════════════════════════
# Root cause of weaker Amharic/Oromo answers: asking the model to REASON
# and WRITE directly in a low-resource language in one shot is noticeably
# less reliable than asking it to reason in English (where it's strongest)
# and then translate the finished answer. This is a dedicated, narrow
# translation call — no reasoning required, so a small/fast model handles
# it well, and numbers/sources are far less likely to drift.
_TRANSLATE_FEW_SHOT = {
    "am": (
        "Example:\n"
        "English: \"The projected population of Ethiopia for 2025 is approximately 110 million, "
        "according to the Ethiopian Statistical Service.\"\n"
        "Amharic: \"በኢትዮጵያ ስታቲስቲክስ አገልግሎት መሠረት፣ የኢትዮጵያ የ2025 ዓ.ም ግምታዊ የሕዝብ ብዛት 110 ሚሊዮን ገደማ ነው።\"\n\n"
    ),
    "om": (
        "Example:\n"
        "English: \"The projected population of Ethiopia for 2025 is approximately 110 million, "
        "according to the Ethiopian Statistical Service.\"\n"
        "Afaan Oromoo: \"Akka Tajaajila Istaatistiksii Itoophiyaatti, baay'inni ummata Itoophiyaa "
        "bara 2025 tilmaamaan miliyoona 110 ta'a.\"\n\n"
    ),
}

def translate_answer(english_answer: str, target_lang: str) -> str:
    """Translate a finished English answer into Amharic/Afaan Oromoo.
    Returns the English answer unchanged for 'en' or on translation failure
    (never block the user on a translation error)."""
    if target_lang not in ("am", "om") or not english_answer or not english_answer.strip():
        return english_answer

    lang_name = "Amharic (አማርኛ)" if target_lang == "am" else "Afaan Oromoo"
    few_shot = _TRANSLATE_FEW_SHOT.get(target_lang, "")
    glossary = _build_glossary_block(target_lang)

    prompt = f"""Translate the following English answer into {lang_name}.

{glossary}{few_shot}Rules:
- Translate ONLY — do not add, remove, or reinterpret any information.
- Preserve every number, statistic, date, and percentage EXACTLY as written.
- Keep proper nouns, region names, and document/report titles in their recognizable form.
- Output ONLY the translation, nothing else — no notes, no English text alongside it.

English answer:
\"{english_answer}\"

{lang_name} translation:"""

    translated = call_groq(prompt, retries=2, max_tokens=400, model=MODEL)
    if is_bad_answer(translated):
        log.warning(f"Translation to '{target_lang}' failed, falling back to English answer.")
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
    rewrite_model = MODEL if target_lang in ("am", "om") else FAST_MODEL

    prompt = f"""You clean up user questions for the Ethiopian Statistical Service (ESS) AI Assistant
before they are routed to a database or document search. The document/CSV corpus is English-only.

Do ALL of the following that apply:
1. Fix typos, misspellings, or phonetically written Amharic/Oromo-in-Latin-script text.
2. CRITICAL: If written in native Amharic script (Fidel) or Afaan Oromoo, TRANSLATE into a clear,
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

    cleaned = call_groq(prompt, retries=2, max_tokens=100, model=rewrite_model)

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

    raw = call_groq(prompt, retries=2, max_tokens=5, model=FAST_MODEL)
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
    rewrite_model = MODEL if target_lang in ("am", "om") else FAST_MODEL

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

Prefer "pdf" when the question mentions: report, survey, findings, statistics, characteristics, analysis, methodology, projected, trade, manufacturing, livestock, housing, labour/labor, inflation report.

{"Recent conversation:" if history_block else ""}
{history_block}

Original question: "{question}"

Respond with ONLY this JSON, no markdown fences, no commentary:
{{"question": "<cleaned English question>", "route": "<meta|csv|pdf|hybrid>"}}"""

    raw = call_groq(prompt, retries=2, max_tokens=180, model=rewrite_model)

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
PER_FILE_SCHEMA = build_per_file_schema(CSV_DATA)
META_CONTEXT = "CSV DATASETS:\n" + "".join(list(PER_FILE_SCHEMA.values())) + "\nPDF REPORTS:\n" + "\n".join(PDF_TITLES)

def _gather_sources(question: str, route: str):
    """Fetch CSV + PDF context. For 'hybrid', both branches run unconditionally
    and don't depend on each other, so run them concurrently instead of back
    to back — the CSV codegen Groq call and the PDF vector search overlap."""
    csv_result = None
    pdf_context, best_source, best_page = "", "ESS Reference Document", 0

    if route == "hybrid" and CSV_DATA:
        with ThreadPoolExecutor(max_workers=2) as ex:
            csv_future = ex.submit(query_csv_with_groq, question, CSV_DATA, PER_FILE_SCHEMA)
            pdf_future = ex.submit(_run_pdf_search, question)
            csv_result = csv_future.result()
            pdf_context, source, page = pdf_future.result()
            if pdf_context:
                best_source, best_page = source, page
        return csv_result, pdf_context, best_source, best_page

    if route == "hybrid":
        # hybrid but no CSV_DATA loaded at all
        pdf_context, source, page = _run_pdf_search(question)
        if pdf_context:
            best_source, best_page = source, page
        return csv_result, pdf_context, best_source, best_page

    if route == "csv" and CSV_DATA:
        csv_result = query_csv_with_groq(question, CSV_DATA, PER_FILE_SCHEMA)

    if route == "pdf" or (route == "csv" and not csv_result):
        pdf_context, source, page = _run_pdf_search(question)
        if pdf_context:
            best_source, best_page = source, page

    if route == "pdf" and not pdf_context and not csv_result and CSV_DATA:
        csv_result = query_csv_with_groq(question, CSV_DATA, PER_FILE_SCHEMA)

    return csv_result, pdf_context, best_source, best_page


def _run_pdf_search(question: str):
    try:
        results = search_documents(question, top_k=PDF_RETRIEVAL_TOP_K)
    except Exception as e:
        log.warning(f"search_documents failed: {e}")
        results = None

    if not results:
        return None, None, 0

    context, source, page = format_context(results)
    return context, source, page

# ═══════════════════════════════════════
# MAIN HYBRID PIPELINE
# ═══════════════════════════════════════
def get_answer(question: str, chat_history: list = None) -> dict:
    original_question = question
    log.info(f"Processing Question: '{question[:80]}'")

    target_lang = resolve_language(original_question)

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
        answer_en = call_groq(prompt, max_tokens=180)
        answer = translate_answer(answer_en, target_lang)
        result = {"answer": answer, "source": "ESS Asset Index", "page": 0, "route": "meta", "cached": False, "language": target_lang}
        save_cache(question, answer, result["source"], result["page"], target_lang)
        return result

    effective_route = route
    csv_result, pdf_context, best_source, best_page = _gather_sources(question, route)

    # NOTE: all generation prompts below now instruct ENGLISH output only.
    # The target-language version is produced afterward by translate_answer(),
    # a dedicated translation call — this is more reliable for Amharic/Afaan
    # Oromoo than asking the model to reason-and-write directly in that
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
3. NEVER say the figure is not provided/mentioned if any relevant number exists in the sources above.
4. If both sources give numbers, lead with the database result and add brief context from the report.
5. Max 150 words. Be direct and factual. Source: {best_source} (Page {best_page})

Question: {question}
Answer:"""
        answer_en = call_groq(final_prompt, max_tokens=220)
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
        answer_en = call_groq(final_prompt, max_tokens=200)
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
3. NEVER say the figure is "not explicitly mentioned" or "cannot be determined" if any relevant number appears in the text above.
4. If multiple years or regions appear, list the most relevant ones clearly.
5. Only say nothing relevant was found if that is genuinely true — then say so in one short sentence.
6. Max 150 words. Be factual and direct.

Question: {question}
Answer:"""
        answer_en = call_groq(final_prompt, max_tokens=220)
    else:
        effective_route = "fallback"
        final_prompt = f"""You are the ESS (Ethiopian Statistical Service) AI Assistant.
No matching database result or report text was found for this question.
Reply in clear English, in one or two short sentences. Do not invent statistics or sources.
Question: {question}
Answer:"""
        answer_en = call_groq(final_prompt, max_tokens=150)

    answer = translate_answer(answer_en, target_lang)

    save_cache(question, answer, best_source, best_page, target_lang)
    return {
        "answer": answer,
        "source": best_source,
        "page": best_page,
        "route": effective_route,
        "cached": False,
        "language": target_lang,
        "resolved_question": question if question != original_question else None,
    }

# ═══════════════════════════════════════
# STREAMING VARIANT
# ═══════════════════════════════════════
def get_answer_stream(question: str, chat_history: list = None):
    original_question = question
    log.info(f"Processing Question (stream): '{question[:80]}'")

    target_lang = resolve_language(original_question)

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
        result = {"answer": meta["answer"], "source": "ESS Asset Index", "page": 0, "route": "meta", "cached": False, "language": target_lang}
        save_cache(question, meta["answer"], result["source"], result["page"], target_lang)
        yield ("done", result)
        return

    effective_route = route
    csv_result, pdf_context, best_source, best_page = _gather_sources(question, route)

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
3. NEVER say the figure is not provided/mentioned if any relevant number exists in the sources above.
4. If both sources give numbers, lead with the database result and add brief context from the report.
5. Max 150 words. Be direct and factual. Source: {best_source} (Page {best_page})

Question: {question}
Answer:"""
        max_tok = 320
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
        max_tok = 200
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
3. NEVER say the figure is "not explicitly mentioned" or "cannot be determined" if any relevant number appears in the text above.
4. If multiple years or regions appear, list the most relevant ones clearly.
5. Only say nothing relevant was found if that is genuinely true — then say so in one short sentence.
6. Max 150 words. Be factual and direct.

Question: {question}
Answer:"""
        max_tok = 320

    else:
        effective_route = "fallback"
        final_prompt = f"""You are the ESS (Ethiopian Statistical Service) AI Assistant.
No matching database result or report text was found for this question.
Reply in clear English, in one or two short sentences. Do not invent statistics or sources.
Question: {question}
Answer:"""
        max_tok = 150

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
        "language": target_lang,
        "resolved_question": question if question != original_question else None,
    }
    yield ("done", result)


def _stream_or_translate(prompt: str, max_tokens: int, target_lang: str):
    """Shared streaming helper for get_answer_stream():
    - English: stream tokens live as they're generated (unchanged behavior).
    - Amharic/Oromo: generate the English answer first (not streamed, since
      it isn't the final text), then STREAM the translation — user still
      sees live token-by-token output, it's just the translation pass.
    Yields ("chunk", text) any number of times, then exactly one
    ("done", {"answer": full_final_text})."""
    if target_lang not in ("am", "om"):
        full = ""
        for piece in call_groq_stream(prompt, max_tokens=max_tokens):
            full += piece
            yield ("chunk", piece)
        if not full.strip():
            full = GROQ_UNAVAILABLE_MSG
            yield ("chunk", full)
        yield ("done", {"answer": full})
        return

    answer_en = call_groq(prompt, max_tokens=max_tokens)
    if is_bad_answer(answer_en):
        yield ("chunk", GROQ_UNAVAILABLE_MSG)
        yield ("done", {"answer": GROQ_UNAVAILABLE_MSG})
        return

    lang_name = "Amharic (አማርኛ)" if target_lang == "am" else "Afaan Oromoo"
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
    # Amharic/Oromo script tokenizes far less efficiently than English — the
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
    if not full.strip():
        # Translation stream failed — fall back to the English answer rather
        # than showing nothing.
        full = answer_en
        yield ("chunk", full)
    yield ("done", {"answer": full})


if __name__ == "__main__":
    ensure_cache_table()
    print("Multilingual RAG Engine initialized.")