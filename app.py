import os
import re
import sys
import json
import time
import uuid
import logging
import threading
from functools import wraps
from flask import (Flask, render_template, request,
                    jsonify, session, redirect, url_for,
                    Response, stream_with_context)
from flask_wtf.csrf import CSRFProtect, generate_csrf
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
from dotenv import load_dotenv

# CRITICAL: must run before importing rag — rag.py imports retriever.py,
# which initializes the SentenceTransformer model at import time and needs
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE already set in the environment.
load_dotenv()

from rag import get_answer, get_answer_stream, ensure_cache_table, CSV_DATA, db_cursor, get_budget_status
from pdf_extract import extract_document
from tts_route import tts_bp

# Reused by /admin/add_report below — this is the SAME ChromaDB collection
# (and lock) rag.py's retriever already created for answering questions.
# Deliberately imported as a module (not "from retriever import collection")
# so we always see the live object, never a stale copy.
import retriever
from report_indexer import index_single_pdf
from indexing_helpers import append_metadata_csv

STREAM_META_SEP = "\x00__ESS_META__\x00"
MAX_QUESTION_LENGTH = 1000  # chars — generous for real questions, blocks megabyte-scale abuse
GUEST_QUESTION_LIMIT = int(os.getenv("GUEST_QUESTION_LIMIT", "0"))  # per-browser-session cap for guests; 0 (default) = unlimited

# ── Uploaded-document config ──
UPLOAD_MAX_BYTES = 15 * 1024 * 1024   # 15 MB
UPLOAD_MAX_PAGES = 60                 # pages read from the PDF before truncating
UPLOAD_ALLOWED_EXTENSIONS = {".pdf"}

# ── "Add report" admin config ──
# Previously gated by a single shared secret (ADMIN_REPORT_TOKEN). Replaced
# with real per-user admin roles: a `users.is_admin` column + normal login.
# ADMIN_USERNAMES (comma-separated) is applied once at startup to promote
# specific accounts — set it in .env, e.g. ADMIN_USERNAMES=roba,data_team.
# To promote someone later without touching .env, run directly against the
# DB: UPDATE users SET is_admin = TRUE WHERE username = 'someone';
ADMIN_USERNAMES = [u.strip() for u in os.getenv("ADMIN_USERNAMES", "").split(",") if u.strip()]
REPORT_PDF_FOLDER = os.getenv("REPORT_PDF_FOLDER", "data/pdf")
# Caps the whole request body Werkzeug will buffer, not just this route —
# without it, a huge POST is read into memory in full before our own
# UPLOAD_MAX_BYTES check in upload_document() ever runs.
_MAX_CONTENT_LENGTH = UPLOAD_MAX_BYTES + (1 * 1024 * 1024)  # small margin for multipart overhead

# ═══════════════════════════════════════
# UTF-8 FIX — Defensive Check
# ═══════════════════════════════════════
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_CHAT_TITLE = "New Chat"

def _resolve_secret_key() -> str:
    key = os.getenv("SECRET_KEY")
    if key:
        return key
    if os.getenv("FLASK_ENV") == "production":
        raise RuntimeError(
            "SECRET_KEY is not set. Refusing to start in production without one."
        )
    logging.getLogger(__name__).warning(
        "SECRET_KEY not set — using a random throwaway key for this process."
    )
    return os.urandom(32).hex()

app = Flask(__name__)
app.secret_key = _resolve_secret_key()

_cookie_secure_default = os.getenv("FLASK_ENV") == "production"
_cookie_secure = os.getenv(
    "FLASK_COOKIE_SECURE",
    "true" if _cookie_secure_default else "false"
).lower() == "true"
app.config.update(
    SESSION_COOKIE_HTTPONLY = True,
    # The /widget page runs inside an <iframe> embedded on the ESS website's
    # own domain (see ess-embed-loader.js) — a DIFFERENT origin than this
    # Flask app. Browsers treat every fetch() made from inside that iframe
    # as a cross-site request relative to the host page, so a "Lax" cookie
    # (the old value here) was silently dropped on every widget request:
    # /guest_status, /api/login, /get_sessions, /ask_stream, etc. never
    # carried the session cookie, which is why login + chat history worked
    # fine on the full-page /chat route (same-site) but appeared to reset
    # constantly inside the embedded widget popup (cross-site).
    #
    # "None" is required to allow the cookie on cross-site requests, but
    # browsers only honor SameSite=None when Secure=True (HTTPS). Locally
    # over plain http, None+Secure cookies are rejected outright, so we
    # fall back to Lax there — the widget's cross-site case can't be
    # properly tested over http anyway; use HTTPS (e.g. an ngrok/caddy
    # tunnel) to reproduce and verify the embedded-widget login flow.
    SESSION_COOKIE_SAMESITE = "None" if _cookie_secure else "Lax",
    SESSION_COOKIE_SECURE   = _cookie_secure,
    MAX_CONTENT_LENGTH = _MAX_CONTENT_LENGTH,
)

csrf = CSRFProtect(app)

@app.route("/csrf-token")
def csrf_token():
    return jsonify({"csrf_token": generate_csrf()})

limiter = Limiter(get_remote_address, app=app, default_limits=[])

# /tts (server-side Amharic speech synthesis, see tts_route.py). It doesn't
# touch the DB or session, and already has its own per-IP rate limit, so it
# doesn't need a CSRF token — but it DOES need to be explicitly exempted,
# or CSRFProtect(app) above rejects the plain fetch() the Listen button
# sends (chat.js/widget.js don't attach X-CSRFToken to this call, unlike
# /ask_stream and /upload_document which do).
app.register_blueprint(tts_bp)
csrf.exempt(tts_bp)

# ═══════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════
log = logging.getLogger("app")
log.setLevel(logging.INFO)
if not log.handlers:
    from logging.handlers import RotatingFileHandler
    _fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _fh = RotatingFileHandler("app_log.txt", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    _fh.setFormatter(_fmt)
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    log.addHandler(_fh)
    log.addHandler(_sh)
    log.propagate = False

# ═══════════════════════════════════════
# DATABASE SETUP (Using db_cursor for safe pooling)
# ═══════════════════════════════════════
def init_db():
    try:
        with db_cursor(commit=True) as cur:
            # 1. Users Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            SERIAL PRIMARY KEY,
                    username      VARCHAR(80)  UNIQUE NOT NULL,
                    email         VARCHAR(120) UNIQUE NOT NULL,
                    password_hash VARCHAR(256) NOT NULL,
                    created_at    TIMESTAMP DEFAULT NOW()
                )
            """)

            # 2. Chat Sessions Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id         SERIAL PRIMARY KEY,
                    user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    title      VARCHAR(255) DEFAULT 'New Chat',
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # 3. Chat History Table (With flexible route_used constraints)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id          SERIAL PRIMARY KEY,
                    session_id  INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    sender      VARCHAR(10) NOT NULL CHECK (sender IN ('user', 'bot')),
                    message     TEXT NOT NULL,
                    source_doc  VARCHAR(255),
                    source_page INTEGER,
                    route_used  VARCHAR(20),
                    created_at  TIMESTAMP DEFAULT NOW()
                )
            """)

            # Query Cache Table is created by rag.py's ensure_cache_table()
            # (called below) — single source of truth, avoids schema drift
            # between the two definitions.

            # 4. File-level indexing completion tracker — referenced by
            # index_pdfs.py, /admin/add_report, and the dashboard below.
            # Created here defensively in case it only ever existed via a
            # manual migration on the live DB.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    filename     VARCHAR(255) PRIMARY KEY,
                    chunk_count  INTEGER,
                    ocr_pages    INTEGER DEFAULT 0,
                    indexed_at   TIMESTAMP DEFAULT NOW()
                )
            """)

            # 5. Real admin role — replaces the old shared ADMIN_REPORT_TOKEN.
            cur.execute("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE
            """)

            # 6. Audit log for every /admin/add_report attempt (who, what,
            # whether it worked) — replaces having to grep app_log.txt to
            # find out who added a given report, and doubles as the
            # "failed index jobs" feed for the observability dashboard.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_report_log (
                    id            SERIAL PRIMARY KEY,
                    user_id       INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    username      VARCHAR(80),
                    filename      VARCHAR(255),
                    category      VARCHAR(50),
                    year          VARCHAR(10),
                    chunks_added  INTEGER,
                    tables_found  INTEGER,
                    pages_ocrd    INTEGER,
                    success       BOOLEAN NOT NULL,
                    error_message TEXT,
                    created_at    TIMESTAMP DEFAULT NOW()
                )
            """)

            # 7. Per-question request log — powers the observability
            # dashboard (cache hit rate, average latency by route, recent
            # errors). Deliberately separate from chat_history: that table
            # only stores turns for logged-in users with a saved session,
            # so it undercounts guest traffic entirely.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS request_log (
                    id          SERIAL PRIMARY KEY,
                    route       VARCHAR(20),
                    cached      BOOLEAN NOT NULL DEFAULT FALSE,
                    elapsed_ms  INTEGER,
                    language    VARCHAR(5),
                    error       BOOLEAN NOT NULL DEFAULT FALSE,
                    is_guest    BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at  TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_request_log_created ON request_log(created_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_admin_report_log_created ON admin_report_log(created_at)")

            # Indexes for performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

        log.info("Database initialized successfully with updated schema constraints.")
    except Exception:
        log.exception("DB init error")

# Force initialization at startup
init_db()
ensure_cache_table()

def _promote_configured_admins():
    """Applies ADMIN_USERNAMES on every startup — idempotent (just sets
    is_admin=TRUE again for names already promoted), and picks up a name
    added to .env without needing a manual SQL UPDATE."""
    if not ADMIN_USERNAMES:
        return
    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE users SET is_admin = TRUE WHERE username = ANY(%s) RETURNING username",
                (ADMIN_USERNAMES,)
            )
            promoted = [r[0] for r in cur.fetchall()]
        if promoted:
            log.info(f"Admin role confirmed for: {', '.join(promoted)}")
        missing = set(ADMIN_USERNAMES) - set(promoted)
        if missing:
            log.warning(
                f"ADMIN_USERNAMES lists account(s) that don't exist yet (they'll "
                f"become admin automatically once registered): {', '.join(missing)}"
            )
    except Exception:
        log.exception("Could not apply ADMIN_USERNAMES")

_promote_configured_admins()

# ═══════════════════════════════════════
# AUTH HELPERS
# ═══════════════════════════════════════
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
EMAIL_RE    = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    """Replaces the old shared-token check on /admin/add_report. Requires a
    real logged-in session AND users.is_admin — no separate secret to leak
    or rotate. Not-logged-in -> straight to /login (same as login_required).
    Logged in but not an admin -> a 403 with a clear message instead of a
    silent redirect, so it's obvious why they're blocked."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if not session.get("is_admin"):
            return render_template(
                "admin_add_report.html",
                error="Your account doesn't have admin access. Ask an existing admin to grant it."
            ), 403
        return f(*args, **kwargs)
    return decorated

def optional_login(f):
    """Like login_required but lets guests through. current_user_id() will
    simply be None for them — routes decorated with this must handle a
    missing user_id (no saved history, no session ownership)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated

def current_user_id():
    return session.get("user_id")

def current_username():
    return session.get("username", "guest")

def guest_questions_used() -> int:
    return session.get("guest_qcount", 0)

def guest_questions_remaining():
    """Returns None when guests are unlimited (GUEST_QUESTION_LIMIT <= 0) —
    every caller (the /ask, /ask_stream limit checks, /guest_status, and the
    chat.js/widget.js banners which explicitly check `typeof x === 'number'`
    before showing anything) already treats None as "no limit, don't show
    a counter", so this one change turns the whole quota UI off cleanly."""
    if GUEST_QUESTION_LIMIT <= 0:
        return None
    return max(0, GUEST_QUESTION_LIMIT - guest_questions_used())

def register_guest_question():
    session["guest_qcount"] = guest_questions_used() + 1

def log_request_async(route: str, cached: bool, elapsed_ms: int, language: str, error: bool, is_guest: bool):
    """Fire-and-forget insert into request_log, powering the admin
    dashboard's cache-hit-rate / latency / error-rate panels. Runs in a
    background thread so a slow DB write never adds latency to the answer
    the user is waiting on — same pattern as rag.py's token-budget persist."""
    def _write():
        try:
            with db_cursor(commit=True) as cur:
                cur.execute(
                    """INSERT INTO request_log (route, cached, elapsed_ms, language, error, is_guest)
                       VALUES (%s,%s,%s,%s,%s,%s)""",
                    (route, cached, elapsed_ms, language, error, is_guest)
                )
        except Exception:
            log.exception("Failed to write request_log entry")
    threading.Thread(target=_write, daemon=True).start()

GUEST_HISTORY_TURNS = 3       # kept small — server-memory footprint, not DB
GUEST_HISTORY_TTL_SECONDS = 2 * 3600

# In-memory store keyed by a random per-guest token (NOT the Flask session
# cookie itself). Needed because /ask_stream can't write new cookie data
# once streaming has started — headers are already sent by then — so actual
# history content has to live server-side. Same style as rag.py's existing
# _REWRITE_ROUTE_CACHE. {token: (turns_list, last_seen_timestamp)}
_GUEST_HISTORY_STORE = {}
_GUEST_HISTORY_LOCK = threading.Lock()

def _get_or_create_guest_token() -> str:
    """Safe to call anywhere in the main view function body (before the
    Response is constructed) — this is a normal session write and goes out
    in the Set-Cookie header as usual."""
    token = session.get("guest_token")
    if not token:
        token = uuid.uuid4().hex
        session["guest_token"] = token
    return token

def get_guest_history() -> list:
    token = session.get("guest_token")
    if not token:
        return []
    with _GUEST_HISTORY_LOCK:
        entry = _GUEST_HISTORY_STORE.get(token)
    return list(entry[0]) if entry else []

def append_guest_history(question: str, answer: str):
    """Writes to server memory, not the cookie — safe to call from inside
    the streaming generator, after the response has already started."""
    token = session.get("guest_token")
    if not token:
        return
    with _GUEST_HISTORY_LOCK:
        existing = _GUEST_HISTORY_STORE.get(token)
        turns = list(existing[0]) if existing else []
        turns.append({"user": question[:300], "assistant": (answer or "")[:300]})
        turns = turns[-GUEST_HISTORY_TURNS:]
        _GUEST_HISTORY_STORE[token] = (turns, time.time())
        if len(_GUEST_HISTORY_STORE) > 5000:
            cutoff = time.time() - GUEST_HISTORY_TTL_SECONDS
            for k in [k for k, (_, ts) in _GUEST_HISTORY_STORE.items() if ts < cutoff]:
                del _GUEST_HISTORY_STORE[k]

# ═══════════════════════════════════════
# UPLOADED DOCUMENT — per-browser-session, in-memory (not the ESS corpus)
# ═══════════════════════════════════════
# Same reasoning as _GUEST_HISTORY_STORE above: the extracted text can be up
# to UPLOAD_MAX_CHARS-worth of characters, too big/volatile for the session
# cookie, and needs to survive across the streaming response. Keyed by a
# random per-browser token (works for guests AND logged-in users — unlike
# guest_token, this is set regardless of login state).
UPLOAD_TTL_SECONDS = 2 * 3600
_UPLOAD_STORE = {}
_UPLOAD_LOCK = threading.Lock()


def _get_or_create_upload_token() -> str:
    token = session.get("upload_token")
    if not token:
        token = uuid.uuid4().hex
        session["upload_token"] = token
    return token


def get_uploaded_document() -> dict:
    """Returns {"filename": ..., "text": ...} if this browser session has an
    active uploaded document, else None."""
    token = session.get("upload_token")
    if not token:
        return None
    with _UPLOAD_LOCK:
        entry = _UPLOAD_STORE.get(token)
        if not entry:
            return None
        data, last_seen = entry
        if time.time() - last_seen > UPLOAD_TTL_SECONDS:
            del _UPLOAD_STORE[token]
            return None
    return data


def set_uploaded_document(filename: str, text: str):
    token = _get_or_create_upload_token()
    with _UPLOAD_LOCK:
        _UPLOAD_STORE[token] = ({"filename": filename, "text": text}, time.time())
        if len(_UPLOAD_STORE) > 5000:
            cutoff = time.time() - UPLOAD_TTL_SECONDS
            for k in [k for k, (_, ts) in _UPLOAD_STORE.items() if ts < cutoff]:
                del _UPLOAD_STORE[k]


def clear_uploaded_document():
    token = session.get("upload_token")
    if not token:
        return
    with _UPLOAD_LOCK:
        _UPLOAD_STORE.pop(token, None)


def _owns_session(user_id: int, session_id: int) -> bool:
    try:
        with db_cursor() as cur:
            cur.execute(
                "SELECT id FROM chat_sessions WHERE id=%s AND user_id=%s",
                (session_id, user_id)
            )
            return cur.fetchone() is not None
    except Exception:
        log.exception("Session ownership check failed")
        return False

def _fetch_history_for_rewrite(session_id: int) -> list:
    """Retrieves previous QA turns to feed context into query_rewrite."""
    try:
        with db_cursor() as cur:
            cur.execute(
                """SELECT sender, message FROM (
                       SELECT sender, message, created_at FROM chat_history
                       WHERE session_id = %s
                       ORDER BY created_at DESC LIMIT 10
                   ) recent
                   ORDER BY created_at ASC""",
                (session_id,)
            )
            rows = cur.fetchall()
        
        chat_history = []
        temp_turn = {}
        for sender, message in rows:
            if sender == "user":
                if "user" in temp_turn:
                    chat_history.append(temp_turn)
                    temp_turn = {}
                temp_turn["user"] = message
            elif sender == "bot":
                temp_turn["assistant"] = message
                chat_history.append(temp_turn)
                temp_turn = {}
        if temp_turn:
            chat_history.append(temp_turn)
        return chat_history
    except Exception:
        log.exception("Failed loading historical turns")
        return []

# ═══════════════════════════════════════
# ROUTES — AUTH
# ═══════════════════════════════════════

@app.route("/")
def index():
    return redirect(url_for("chat"))

@app.route("/register", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()

    if not username or not email or not password:
        return render_template("register.html", error="All fields are required.")

    if not USERNAME_RE.match(username):
        return render_template(
            "register.html",
            error="Username must be 3-30 characters: letters, numbers, underscore only."
        )

    if not EMAIL_RE.match(email):
        return render_template("register.html", error="Please enter a valid email address.")

    if len(password) < 8:
        return render_template(
            "register.html",
            error="Password must be at least 8 characters."
        )

    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                "SELECT id FROM users WHERE username=%s OR email=%s",
                (username, email)
            )
            if cur.fetchone():
                return render_template(
                    "register.html",
                    error="Username or email already exists."
                )

            password_hash = generate_password_hash(password)
            is_admin = username in ADMIN_USERNAMES
            cur.execute(
                """INSERT INTO users (username, email, password_hash, is_admin)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (username, email, password_hash, is_admin)
            )
            user_id = cur.fetchone()[0]

        session["user_id"]  = user_id
        session["username"] = username
        session["is_admin"] = is_admin
        log.info(f"New user registered: {username}" + (" (admin)" if is_admin else ""))
        return redirect(url_for("chat"))

    except Exception:
        log.exception("Register error")
        return render_template("register.html", error="Registration failed. Please try again.")

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not username or not password:
        return render_template("login.html", error="Username and password are required.")

    try:
        with db_cursor() as cur:
            cur.execute(
                "SELECT id, username, password_hash, is_admin FROM users WHERE username=%s",
                (username,)
            )
            user = cur.fetchone()

        if not user or not check_password_hash(user[2], password):
            return render_template("login.html", error="Invalid username or password.")

        session["user_id"]  = user[0]
        session["username"] = user[1]
        session["is_admin"] = bool(user[3])
        log.info(f"User logged in: {username}")
        return redirect(url_for("chat"))

    except Exception:
        log.exception("Login error")
        return render_template("login.html", error="Login failed. Please try again.")

@app.route("/logout")
def logout():
    username = session.get("username", "unknown")
    session.clear()
    log.info(f"User logged out: {username}")
    return redirect(url_for("login"))

# ═══════════════════════════════════════
# JSON AUTH — used ONLY by the popup widget (static/js/widget.js). The
# regular /login, /register, /logout above render full HTML pages and
# redirect on success, which is why the popup previously had to leave
# the iframe (target="_blank") to sign in at all. These mirror the exact
# same validation/session logic but respond with JSON and never redirect,
# so the widget can authenticate in place and keep the popup open.
# ═══════════════════════════════════════
@app.route("/api/login", methods=["POST"])
@limiter.limit("10 per minute")
def api_login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    try:
        with db_cursor() as cur:
            cur.execute(
                "SELECT id, username, password_hash, is_admin FROM users WHERE username=%s",
                (username,)
            )
            user = cur.fetchone()

        if not user or not check_password_hash(user[2], password):
            return jsonify({"error": "Invalid username or password."}), 401

        session["user_id"]  = user[0]
        session["username"] = user[1]
        session["is_admin"] = bool(user[3])
        log.info(f"User logged in via widget: {username}")
        return jsonify({"success": True, "username": user[1]})

    except Exception:
        log.exception("Widget login error")
        return jsonify({"error": "Login failed. Please try again."}), 500


@app.route("/api/register", methods=["POST"])
@limiter.limit("10 per minute")
def api_register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email    = (data.get("email") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not email or not password:
        return jsonify({"error": "All fields are required."}), 400
    if not USERNAME_RE.match(username):
        return jsonify({"error": "Username must be 3-30 characters: letters, numbers, underscore only."}), 400
    if not EMAIL_RE.match(email):
        return jsonify({"error": "Please enter a valid email address."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                "SELECT id FROM users WHERE username=%s OR email=%s",
                (username, email)
            )
            if cur.fetchone():
                return jsonify({"error": "Username or email already exists."}), 409

            password_hash = generate_password_hash(password)
            is_admin = username in ADMIN_USERNAMES
            cur.execute(
                """INSERT INTO users (username, email, password_hash, is_admin)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (username, email, password_hash, is_admin)
            )
            user_id = cur.fetchone()[0]

        session["user_id"]  = user_id
        session["username"] = username
        session["is_admin"] = is_admin
        log.info(f"New user registered via widget: {username}")
        return jsonify({"success": True, "username": username})

    except Exception:
        log.exception("Widget register error")
        return jsonify({"error": "Registration failed. Please try again."}), 500


@app.route("/api/logout", methods=["POST"])
def api_logout():
    username = session.get("username", "unknown")
    session.clear()
    log.info(f"User logged out via widget: {username}")
    return jsonify({"success": True})

# ═══════════════════════════════════════
# ROUTES — CHAT
# ═══════════════════════════════════════

# ═══════════════════════════════════════
# ROUTES — EMBEDDABLE POPUP WIDGET
# Served for the official ESS website's iframe (see
# static/ess-embed-loader.js). Deliberately guest-only, no login/sidebar —
# a lighter page than /chat so it loads fast inside a small iframe panel.
# Reuses /ask_stream, /csrf-token, /suggestions, /guest_status unchanged —
# same-origin with this route, so no CORS/cross-site-cookie setup needed.
# ═══════════════════════════════════════
# Comma-separated list of origins allowed to iframe this route, e.g.
# "https://www.statsethiopia.gov.et,https://statsethiopia.gov.et". Leave
# unset only for local testing — an unset/empty value falls back to
# 'self' only, meaning the widget won't embed anywhere outside this app
# until the real ESS domain is added here.
WIDGET_EMBED_ORIGINS = [o.strip() for o in os.getenv("WIDGET_EMBED_ORIGINS", "").split(",") if o.strip()]

@app.route("/widget")
def widget():
    resp = app.make_response(render_template("widget.html"))
    frame_ancestors = "'self' " + " ".join(WIDGET_EMBED_ORIGINS) if WIDGET_EMBED_ORIGINS else "'self'"
    resp.headers["Content-Security-Policy"] = f"frame-ancestors {frame_ancestors};"
    return resp

@app.route("/test")
def test_page():
    """Local-only page for testing the popup widget (ess-embed-loader.js ->
    /widget iframe) same-origin with Flask. Opening templates/test.html
    directly as a file:// path instead of through this route makes Chrome
    block the loader script entirely ("file: URLs are treated as unique
    security origins") — this route exists so local testing matches how
    the widget will actually be embedded (same-origin script load), without
    needing the real ESS website to test against.
    Not gated behind FLASK_ENV since it renders a static local test page
    with no data access — safe to leave enabled, but remove/comment out
    before deploying if you'd rather not expose it publicly."""
    return render_template("test.html")

@app.route("/chat")
@optional_login
def chat():
    user_id  = current_user_id()
    username = current_username()

    if user_id is None:
        # Guest — no account, so no saved sessions to show.
        return render_template("chat.html", user=None, sessions=[], is_guest=True, is_admin=False)

    try:
        with db_cursor() as cur:
            cur.execute(
                """SELECT id, title, created_at
                   FROM chat_sessions
                   WHERE user_id = %s
                   ORDER BY created_at DESC""",
                (user_id,)
            )
            rows = cur.fetchall()
            
        sessions_list = [{
            "id":         r[0],
            "title":      r[1],
            "updated_at": r[2]
        } for r in rows]
        
        return render_template("chat.html", user=username, sessions=sessions_list, is_guest=False, is_admin=session.get("is_admin", False))

    except Exception:
        log.exception("Chat page error")
        return render_template("chat.html", user=username, sessions=[], is_guest=False, is_admin=session.get("is_admin", False))

@app.route("/new_session", methods=["POST"])
@login_required
def new_session():
    user_id = current_user_id()
    data    = request.get_json(silent=True) or {}
    title   = (data.get("title") or DEFAULT_CHAT_TITLE).strip()[:255]

    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO chat_sessions (user_id, title)
                   VALUES (%s, %s) RETURNING id""",
                (user_id, title)
            )
            session_id = cur.fetchone()[0]
        return jsonify({"success": True, "session_id": session_id})
    except Exception:
        log.exception("New session error")
        return jsonify({"success": False, "message": "Could not create session"}), 500

@app.route("/get_history/<int:session_id>")
@login_required
def get_history(session_id):
    user_id = current_user_id()

    if not _owns_session(user_id, session_id):
        return jsonify({"error": "Not found"}), 404

    limit  = min(int(request.args.get("limit", 100)), 500)
    offset = max(int(request.args.get("offset", 0)), 0)

    try:
        with db_cursor() as cur:
            cur.execute(
                """SELECT sender, message, source_doc, source_page, route_used, created_at
                   FROM chat_history
                   WHERE session_id = %s
                   ORDER BY created_at ASC
                   LIMIT %s OFFSET %s""",
                (session_id, limit, offset)
            )
            messages = cur.fetchall()

        history = [{
            "sender":      m[0],
            "message":     m[1],
            "source_doc":  m[2],
            "source_page": m[3],
            "route_used":  m[4],
            "created_at":  m[5].strftime("%H:%M") if m[5] else ""
        } for m in messages]

        return jsonify({"history": history})

    except Exception:
        log.exception("History error")
        return jsonify({"error": "Failed to load history"}), 500

@app.route("/ask", methods=["POST"])
@optional_login
@limiter.limit("10 per minute")
def ask():
    _t0 = time.time()
    user_id    = current_user_id()
    data       = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    question   = (data.get("question") or "").strip()
    session_id = data.get("session_id")
    force_lang = data.get("ui_lang")
    if force_lang not in ("en", "am"):
        force_lang = None

    if not question:
        return jsonify({"error": "Empty question"}), 400

    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({"error": f"Question too long — please keep it under {MAX_QUESTION_LENGTH} characters."}), 400

    if user_id is None:
        # Guests have no saved sessions — never persist/own one.
        session_id = None
        _get_or_create_guest_token()
        remaining = guest_questions_remaining()
        if remaining is not None and remaining <= 0:
            return jsonify({
                "error": f"You've used your {GUEST_QUESTION_LIMIT} free guest questions. Please sign in or create a free account to keep asking.",
                "code": "guest_limit_reached"
            }), 403
    elif session_id is not None and not _owns_session(user_id, session_id):
        return jsonify({"error": "Invalid session"}), 403

    log.info(f"User {current_username()} asked: '{question[:60]}'")

    chat_history = _fetch_history_for_rewrite(session_id) if session_id else (
        get_guest_history() if user_id is None else None
    )

    uploaded = get_uploaded_document()

    try:
        result = get_answer(
            question, chat_history=chat_history,
            uploaded_context=uploaded["text"] if uploaded else None,
            uploaded_filename=uploaded["filename"] if uploaded else None,
            force_lang=force_lang,
        ) or {}
    except Exception:
        log.exception("RAG pipeline error")
        log_request_async("ask", cached=False, elapsed_ms=int((time.time() - _t0) * 1000),
                           language=force_lang or "en", error=True, is_guest=user_id is None)
        return jsonify({"error": "Failed to get answer. Please try again."}), 500

    if user_id is None:
        register_guest_question()

    answer = result.get("answer") or "Sorry, I couldn't generate an answer."
    source = result.get("source", "Unknown")
    page   = result.get("page", 0)
    route  = result.get("route", "unknown")
    cached = result.get("cached", False)

    log_request_async(route, cached=cached, elapsed_ms=int((time.time() - _t0) * 1000),
                       language=force_lang or "en", error=False, is_guest=user_id is None)

    if user_id is None:
        append_guest_history(question, answer)

    if session_id:
        try:
            with db_cursor(commit=True) as cur:
                cur.execute(
                    """INSERT INTO chat_history (session_id, sender, message)
                       VALUES (%s, %s, %s)""",
                    (session_id, "user", question)
                )
                cur.execute(
                    """INSERT INTO chat_history (session_id, sender, message, source_doc, source_page, route_used)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (session_id, "bot", answer, source, page, route)
                )
                cur.execute(
                    """UPDATE chat_sessions
                       SET title = %s
                       WHERE id = %s AND user_id = %s AND title = %s""",
                    (question[:50], session_id, user_id, DEFAULT_CHAT_TITLE)
                )
        except Exception:
            log.exception("Failed to save chat history")

    return jsonify({
        "answer": answer,
        "source": source,
        "page":   page,
        "route":  route,
        "cached": cached,
        "elapsed": round(time.time() - _t0, 1),
        "guest_remaining": guest_questions_remaining() if user_id is None else None
    })

@app.route("/ask_stream", methods=["POST"])
@optional_login
@limiter.limit("10 per minute")
def ask_stream():
    _t0        = time.time()
    user_id    = current_user_id()
    data       = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    question   = (data.get("question") or "").strip()
    session_id = data.get("session_id")
    force_lang = data.get("ui_lang")
    if force_lang not in ("en", "am"):
        force_lang = None

    if not question:
        return jsonify({"error": "Empty question"}), 400

    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({"error": f"Question too long — please keep it under {MAX_QUESTION_LENGTH} characters."}), 400

    if user_id is None:
        # Guests have no saved sessions — never persist/own one.
        session_id = None
        _get_or_create_guest_token()
        remaining = guest_questions_remaining()
        if remaining is not None and remaining <= 0:
            return jsonify({
                "error": f"You've used your {GUEST_QUESTION_LIMIT} free guest questions. Please sign in or create a free account to keep asking.",
                "code": "guest_limit_reached"
            }), 403
        # Increment HERE, before the streaming Response is constructed — not
        # inside generate(). Once streaming starts, headers (including
        # Set-Cookie) are already sent, so a session write inside generate()
        # never reaches the browser and the counter silently never advances.
        register_guest_question()
    elif session_id is not None and not _owns_session(user_id, session_id):
        return jsonify({"error": "Invalid session"}), 403

    log.info(f"User {current_username()} asked (stream): '{question[:60]}'")

    chat_history = _fetch_history_for_rewrite(session_id) if session_id else (
        get_guest_history() if user_id is None else None
    )

    uploaded = get_uploaded_document()
    uploaded_text = uploaded["text"] if uploaded else None
    uploaded_filename = uploaded["filename"] if uploaded else None

    def generate():
        full_answer = ""
        meta = {}
        try:
            for kind, payload in get_answer_stream(
                question, chat_history=chat_history,
                uploaded_context=uploaded_text, uploaded_filename=uploaded_filename,
                force_lang=force_lang,
            ):
                if kind == "chunk":
                    full_answer += payload
                    yield payload
                elif kind == "done":
                    meta = payload
        except Exception:
            log.exception("RAG streaming pipeline error")
            if not full_answer:
                fallback = "Sorry, I couldn't generate an answer."
                full_answer = fallback
                yield fallback
            meta = {"source": "Unknown", "page": 0, "route": "error"}
            log_request_async("ask_stream", cached=False, elapsed_ms=int((time.time() - _t0) * 1000),
                               language=force_lang or "en", error=True, is_guest=user_id is None)

        answer = meta.get("answer") or full_answer
        source = meta.get("source", "Unknown")
        page   = meta.get("page", 0)
        route  = meta.get("route", "unknown")

        if route != "error":
            log_request_async(route, cached=(route == "cache"), elapsed_ms=int((time.time() - _t0) * 1000),
                               language=force_lang or "en", error=False, is_guest=user_id is None)

        if user_id is None:
            append_guest_history(question, answer)

        if session_id:
            try:
                with db_cursor(commit=True) as cur:
                    cur.execute(
                        """INSERT INTO chat_history (session_id, sender, message)
                           VALUES (%s, %s, %s)""",
                        (session_id, "user", question)
                    )
                    cur.execute(
                        """INSERT INTO chat_history (session_id, sender, message, source_doc, source_page, route_used)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (session_id, "bot", answer, source, page, route)
                    )
                    cur.execute(
                        """UPDATE chat_sessions
                           SET title = %s
                           WHERE id = %s AND user_id = %s AND title = %s""",
                        (question[:50], session_id, user_id, DEFAULT_CHAT_TITLE)
                    )
            except Exception:
                log.exception("Failed to save chat history (stream)")

        yield STREAM_META_SEP + json.dumps({
            "source": source,
            "page": page,
            "route": route,
            "elapsed": round(time.time() - _t0, 1),
            "guest_remaining": guest_questions_remaining() if user_id is None else None
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    )

@app.route("/get_sessions")
@login_required
def get_sessions():
    user_id = current_user_id()
    try:
        with db_cursor() as cur:
            cur.execute(
                """SELECT id, title, created_at
                   FROM chat_sessions
                   WHERE user_id = %s
                   ORDER BY created_at DESC""",
                (user_id,)
            )
            rows = cur.fetchall()

        sessions_list = [{
            "id":         r[0],
            "title":      r[1],
            "created_at": r[2].strftime("%b %d") if r[2] else ""
        } for r in rows]

        return jsonify({"sessions": sessions_list})
    except Exception:
        log.exception("Get sessions error")
        return jsonify({"sessions": []}), 500

@app.route("/delete_session/<int:session_id>", methods=["DELETE"])
@login_required
def delete_session(session_id):
    user_id = current_user_id()
    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                "DELETE FROM chat_sessions WHERE id=%s AND user_id=%s",
                (session_id, user_id)
            )
            deleted = cur.rowcount
        if deleted == 0:
            return jsonify({"success": False, "message": "Not found"}), 404
        return jsonify({"success": True})
    except Exception:
        log.exception("Delete session error")
        return jsonify({"success": False}), 500

@app.route("/admin/add_report", methods=["GET", "POST"])
@admin_required
@limiter.limit("10 per hour")
def admin_add_report():
    """Adds a PDF straight into the permanent ESS corpus from the browser —
    replaces "copy the file onto the server and run `python index_pdfs.py`
    by hand". Runs the exact same extraction/chunking pipeline as that
    script (via report_indexer.py) and writes into the SAME live ChromaDB
    collection the chatbot already answers from, so the report is
    answerable immediately — no restart, no separate reindex step.

    Gated by @admin_required (real per-user role, see users.is_admin) —
    replaces the old shared ADMIN_REPORT_TOKEN. POST responds with JSON so
    the page's own JS can show real upload progress + an indexing spinner
    instead of a blocking full-page reload."""
    if request.method == "GET":
        return render_template("admin_add_report.html")

    def _log_attempt(filename, category, year, result, success, error_message=None):
        try:
            with db_cursor(commit=True) as cur:
                cur.execute(
                    """INSERT INTO admin_report_log
                       (user_id, username, filename, category, year, chunks_added,
                        tables_found, pages_ocrd, success, error_message)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        current_user_id(), current_username(), filename, category, year,
                        (result or {}).get("chunks_added"), (result or {}).get("tables_found"),
                        (result or {}).get("pages_ocrd"), success, error_message,
                    ),
                )
        except Exception:
            log.exception("Failed to write admin_report_log entry")

    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename = os.path.basename(f.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    category = (request.form.get("category") or "").strip() or None
    year = (request.form.get("year") or "").strip() or None

    file_bytes = f.read(UPLOAD_MAX_BYTES + 1)
    if len(file_bytes) > UPLOAD_MAX_BYTES:
        return jsonify({"error": f"File too large — max {UPLOAD_MAX_BYTES // (1024 * 1024)} MB."}), 400
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        os.makedirs(REPORT_PDF_FOLDER, exist_ok=True)
        dest_path = os.path.join(REPORT_PDF_FOLDER, filename)
        with open(dest_path, "wb") as out:
            out.write(file_bytes)

        result = index_single_pdf(
            file_bytes, filename,
            collection=retriever.collection,
            collection_lock=retriever._collection_lock,
            category=category, year=year,
        )

        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, chunk_count, indexed_at, ocr_pages)
                VALUES (%s, %s, NOW(), %s)
                ON CONFLICT (filename) DO UPDATE
                    SET chunk_count = EXCLUDED.chunk_count,
                        indexed_at  = NOW(),
                        ocr_pages   = EXCLUDED.ocr_pages;
                """,
                (filename, result["chunks_added"], result["pages_ocrd"]),
            )

        append_metadata_csv(REPORT_PDF_FOLDER, filename, result["category"], result["year"])
        _log_attempt(filename, result["category"], result["year"], result, success=True)

        log.info(
            f"Admin '{current_username()}' added report '{filename}' — {result['chunks_added']} chunks "
            f"({result['tables_found']} tables, {result['pages_ocrd']} OCR'd pages), "
            f"category={result['category']}, year={result['year']}"
        )

        warning = (
            " No text or tables could be extracted from this PDF (even with OCR) — "
            "it won't be answerable until that's fixed."
        ) if result["chunks_added"] == 0 else ""

        return jsonify({
            "success": True,
            "message": (
                f"'{filename}' added — {result['chunks_added']} chunks indexed "
                f"({result['tables_found']} tables, {result['pages_ocrd']} of "
                f"{result['pages_total']} pages needed OCR), category="
                f"{result['category']}, year={result['year']}. "
                f"Answerable immediately — no restart needed.{warning}"
            ),
        })

    except Exception as e:
        log.exception(f"Admin add_report failed for '{filename}'")
        _log_attempt(filename, category, year, None, success=False, error_message=str(e)[:500])
        return jsonify({"error": "Something went wrong indexing that PDF — check server logs."}), 500


def _gather_dashboard_stats():
    """Shared stats-gathering used by BOTH the server-rendered dashboard
    page (first paint) and the /admin/dashboard/data JSON endpoint (used
    by the page's own auto-refresh polling). Keeping this in one function
    means the two can never drift apart on what a number means."""
    stats = {}
    try:
        with db_cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') AS total_24h,
                    COUNT(*) FILTER (WHERE cached AND created_at > NOW() - INTERVAL '24 hours') AS cached_24h,
                    COUNT(*) FILTER (WHERE error AND created_at > NOW() - INTERVAL '24 hours') AS errors_24h,
                    AVG(elapsed_ms) FILTER (WHERE NOT cached AND created_at > NOW() - INTERVAL '24 hours') AS avg_latency_ms
                FROM request_log
            """)
            row = cur.fetchone() or (0, 0, 0, None)
            total_24h, cached_24h, errors_24h, avg_latency_ms = row
            stats["total_24h"] = total_24h or 0
            stats["cache_hit_rate"] = round(100 * (cached_24h or 0) / total_24h, 1) if total_24h else 0.0
            stats["errors_24h"] = errors_24h or 0
            stats["avg_latency_ms"] = round(avg_latency_ms) if avg_latency_ms else None

            # Hourly buckets for the last 24h — powers the traffic-over-time
            # chart on the dashboard. Zero-filled so the chart always has 24
            # points even on hours with no traffic at all.
            cur.execute("""
                SELECT date_trunc('hour', created_at) AS hr, COUNT(*)
                FROM request_log
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY hr ORDER BY hr
            """)
            hourly_map = {r[0].strftime("%H:00"): r[1] for r in cur.fetchall()}
            stats["hourly_labels"] = list(hourly_map.keys())
            stats["hourly_counts"] = list(hourly_map.values())

            cur.execute("""
                SELECT route, COUNT(*), AVG(elapsed_ms) FILTER (WHERE NOT cached)
                FROM request_log
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY route ORDER BY COUNT(*) DESC
            """)
            stats["by_route"] = [
                {"route": r[0] or "unknown", "count": r[1], "avg_latency_ms": round(r[2]) if r[2] else None}
                for r in cur.fetchall()
            ]

            cur.execute("""
                SELECT filename, username, category, year, chunks_added, success, error_message, created_at
                FROM admin_report_log ORDER BY created_at DESC LIMIT 15
            """)
            stats["recent_reports"] = [
                {"filename": r[0], "username": r[1] or "unknown", "category": r[2], "year": r[3],
                 "chunks_added": r[4], "success": r[5], "error_message": r[6],
                 "created_at": r[7].strftime("%b %d, %H:%M") if r[7] else ""}
                for r in cur.fetchall()
            ]

            cur.execute("SELECT COUNT(*), COALESCE(SUM(ocr_pages), 0) FROM documents")
            doc_count, ocr_pages_total = cur.fetchone()
            stats["documents_indexed"] = doc_count or 0
            stats["ocr_pages_total"] = ocr_pages_total or 0

    except Exception:
        log.exception("Dashboard query failed")
        stats.setdefault("total_24h", 0)
        stats.setdefault("cache_hit_rate", 0.0)
        stats.setdefault("errors_24h", 0)
        stats.setdefault("avg_latency_ms", None)
        stats.setdefault("hourly_labels", [])
        stats.setdefault("hourly_counts", [])
        stats.setdefault("by_route", [])
        stats.setdefault("recent_reports", [])
        stats.setdefault("documents_indexed", 0)
        stats.setdefault("ocr_pages_total", 0)

    try:
        stats["corpus_by_category"] = retriever.get_corpus_summary()
    except Exception:
        log.exception("Corpus summary failed")
        stats["corpus_by_category"] = {}

    stats["groq_budget"] = get_budget_status()
    return stats


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """Lightweight observability page: cache hit rate, average latency by
    route, recent errors, indexed corpus breakdown, and the report-add
    audit trail (including failures) — everything that previously required
    grepping app_log.txt / rag_log.txt / retriever_log.txt by hand.

    First paint is server-rendered (works even with JS disabled); the page
    then polls /admin/dashboard/data on its own to stay live without a
    full reload."""
    stats = _gather_dashboard_stats()
    return render_template("admin_dashboard.html", stats=stats)


@app.route("/admin/dashboard/data")
@admin_required
def admin_dashboard_data():
    """JSON version of the same stats, polled every few seconds by
    admin_dashboard.html's own JS so the numbers/charts/log update live
    without the admin having to refresh the page."""
    return jsonify(_gather_dashboard_stats())


@app.route("/upload_document", methods=["POST"])
@optional_login
@limiter.limit("5 per minute")
def upload_document():
    """Accepts a single PDF, extracts its full text (text layer -> OCR
    fallback -> table extraction, via pdf_extract.extract_document — the
    same pipeline the bulk indexer uses), and holds it server-side for this
    browser session. Subsequent /ask and /ask_stream calls answer from this
    document instead of the ESS corpus until it's cleared."""
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify({"error": "No file provided"}), 400

    filename = f.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in UPLOAD_ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only PDF files are supported right now."}), 400

    file_bytes = f.read(UPLOAD_MAX_BYTES + 1)
    if len(file_bytes) > UPLOAD_MAX_BYTES:
        return jsonify({"error": f"File too large — max {UPLOAD_MAX_BYTES // (1024*1024)} MB."}), 400
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        result = extract_document(file_bytes, max_pages=UPLOAD_MAX_PAGES)
    except Exception:
        log.exception(f"Upload extraction failed for '{filename}'")
        return jsonify({"error": "Could not read that PDF — it may be corrupted or password-protected."}), 400

    if not result["text"]:
        return jsonify({
            "error": "No readable text found in this PDF, even with OCR. "
                     "If it's a scanned document, make sure it's clear/high-resolution."
        }), 400

    set_uploaded_document(filename, result["text"])
    log.info(
        f"User {current_username()} uploaded '{filename}' — "
        f"{result['pages_total']} pages ({result['pages_ocrd']} via OCR, "
        f"{result['pages_empty']} unreadable), {result['tables_found']} tables, "
        f"truncated={result['truncated']}"
    )

    return jsonify({
        "filename": filename,
        "pages_total": result["pages_total"],
        "pages_ocrd": result["pages_ocrd"], 
        "pages_empty": result["pages_empty"],
        "tables_found": result["tables_found"],
        "ocr_available": result["ocr_available"],
        "truncated": result["truncated"],
    })


@app.route("/clear_upload", methods=["POST"])
@optional_login
def clear_upload():
    clear_uploaded_document()
    return jsonify({"cleared": True})


@app.route("/upload_status")
@optional_login
def upload_status():
    """Lets the frontend restore the 'attached file' chip after a page
    reload, since the extracted text lives server-side, not in the DOM."""
    uploaded = get_uploaded_document()
    return jsonify({"filename": uploaded["filename"] if uploaded else None})


SUGGESTIONS_BY_LANG = {
    "en": [
        "What was the inflation rate in EFY 2018?",
        "Tell me about livestock production in Ethiopia",
        "What is the land utilization in Amhara region?",
        "How many households are in the survey?",
        "What does the 2022 statistical report say?",
        "Tell me about the demographic health survey",
        "What is the population of Ethiopia?",
        "Describe the agricultural survey findings"
    ],
    "am": [
        "በ2018 ዓ.ም (EFY) የኢትዮጵያ የዋጋ ግሽበት መጠን ስንት ነበር?",
        "ስለ ኢትዮጵያ የከብት እርባታ ንገረኝ",
        "በአማራ ክልል የመሬት አጠቃቀም ምንድን ነው?",
        "በጥናቱ ምን ያህል ቤተሰቦች ተካተዋል?",
        "የ2022 ስታቲስቲክስ ሪፖርት ምን ይላል?",
        "ስለ ስነ-ሕዝብ ጤና ጥናት ንገረኝ",
        "የኢትዮጵያ ህዝብ ብዛት ስንት ነው?",
        "የግብርና ጥናት ግኝቶችን ግለጽ"
    ],
}

@app.route("/suggestions")
def suggestions():
    lang = request.args.get("lang", "en")
    questions = SUGGESTIONS_BY_LANG.get(lang, SUGGESTIONS_BY_LANG["en"])
    return jsonify({"suggestions": questions})

@app.route("/guest_status")
def guest_status():
    return jsonify({
        "is_guest": current_user_id() is None,
        "remaining": guest_questions_remaining() if current_user_id() is None else None,
        "limit": GUEST_QUESTION_LIMIT,
        "username": session.get("username") if current_user_id() is not None else None
    })

@app.route("/health")
def health():
    return jsonify({
        "status":       "running",
        "csv_files":    len(CSV_DATA),
        "user":         session.get("username", "not logged in"),
        "groq_budget":  get_budget_status()
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Page not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    log.info("Starting ESS Chatbot...")
    log.info(f"ESS Chatbot running at http://localhost:{port}")
    try:
        # waitress: a production-grade WSGI server that runs on Windows
        # (unlike gunicorn, which needs Linux). Use this everywhere except
        # the final Linux deployment, where gunicorn+nginx take over.
        from waitress import serve
        serve(app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        log.warning("waitress not installed (pip install waitress) — "
                    "falling back to Flask's dev server. Do not use this for real traffic.")
        app.run(debug=False, host="0.0.0.0", port=port)