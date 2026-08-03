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

STREAM_META_SEP = "\x00__ESS_META__\x00"
MAX_QUESTION_LENGTH = 1000  # chars — generous for real questions, blocks megabyte-scale abuse
GUEST_QUESTION_LIMIT = int(os.getenv("GUEST_QUESTION_LIMIT", "5"))  # per-browser-session cap for guests

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
app.config.update(
    SESSION_COOKIE_HTTPONLY = True,
    SESSION_COOKIE_SAMESITE = "Lax",
    SESSION_COOKIE_SECURE   = os.getenv(
        "FLASK_COOKIE_SECURE",
        "true" if _cookie_secure_default else "false"
    ).lower() == "true",
)

csrf = CSRFProtect(app)

@app.route("/csrf-token")
def csrf_token():
    return jsonify({"csrf_token": generate_csrf()})

limiter = Limiter(get_remote_address, app=app, default_limits=[])

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

def guest_questions_remaining() -> int:
    return max(0, GUEST_QUESTION_LIMIT - guest_questions_used())

def register_guest_question():
    session["guest_qcount"] = guest_questions_used() + 1

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
            cur.execute(
                """INSERT INTO users (username, email, password_hash)
                   VALUES (%s, %s, %s) RETURNING id""",
                (username, email, password_hash)
            )
            user_id = cur.fetchone()[0]

        session["user_id"]  = user_id
        session["username"] = username
        log.info(f"New user registered: {username}")
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
                "SELECT id, username, password_hash FROM users WHERE username=%s",
                (username,)
            )
            user = cur.fetchone()

        if not user or not check_password_hash(user[2], password):
            return render_template("login.html", error="Invalid username or password.")

        session["user_id"]  = user[0]
        session["username"] = user[1]
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
# ROUTES — CHAT
# ═══════════════════════════════════════

@app.route("/chat")
@optional_login
def chat():
    user_id  = current_user_id()
    username = current_username()

    if user_id is None:
        # Guest — no account, so no saved sessions to show.
        return render_template("chat.html", user=None, sessions=[], is_guest=True)

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
        
        return render_template("chat.html", user=username, sessions=sessions_list, is_guest=False)

    except Exception:
        log.exception("Chat page error")
        return render_template("chat.html", user=username, sessions=[], is_guest=False)

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

    if not question:
        return jsonify({"error": "Empty question"}), 400

    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({"error": f"Question too long — please keep it under {MAX_QUESTION_LENGTH} characters."}), 400

    if user_id is None:
        # Guests have no saved sessions — never persist/own one.
        session_id = None
        _get_or_create_guest_token()
        if guest_questions_remaining() <= 0:
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

    try:
        result = get_answer(question, chat_history=chat_history) or {}
    except Exception:
        log.exception("RAG pipeline error")
        return jsonify({"error": "Failed to get answer. Please try again."}), 500

    if user_id is None:
        register_guest_question()

    answer = result.get("answer") or "Sorry, I couldn't generate an answer."
    source = result.get("source", "Unknown")
    page   = result.get("page", 0)
    route  = result.get("route", "unknown")
    cached = result.get("cached", False)

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

    if not question:
        return jsonify({"error": "Empty question"}), 400

    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({"error": f"Question too long — please keep it under {MAX_QUESTION_LENGTH} characters."}), 400

    if user_id is None:
        # Guests have no saved sessions — never persist/own one.
        session_id = None
        _get_or_create_guest_token()
        if guest_questions_remaining() <= 0:
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

    def generate():
        full_answer = ""
        meta = {}
        try:
            for kind, payload in get_answer_stream(question, chat_history=chat_history):
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

        answer = meta.get("answer") or full_answer
        source = meta.get("source", "Unknown")
        page   = meta.get("page", 0)
        route  = meta.get("route", "unknown")

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

@app.route("/suggestions")
def suggestions():
    questions = [
        "What was the inflation rate in Ethiopia in 2018?",
        "Tell me about livestock production in Ethiopia",
        "What is the land utilization in Amhara region?",
        "How many households are in the survey?",
        "What does the 2022 statistical report say?",
        "Tell me about the demographic health survey",
        "What is the population of Ethiopia?",
        "Describe the agricultural survey findings"
    ]
    return jsonify({"suggestions": questions})

@app.route("/guest_status")
def guest_status():
    return jsonify({
        "is_guest": current_user_id() is None,
        "remaining": guest_questions_remaining() if current_user_id() is None else None,
        "limit": GUEST_QUESTION_LIMIT
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