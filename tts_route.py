# ═══════════════════════════════════════════════════════════════════
# /tts — server-side speech synthesis for languages the browser can't
# speak correctly on its own (Amharic, Afaan Oromoo).
#
# WHY THIS EXISTS:
# The browser's built-in Web Speech API (window.speechSynthesis) only
# sounds right for languages that have an installed OS/browser voice.
# Almost no desktop, Android, or iOS browser ships an Amharic (am-ET) or
# Afaan Oromoo (om-ET) voice. Setting utterance.lang to one of those
# doesn't throw an error — it silently substitutes a default voice
# (usually English), which reads Amharic/Oromo text with the wrong
# pronunciation. That's the exact bug being fixed here.
#
# FREE FIX: gTTS (`pip install gTTS`) is a thin, free, no-API-key wrapper
# around Google Translate's own speech endpoint. It has a real Amharic
# ('am') voice. It confirmed does NOT have an Afaan Oromoo ('om') voice —
# neither does Google Translate's own "listen" button, nor any other free
# provider found — so this route returns a clear "not supported yet"
# response for 'om' instead of silently mispronouncing it or crashing.
# (Next step for Oromo: the free Hugging Face Space "snackshell/selam-tts"
# has real Oromo voices via its Gradio API — worth wiring up after today's
# demo.)
#
# COST: $0. gTTS makes one HTTPS call per (uncached) request to Google
# Translate's public TTS endpoint — no account, no key, no quota you
# control. Caching + rate limiting below exist so a demo doesn't hammer
# that endpoint or feel slow on repeat clicks — not because it costs money.
# ═══════════════════════════════════════════════════════════════════

import hashlib
import io
import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from flask import Blueprint, request, jsonify, Response, send_file
from gtts import gTTS

log = logging.getLogger("tts")
tts_bp = Blueprint("tts", __name__)

# Keep this in sync with rag.py's target_lang values ('en' | 'am' | 'om').
GTTS_SUPPORTED = {"en", "am"}
MAX_TTS_CHARS = 1500  # keep requests fast + within gTTS' safe chunking size

# ─── Disk cache ───────────────────────────────────────────────────
# Same answer gets Listen-ed to more than once (user re-plays it, or the
# same cached chat answer gets served to a second person). Keying on a
# hash of (lang, text) means a repeat request is an instant file read
# instead of a fresh call to Google Translate.
CACHE_DIR = Path("tts_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(lang: str, text: str) -> Path:
    digest = hashlib.sha256(f"{lang}:{text}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.mp3"


# ─── Lightweight per-IP rate limit (no new dependency) ────────────
# 15 requests / minute / IP is generous for a human clicking Listen
# repeatedly, but stops a runaway script or a demo-day accident from
# hammering Google Translate's endpoint.
RATE_LIMIT = 15
RATE_WINDOW_SECONDS = 60
_hits = defaultdict(deque)  # ip -> deque of request timestamps


def _rate_limited(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW_SECONDS:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


@tts_bp.route("/tts", methods=["POST"])
def tts():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
    if _rate_limited(ip):
        return jsonify({"error": "Too many voice requests — wait a moment and try again."}), 429

    lang = (request.args.get("lang") or "en").strip().lower()
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    if lang not in GTTS_SUPPORTED:
        lang = "en"

    text = text[:MAX_TTS_CHARS]

    cache_file = _cache_path(lang, text)
    if cache_file.exists():
        return send_file(cache_file, mimetype="audio/mpeg")

    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        audio_bytes = buf.getvalue()

        # Write-then-rename avoids serving a half-written file if two
        # requests for the same brand-new text land at the same moment.
        tmp_path = cache_file.with_suffix(".tmp")
        tmp_path.write_bytes(audio_bytes)
        os.replace(tmp_path, cache_file)

        return Response(audio_bytes, mimetype="audio/mpeg")
    except Exception as e:
        log.error(f"gTTS synthesis failed (lang={lang}): {e}")
        return jsonify({"error": "Speech synthesis failed."}), 502


# ─── Wiring instructions (app.py) ───
# from tts_route import tts_bp
# app.register_blueprint(tts_bp)
#
# Add "gTTS" to requirements.txt (pure-Python, no system dependency —
# unlike pyttsx3 or Tesseract, nothing else to install on the server).
#
# Optional: add tts_cache/ to .gitignore — it's a runtime cache, not
# something to commit.