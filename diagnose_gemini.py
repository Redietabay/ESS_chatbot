"""
Reproduces the EXACT REST call _call_gemini() in rag.py makes, but prints
the full response body instead of just letting requests.raise_for_status()
swallow it into a one-line "404 Client Error" message. The SDK-based
list_models() call succeeded with this same key, so the goal here is to see
WHY the direct REST call to generateContent fails differently.

Run:  python diagnose_gemini.py
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

print(f"Key present: {bool(GEMINI_API_KEY)}")
print(f"Key length: {len(GEMINI_API_KEY)}")
print(f"Key repr (first/last 6 chars): {GEMINI_API_KEY[:6]!r} ... {GEMINI_API_KEY[-6:]!r}")
print(f"Model: {GEMINI_MODEL!r}")
print(f"URL: {GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent")
print("-" * 60)

resp = requests.post(
    f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent",
    params={"key": GEMINI_API_KEY},
    json={
        "contents": [{"parts": [{"text": "Say OK"}]}],
        "generationConfig": {"maxOutputTokens": 10, "temperature": 0.0},
    },
    timeout=15,
)

print(f"Status code: {resp.status_code}")
print("Full response body:")
print(resp.text)
