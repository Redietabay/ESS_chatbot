"""
Run this directly:  python test_groq_connection.py
Tells you EXACTLY which layer is broken: DNS, TCP/firewall, TLS, or your API key.
"""
import os
import socket
import traceback
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("1. Checking GROQ_API_KEY is loaded from .env")
print("=" * 60)
key = os.getenv("GROQ_API_KEY")
if not key:
    print("FAIL: GROQ_API_KEY is not set / .env not found in this folder.")
    print("      Make sure you run this from the same folder as .env")
else:
    print(f"OK: key loaded, starts with '{key[:7]}...', length={len(key)}")

print()
print("=" * 60)
print("2. DNS resolution for api.groq.com")
print("=" * 60)
try:
    ip = socket.gethostbyname("api.groq.com")
    print(f"OK: api.groq.com -> {ip}")
except Exception as e:
    print(f"FAIL: DNS lookup failed: {e}")
    print("      Your network/DNS can't resolve Groq's domain at all.")
    print("      Try a different network (mobile hotspot) to confirm.")

print()
print("=" * 60)
print("3. Raw TCP + TLS connection to api.groq.com:443")
print("=" * 60)
try:
    import ssl
    ctx = ssl.create_default_context()
    with socket.create_connection(("api.groq.com", 443), timeout=8) as sock:
        with ctx.wrap_socket(sock, server_hostname="api.groq.com") as ssock:
            print(f"OK: TLS connection established, protocol={ssock.version()}")
except Exception as e:
    print(f"FAIL: Could not open TLS connection: {e}")
    print("      This means a firewall, antivirus, or network proxy is")
    print("      blocking outbound HTTPS to api.groq.com specifically.")
    print("      Try: disable antivirus/VPN temporarily, or switch to")
    print("      mobile hotspot, then re-run this script.")

print()
print("=" * 60)
print("4. Real Groq SDK call (full traceback, not swallowed)")
print("=" * 60)
try:
    from groq import Groq
    client = Groq(api_key=key, max_retries=0)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5,
        timeout=10,
    )
    print(f"OK: Groq responded: {resp.choices[0].message.content!r}")
except Exception:
    print("FAIL: Groq SDK call raised an exception. Full traceback:")
    print("-" * 60)
    traceback.print_exc()
    print("-" * 60)

print()
print("Done. Send me the full output of this script.")
