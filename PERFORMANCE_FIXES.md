# ESS Chatbot Performance Optimization Guide

## Root Cause Analysis

Your response times are increasing due to these bottlenecks (in order of impact):

| Issue | Delay | Location | Fix Priority |
|-------|-------|----------|--------------|
| **Groq API rate limiting (429)** | 17-50 sec | `rag.py:call_groq()` | **URGENT** |
| **Query rewrite LLM call** | 1-2 sec/req | `rag.py:rewrite_and_route()` | **HIGH** |
| **PDF vector search latency** | 1-2 sec/req | `retriever.py:search_documents()` | **MEDIUM** |
| **Pandas code generation** | 0.5-1 sec | `rag.py:query_csv_with_groq()` | **MEDIUM** |
| **Initial model loading** | 30+ sec startup | First request only | **LOW** (one-time) |

---

## QUICK FIXES (Apply First)

### Fix #1: Reduce Rate Limit Wait Time (5 min work)

**File:** [rag.py](rag.py#L58)

The Groq API is asking for 17-50 second retries. You're currently respecting up to 4 seconds—cap it at 1 second so failures happen faster and the fallback logic kicks in.

**Change in `.env`:**
```bash
# From: 4.0 seconds default
# To: 1 second cap
GROQ_RATE_LIMIT_MAX_WAIT=1.0
```

**Why:** Your logs show:
```
Retrying request to /openai/v1/chat/completions in 17.000000 seconds
Retrying request to /openai/v1/chat/completions in 50.000000 seconds
```
When rate-limited, you'll fail faster and fallback to PDF-only search (which doesn't hit rate limits).

---

### Fix #2: Cache Rewrite+Route Decisions (10 min work)

The `rewrite_and_route()` function calls an LLM on **every single question**, even duplicates. Add a memory cache.

**File:** [rag.py](rag.py#L684)

Replace the `rewrite_and_route` function with this optimized version that caches decisions:

```python
# Add this near the top of rag.py, after imports
_REWRITE_ROUTE_CACHE = {}  # In-memory cache: {normalized_q: (cleaned_q, route)}
_REWRITE_CACHE_TTL_SECONDS = 3600  # 1 hour

def _get_cached_rewrite_route(question: str):
    """Fast return if we've seen this question before (within 1 hour)."""
    normalized = _normalize_question(question)
    if normalized in _REWRITE_ROUTE_CACHE:
        entry, timestamp = _REWRITE_ROUTE_CACHE[normalized]
        if time.time() - timestamp < _REWRITE_CACHE_TTL_SECONDS:
            return entry, True  # Return cached (cleaned_q, route), is_cached=True
    return None, False

def _cache_rewrite_route(question: str, cleaned_q: str, route: str):
    """Store rewrite decision for future similar questions."""
    normalized = _normalize_question(question)
    _REWRITE_ROUTE_CACHE[normalized] = ((cleaned_q, route), time.time())

# Now modify rewrite_and_route() to use the cache:
def rewrite_and_route(question: str, chat_history: list = None) -> tuple:
    # Try the cache first
    cached_result, is_cached = _get_cached_rewrite_route(question)
    if is_cached and not chat_history:  # Skip cache if there's conversation context
        log.info(f"Using cached rewrite+route for: '{question[:50]}'")
        return cached_result

    # ... rest of existing rewrite_and_route code ...
    
    # At the END of the function, add caching before return:
    cleaned_question, route = ...  # from existing code
    
    # NEW: Cache this decision
    _cache_rewrite_route(question, cleaned_question, route)
    
    return cleaned_question, route
```

**Why:** Most users ask variations of the same questions. This saves ~1-2 seconds per duplicate question.

---

### Fix #3: Add Request Queuing (Optional, 20 min work)

When multiple users hit Groq at once, they all compete. Add simple queuing:

**File:** Create a new file `request_queue.py`:

```python
import threading
import queue
import time
from datetime import datetime, timedelta

class RequestThrottler:
    """Rate-limit requests to Groq to avoid 429s."""
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.request_times = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Block until safe to make next request."""
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            if len(self.request_times) >= self.max_requests:
                # Need to wait
                oldest = self.request_times[0]
                wait_time = (oldest + 60) - now
                if wait_time > 0:
                    return wait_time
            
            self.request_times.append(now)
            return 0

# In rag.py, add at module level:
throttler = RequestThrottler(max_requests_per_minute=50)  # Adjust based on your limit

# In call_groq() function, add before the try block:
wait_time = throttler.wait_if_needed()
if wait_time > 0:
    log.info(f"Rate limit throttling: waiting {wait_time:.1f}s")
    time.sleep(wait_time)
```

---

## MEDIUM-TERM FIXES (1-2 hours)

### Fix #4: Async PDF Indexing + Caching

Your `retriever.py` is doing vector search from scratch each time. Pre-cache embeddings.

**File:** [retriever.py](retriever.py)

Add persistent ChromaDB persistence (it's partially there, but not fully optimized):

```python
# In retriever.py, ensure this is configured:
PERSIST_DIRECTORY = "./chroma_db"  # Already in your workspace!

# When initializing Chroma client:
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
collection = client.get_or_create_collection(name="documents")

# Make sure you're NOT re-embedding identical PDFs on each startup
# The current setup should work, but verify:
```

**What to check:**
- `chroma_db/chroma.sqlite3` should grow after first indexing
- Subsequent searches should hit the cache, not re-embed

---

### Fix #5: Implement Streamed Pandas Execution

The CSV queries sometimes hang. Add timeouts and better error handling:

**File:** [rag.py](rag.py#L150)

The code already has timeout logic (`_eval_with_timeout`), but increase the base timeout and add streaming results:

```python
# In rag.py, find the pandas execution code and ensure timeout is appropriate
PANDAS_EXECUTION_TIMEOUT = 5  # seconds (currently exists, keep as-is)

# But also add early abort for long-running queries:
def query_csv_with_groq(question: str, dataframes: dict, schemas: dict, retries: int = 2):
    """Generate and execute Pandas code with strict timeout."""
    # ... existing code ...
    
    try:
        result = safe_eval_pandas(generated_code, dataframes, timeout_seconds=5)
        if isinstance(result, (pd.DataFrame, pd.Series)):
            if len(result) > 500:
                log.warning(f"Large result truncated: {len(result)} rows")
                result = result.head(100)  # Keep smaller
    except TimeoutError:
        log.warning(f"Pandas execution timed out (5s)")
        return None  # Fall back to PDF search
```

---

## ADVANCED FIXES (If deploying to production)

### Fix #6: Use Redis for Distributed Caching

Replace in-memory cache with Redis for multi-worker deployments:

```python
import redis

# In rag.py
redis_client = redis.Redis(host='localhost', port=6379, db=0)
CACHE_TTL_SECONDS = 3600

def _get_cached_rewrite_route_redis(question: str):
    normalized = _normalize_question(question)
    cached = redis_client.get(f"rewrite:{normalized}")
    if cached:
        return json.loads(cached), True
    return None, False

def _cache_rewrite_route_redis(question: str, cleaned_q: str, route: str):
    normalized = _normalize_question(question)
    redis_client.setex(
        f"rewrite:{normalized}",
        CACHE_TTL_SECONDS,
        json.dumps({"cleaned": cleaned_q, "route": route})
    )
```

### Fix #7: Use Batch Processing for Multiple Users

In `app.py`, implement request batching:

```python
# Add to app.py
from queue import Queue
import threading

request_queue = Queue()

def process_queue():
    """Batch similar requests to reduce API calls."""
    while True:
        # Collect requests for 500ms
        pending = []
        deadline = time.time() + 0.5
        while time.time() < deadline:
            try:
                pending.append(request_queue.get(timeout=0.1))
            except:
                pass
        
        if pending:
            # Process batch...
            pass

threading.Thread(target=process_queue, daemon=True).start()
```

---

## Monitoring & Metrics

### Add Performance Logging

Update `rag.py` to track timing:

```python
import time

def get_answer_stream(question: str, chat_history: list = None):
    start_time = time.time()
    original_question = question
    log.info(f"[START] Processing Question: '{question[:80]}'")

    # ... existing code ...
    
    elapsed = time.time() - start_time
    if elapsed > 5:
        log.warning(f"[SLOW] Query took {elapsed:.1f}s: '{question[:50]}'")
    else:
        log.info(f"[PERF] Query completed in {elapsed:.1f}s")
```

### Check Groq Rate Limits

Add a health endpoint to `app.py`:

```python
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "cache_hits": get_cache_stats(),
        "groq_rate_limit_status": "check Groq dashboard"
    })
```

---

## Implementation Priority

**Apply in this order:**

1. ✅ **Fix #1** (5 min) — Reduce `GROQ_RATE_LIMIT_MAX_WAIT` to 1.0s
2. ✅ **Fix #2** (10 min) — Add rewrite+route caching  
3. ⏱️ **Fix #4** (5 min) — Verify ChromaDB persistence
4. ⏱️ **Fix #3** (20 min) — Add request throttling (optional)
5. 🔧 **Fix #5** (15 min) — Improve Pandas timeout handling
6. 🚀 **Fix #6 & #7** (production only)

---

## Expected Results

| Before | After |
|--------|-------|
| 8+ seconds per query | 2-3 seconds per query |
| 429 errors blocking 50s | Graceful fallback in 1s |
| No duplicate caching | Instant response for repeat Qs |

**Next Steps:**
1. Update `.env` with `GROQ_RATE_LIMIT_MAX_WAIT=1.0`
2. Implement in-memory rewrite cache (Fix #2)
3. Monitor logs for timing improvements
4. Report back with new response times!
