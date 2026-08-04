FROM python:3.11-slim

# System deps: psycopg2-binary needs libpq at runtime; pymupdf/chromadb need build tools for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at BUILD time (needs network here, once).
# This is what index_pdfs.py / retriever.py load at import — baking it in
# means the deployed container never needs outbound HF access, and cold
# starts skip the multi-second (or failing, if egress is blocked) download.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

# Shell form (not exec-array form) so $PORT actually gets expanded.
# Railway assigns PORT dynamically — do not hardcode a port here.
CMD gunicorn --bind 0.0.0.0:$PORT app:app --workers 1 --threads 2 --timeout 120