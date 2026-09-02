FROM python:3.11-slim

# --- System dependencies ---
# tesseract-ocr + poppler-utils: needed for the PDF OCR fallback path
# (extract_pdf_text -> pdf2image -> pytesseract) ported from the sandbox.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code + static frontend.
COPY app ./app
COPY frontend ./frontend

# Data directory (documents, chroma db, hf model cache) — mounted as a
# volume in docker-compose so it survives container restarts/rebuilds.
RUN mkdir -p /app/data/user_docs /app/data/chroma_db /app/data/hf_cache

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/hf_cache

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
