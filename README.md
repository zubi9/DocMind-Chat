# DocMind Chat — API

A dockerized, FastAPI-based RAG service, converted from the original
`RAD_RAG_Sandbox.ipynb` Colab notebook into a deployable project. This
repo contains the backend only — endpoints for ingesting documents/YouTube
transcripts and querying them. A frontend can be built against this API
later.

## What changed vs. the notebook

| Notebook (Colab, RAD) | This project |
|---|---|
| `!pip install` cells | `requirements.txt` + Docker image build |
| `google.colab.files.upload()` | `POST /ingest/document` (multipart upload) |
| Models loaded at top of every cell run | Loaded once at FastAPI startup, kept in memory |
| Index built via `VectorStoreIndex.from_documents(...)`, persisted to a plain `./storage` folder (rebuild-everything only) | Index backed by a persistent **ChromaDB** collection, supporting incremental inserts (`index.insert()`) so adding one document doesn't require re-embedding everything |
| 3-way "guess the storage format" loader (`storage/`, `index.pkl`, `chroma_export.zip`) | Single, consistent Chroma-backed storage path |
| `input()` interactive chat loop | `POST /query` JSON endpoint |
| No persistence across Colab runtime resets | Docker volume (`./data`) persists documents, vector DB, and HF model cache |

The ingestion logic itself (PDF text extraction with OCR fallback, garbage-text
filtering, DOCX/TXT/MD loading, YouTube transcript fetching) is a direct,
line-for-line port of the sandbox's functions — see `app/core/ingestion.py`.

## Project structure

```
docmind-chat/
├── app/
│   ├── main.py              # FastAPI app + startup lifespan
│   ├── config.py            # env-driven settings
│   ├── models.py            # request/response schemas
│   ├── core/
│   │   ├── llm_setup.py     # configures embed model + LLM (Settings)
│   │   ├── ingestion.py     # PDF/DOCX/TXT/MD + YouTube ingestion (ported from sandbox)
│   │   ├── indexing.py      # Chroma-backed VectorStoreIndex management
│   │   └── query_engine.py  # question -> structured answer + sources
│   └── routers/
│       ├── health.py        # GET /health, GET /
│       ├── ingest.py        # POST /ingest/document, /ingest/youtube, /ingest/reindex
│       ├── query.py         # POST /query
│       └── documents.py     # GET /documents, DELETE /documents/{filename}
├── frontend/
│   └── index.html           # single-page UI (Chat + Documents), served at /ui
├── data/                    # persisted volume: user_docs/, chroma_db/, hf_cache/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Running it

```bash
cp .env.example .env      # adjust model names / paths if desired
docker compose up --build
```

The first startup will download the embedding model
(`sentence-transformers/all-MiniLM-L6-v2`) and the LLM
(`microsoft/phi-2`, ~2.7B params) from HuggingFace — this can take a
few minutes and needs a few GB of disk/RAM. Both are cached into
`./data/hf_cache` on the host, so subsequent restarts are fast.

Once running:
- **http://localhost:8000/ui/** — the DocMind single-page frontend (Chat + Documents)
- **http://localhost:8000/docs** — interactive Swagger API docs

The frontend is a single static `frontend/index.html` file (no build step,
no framework) that talks directly to the API endpoints below via `fetch`.
It's served by FastAPI itself (mounted at `/ui`), so nothing extra is
needed to run it — it ships in the same Docker image. CORS is enabled on
the API (`allow_origins=["*"]`) so you can also open `index.html` directly
from disk or serve it from a different host/port during development;
tighten `allow_origins` in `app/main.py` before deploying publicly.

### Hardware notes

- `phi-2` runs on CPU but is slow for interactive use; if you have an
  NVIDIA GPU, install the NVIDIA Container Toolkit on the host and
  uncomment the `deploy.resources` block in `docker-compose.yml`.
- To swap in a different (smaller/larger) LLM or embedding model, just
  change `LLM_MODEL_NAME` / `EMBED_MODEL_NAME` in `.env` — no code changes
  needed.

## API reference

### Health
```
GET /health
```

### Ingest a document
```bash
curl -X POST http://localhost:8000/ingest/document \
  -F "file=@/path/to/paper.pdf"
```

### Ingest a YouTube transcript
```bash
curl -X POST http://localhost:8000/ingest/youtube \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

### Ask a question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the morphology of a wheat spike?", "top_k": 4}'
```

Response:
```json
{
  "question": "What is the morphology of a wheat spike?",
  "answer": "...",
  "sources": [
    {"source": "wheat_paper.pdf", "score": 0.83, "preview": "..."}
  ]
}
```

### List / delete documents
```bash
curl http://localhost:8000/documents
curl -X DELETE http://localhost:8000/documents/paper.pdf
```

### Full reindex
Rebuilds the entire vector store from everything currently in
`data/user_docs/` (useful after several deletes, or after manually
dropping files into the volume):
```bash
curl -X POST http://localhost:8000/ingest/reindex
```

## Notes / limitations carried over intentionally

- `DELETE /documents/{filename}` removes vectors best-effort by matching
  Chroma metadata; if you need a guaranteed-clean store, follow it with
  `POST /ingest/reindex`.
- No auth is included — add an API key / reverse proxy layer before
  exposing this beyond localhost.
- No frontend is included per your request; all functionality is
  exercised via the endpoints above or the Swagger UI at `/docs`.
