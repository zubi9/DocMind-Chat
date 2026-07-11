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
| Models loaded at top of every cell run | Loaded once at FastAPI startup, kept in memory, switchable at runtime from the UI |
| Single hard-coded LLM/embedding model | Curated Model Gallery — pick from Phi-2, Phi-4-mini, Llama 3.2 1B/3B, SmolLM2 1.7B, and 4 lightweight embedding models, right from the sidebar |
| Index built via `VectorStoreIndex.from_documents(...)`, persisted to a plain `./storage` folder | Index backed by **ChromaDB**, one collection per embedding model (so switching embeddings never mixes incompatible vectors) |
| Ingest = embed, always, immediately | Ingest and embed are decoupled: ingesting just saves the source; embedding happens on demand via a dedicated **"Create Embeddings"** button |
| No handling of documents changing after indexing | Deleting/adding documents marks the index **stale**, and both the Documents page and the Chat page tell you to rebuild |
| YouTube only | YouTube transcripts **and** general web pages (article text extraction via trafilatura) |
| `input()` interactive chat loop | `POST /query` JSON endpoint |
| No persistence across Colab runtime resets | Docker volume (`./data`) persists documents, vector DB, active model choice, and the HF model cache |

## New in this version: embeddings, models, and web ingestion

**Ingest ≠ Embed.** Uploading a file, a YouTube URL, or a web page just saves
the source and validates it's readable — it does **not** immediately touch
the vector index. You build (or rebuild) the searchable index explicitly by
clicking **Create Embeddings** on the Documents page (`POST /embeddings/build`).
This is a full rebuild from everything currently in `data/user_docs/`, which
keeps the mental model simple: one button, one source of truth.

**Deleting documents marks the index stale.** `DELETE /documents/{filename}`
removes the file but leaves the vector index untouched until you rebuild —
the Documents page shows an amber "Embeddings out of date" banner, and the
Chat page shows the same warning inline, both linking straight to the
rebuild button.

**Model Gallery.** The Chat page's left sidebar has two rows — "Generation"
and "Embedding" — each opening a picker of curated lightweight models
(`GET /models`, `POST /models/llm`, `POST /models/embedding`):

- **Generation:** Phi-2 (default), Phi-4-mini, Llama 3.2 1B Instruct, Llama 3.2 3B Instruct, SmolLM2 1.7B Instruct
- **Embedding:** MiniLM-L6-v2 (default), BGE-small-en v1.5, E5-small-v2, GTE-small

Switching the generation model unloads the old one and loads the new one —
this can take a while on first use (HuggingFace download) and is quick
afterwards. Switching the embedding model does **not** re-embed
automatically (each embedding model gets its own Chroma collection, since
their vector spaces aren't compatible) — you'll be prompted to rebuild.
The active choice for both is persisted to `data/app_state.json` and
survives restarts. See `app/core/model_catalog.py` to add more models.

> Llama 3.2 models are gated on HuggingFace: set `HF_TOKEN` in `.env` after
> accepting the license on the model's HuggingFace page, or model switching
> to those will fail.

**Web page ingestion.** The Documents page now has a third card, "Web
Page", which fetches a URL and extracts its main article text (boilerplate,
nav, and ads stripped via `trafilatura`), alongside the existing document
upload and YouTube transcript cards.

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
│   │   ├── llm_setup.py     # loads + switches embed model / LLM (Settings)
│   │   ├── model_catalog.py # curated lightweight LLM + embedding model list
│   │   ├── state_store.py   # persisted app_state.json (active models, index status, doc sources)
│   │   ├── ingestion.py     # PDF/DOCX/TXT/MD + YouTube + web ingestion (ported from sandbox)
│   │   ├── indexing.py      # Chroma-backed VectorStoreIndex, one collection per embedding model
│   │   └── query_engine.py  # question -> structured answer + sources
│   └── routers/
│       ├── health.py        # GET /health, GET /
│       ├── ingest.py        # POST /ingest/document, /ingest/youtube, /ingest/web
│       ├── embeddings.py    # POST /embeddings/build, GET /embeddings/status
│       ├── models.py        # GET /models, POST /models/llm, /models/embedding
│       ├── query.py         # POST /query
│       └── documents.py     # GET /documents, DELETE /documents/{filename}
├── frontend/
│   └── index.html           # single-page UI (Chat + Documents + Model Gallery), served at /ui
├── data/                    # persisted volume: user_docs/, chroma_db/, hf_cache/, app_state.json
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

### Ingest sources (does not embed — see Build embeddings below)

Upload a document:
```bash
curl -X POST http://localhost:8000/ingest/document \
  -F "file=@/path/to/paper.pdf"
```

Ingest a YouTube transcript:
```bash
curl -X POST http://localhost:8000/ingest/youtube \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

Ingest a web page:
```bash
curl -X POST http://localhost:8000/ingest/web \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

### Build embeddings
```bash
curl -X POST http://localhost:8000/embeddings/build
curl http://localhost:8000/embeddings/status
```
`build` wipes and re-embeds everything in `data/user_docs/` using the
currently active embedding model. Call it after ingesting new sources,
deleting documents, or switching the embedding model.

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
  ],
  "index_stale": false
}
```
Returns `409` if no embeddings have been built yet.

### List / delete documents
```bash
curl http://localhost:8000/documents
curl -X DELETE http://localhost:8000/documents/paper.pdf
```
Deleting marks the index `stale` — rebuild via `POST /embeddings/build` to
fully remove it from search results.

### Models
```bash
curl http://localhost:8000/models
curl -X POST http://localhost:8000/models/llm -H "Content-Type: application/json" -d '{"model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}'
curl -X POST http://localhost:8000/models/embedding -H "Content-Type: application/json" -d '{"model_id": "BAAI/bge-small-en-v1.5"}'
```

## Notes / limitations carried over intentionally

- Embedding is a full rebuild each time (`POST /embeddings/build`), not an
  incremental insert — simpler and more predictable, at the cost of
  re-embedding everything even for one new document. Fine at the document
  counts this is designed for; revisit if you outgrow that.
- No auth is included — add an API key / reverse proxy layer before
  exposing this beyond localhost.
- Gated models (Llama 3.2 1B/3B) require `HF_TOKEN` in `.env` and accepting
  the license on HuggingFace first, or the model switch will fail with a
  clear error.
