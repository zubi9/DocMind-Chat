"""
DocMind Chat — FastAPI service.

This is the productionized, API-driven version of the RAD_RAG_Sandbox
Colab notebook: the same ingestion, indexing, and query logic, wired up
behind FastAPI endpoints instead of an interactive input() loop, with
models loaded once at startup and the index persisted to disk via
ChromaDB (instead of re-running the whole notebook top to bottom).
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.indexing import get_index
from app.core.llm_setup import configure_models
from app.routers import documents, embeddings, health, ingest, models, query

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("docmind.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s ...", settings.app_name, settings.app_version)
    configure_models()
    get_index()  # loads existing Chroma collection, or creates an empty one
    logger.info("Startup complete. API ready to serve requests.")
    yield
    logger.info("Shutting down %s.", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "RAG-over-your-documents API: ingest PDFs, DOCX, TXT, Markdown, and "
        "YouTube transcripts, then ask questions grounded in that content."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin(s) before exposing publicly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(embeddings.router)
app.include_router(models.router)

# Serves the single-page frontend (app/../frontend/index.html) at /ui/.
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")
