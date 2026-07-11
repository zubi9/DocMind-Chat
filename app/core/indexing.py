"""
Index management, backed by a persistent ChromaDB collection.

Embedding is now a deliberate, user-triggered step (`build_embeddings()`,
wired to POST /embeddings/build) rather than happening automatically on
every ingest. This is a straightforward full-rebuild model: every build
wipes and re-embeds everything currently in `user_docs/`. It's simpler
and more predictable than incremental inserts/deletes, and it plays well
with switching embedding models at runtime, since a different embedding
model produces vectors of a different (and incompatible) dimensionality —
each embedding model gets its own Chroma collection, named from its
model id.
"""
import logging
import re

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import settings
from app.core.ingestion import load_documents_robust
from app.core.state_store import refresh_status_for_embed_model, update_state

logger = logging.getLogger("docmind.indexing")

_index: VectorStoreIndex | None = None
_index_embed_id: str | None = None
_chroma_client: chromadb.ClientAPI | None = None


def _slugify(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_").lower()


def _collection_name_for(embed_id: str) -> str:
    return f"{settings.chroma_collection_name}__{_slugify(embed_id)}"


def _get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _chroma_client


def _get_vector_store(embed_id: str) -> ChromaVectorStore:
    client = _get_chroma_client()
    collection = client.get_or_create_collection(_collection_name_for(embed_id))
    return ChromaVectorStore(chroma_collection=collection)


def _active_embed_id() -> str:
    from app.core.llm_setup import get_active_models

    embed_id = get_active_models()["embedding"]
    return embed_id or settings.embed_model_name


def get_index() -> VectorStoreIndex:
    """Returns the in-memory index for the currently active embedding model."""
    global _index, _index_embed_id
    embed_id = _active_embed_id()
    if _index is None or _index_embed_id != embed_id:
        _index = _load_index(embed_id)
        _index_embed_id = embed_id
    return _index


def _load_index(embed_id: str) -> VectorStoreIndex:
    vector_store = _get_vector_store(embed_id)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    count = vector_store._collection.count()
    if count > 0:
        logger.info("Loading existing index for '%s' (%d vectors).", embed_id, count)
        return VectorStoreIndex.from_vector_store(vector_store)

    logger.info("No existing vectors for '%s'. Creating an empty index.", embed_id)
    return VectorStoreIndex.from_documents([], storage_context=storage_context)


def build_embeddings() -> dict:
    """Wipes the active collection and re-embeds everything in user_docs/. Returns the new state."""
    global _index, _index_embed_id
    from app.core.llm_setup import get_active_models

    active = get_active_models()
    embed_id = active["embedding"]
    llm_id = active["llm"]

    client = _get_chroma_client()
    collection_name = _collection_name_for(embed_id)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # collection may not exist yet

    vector_store = _get_vector_store(embed_id)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = load_documents_robust()

    if not documents:
        logger.warning("No readable documents found in %s.", settings.user_docs_dir)
        _index = VectorStoreIndex.from_documents([], storage_context=storage_context)
        _index_embed_id = embed_id
        return update_state(
            index_status="empty",
            embedded_doc_count=0,
            embed_model_at_build=embed_id,
            llm_model_at_build=llm_id,
            last_built_at=None,
        )

    logger.info("Building embeddings for %d document(s) with '%s'...", len(documents), embed_id)
    _index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    _index_embed_id = embed_id
    logger.info("Embeddings built successfully.")

    from datetime import datetime, timezone

    return update_state(
        index_status="synced",
        embedded_doc_count=len(documents),
        embed_model_at_build=embed_id,
        llm_model_at_build=llm_id,
        last_built_at=datetime.now(timezone.utc).isoformat(),
    )


def refresh_state_after_embed_switch(embed_id: str) -> dict:
    """Call right after switching the active embedding model to reconcile index_status."""
    vector_store = _get_vector_store(embed_id)
    count = vector_store._collection.count()
    return refresh_status_for_embed_model(embed_id, count)


def can_query() -> tuple[bool, str | None]:
    from app.core.state_store import get_state

    status = get_state().get("index_status")
    if status == "empty":
        return False, "No embeddings have been built yet. Go to Documents and click 'Create Embeddings'."
    return True, None


def get_query_engine(top_k: int | None = None, response_mode: str | None = None):
    index = get_index()
    return index.as_query_engine(
        similarity_top_k=top_k or settings.similarity_top_k,
        response_mode=response_mode or settings.response_mode,
    )
