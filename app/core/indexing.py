"""
Index management, backed by a persistent ChromaDB collection.

Embedding is a deliberate, user-triggered step (`build_embeddings()`,
wired to POST /embeddings/build) rather than happening automatically on
every ingest. This is a straightforward full-rebuild model: every build
wipes and re-embeds everything currently in `user_docs/`. It's simpler
and more predictable than incremental inserts/deletes, and it plays well
with switching embedding models at runtime, since a different embedding
model produces vectors of a different (and incompatible) dimensionality —
each embedding model gets its own Chroma collection, named from its
model id.

Two things beyond the original design:

1. Chunk sizing is capped per the ACTIVE embedding model's real capacity
   (`max_input_tokens` in model_catalog.json), not a single global
   constant. Sending oversized chunks to a small BERT-style embedding
   model either errors outright (Ollama's llama-server hard-fails) or
   silently truncates most of the chunk's content (sentence-transformers
   auto-truncates) — see `_build_text_splitter`.

2. The chunked nodes (not just their vectors) are persisted to disk
   alongside Chroma, so BM25 keyword search survives a container
   restart. Chroma only stores vectors; without a persisted docstore,
   hybrid retrieval would silently degrade to vector-only after every
   restart until the next rebuild.
"""
import logging
import re
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import settings
from app.core.ingestion import load_documents_robust
from app.core.model_catalog import get_embedding_entry
from app.core.state_store import refresh_status_for_embed_model, update_state

logger = logging.getLogger("docmind.indexing")

_index: VectorStoreIndex | None = None
_index_embed_id: str | None = None
_chroma_client: chromadb.ClientAPI | None = None


def _slugify(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_").lower()


def _collection_name_for(embed_id: str) -> str:
    return f"{settings.chroma_collection_name}__{_slugify(embed_id)}"


def _docstore_dir(embed_id: str) -> Path:
    """Where chunked node text is persisted for BM25/hybrid retrieval (separate from Chroma's own storage)."""
    d = Path(settings.chroma_dir) / "docstores" / _slugify(embed_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def _build_text_splitter(embed_id: str) -> SentenceSplitter:
    """Builds a SentenceSplitter capped to the active embedding model's real capacity.

    `settings.chunk_size` is a ceiling; if the active embedding model
    advertises a smaller `max_input_tokens` in model_catalog.json, we
    shrink to fit (with a safety margin for special tokens the tokenizer
    adds on top of the raw text). This is the fix for the embedding
    build crashing with "input length exceeds the context length" —
    that happened because no chunk-size cap existed at all, so LlamaIndex's
    global default (1024 tokens) was sent straight to models that can
    only handle 256-512.
    """
    entry = get_embedding_entry(embed_id)
    max_input_tokens = entry.get("max_input_tokens") if entry else None

    chunk_size = settings.chunk_size
    if max_input_tokens:
        safety_margin = 16
        safe_ceiling = max(64, max_input_tokens - safety_margin)
        if safe_ceiling < chunk_size:
            logger.info(
                "Capping chunk_size %d -> %d to fit '%s' (max_input_tokens=%d).",
                chunk_size, safe_ceiling, embed_id, max_input_tokens,
            )
        chunk_size = min(chunk_size, safe_ceiling)

    chunk_overlap = min(settings.chunk_overlap, chunk_size // 4)
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def get_index() -> VectorStoreIndex:
    """Returns the in-memory index for the currently active embedding model."""
    global _index, _index_embed_id
    embed_id = _active_embed_id()
    if _index is None or _index_embed_id != embed_id:
        _index = _load_index(embed_id)
        _index_embed_id = embed_id
    return _index


def _load_storage_context(embed_id: str, vector_store: ChromaVectorStore) -> StorageContext:
    """Loads the persisted docstore (if any) alongside the given vector store.

    Falls back to a fresh, empty (in-memory-only) docstore if nothing has
    been persisted yet — this is expected on first run, or right after
    switching to an embedding model that's never been built before.
    """
    docstore_dir = _docstore_dir(embed_id)
    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(docstore_dir))
        node_count = len(storage_context.docstore.docs)
        if node_count:
            logger.info("Loaded persisted docstore for '%s' (%d node(s)) — hybrid retrieval available.", embed_id, node_count)
        return storage_context
    except FileNotFoundError:
        return StorageContext.from_defaults(vector_store=vector_store)
    except Exception as e:
        logger.warning("Could not load persisted docstore for '%s' (%s) — hybrid retrieval will be vector-only until the next build.", embed_id, e)
        return StorageContext.from_defaults(vector_store=vector_store)


def _load_index(embed_id: str) -> VectorStoreIndex:
    vector_store = _get_vector_store(embed_id)
    storage_context = _load_storage_context(embed_id, vector_store)

    count = vector_store._collection.count()
    if count > 0:
        logger.info("Loading existing index for '%s' (%d vectors).", embed_id, count)
        # Deliberately not using VectorStoreIndex.from_vector_store() here —
        # it builds its own fresh StorageContext internally and won't let
        # us carry over the persisted docstore loaded above. Passing
        # nodes=[] with our own storage_context achieves the same "index
        # backed by existing vectors" result, while keeping the docstore
        # (needed for BM25) attached.
        return VectorStoreIndex(nodes=[], storage_context=storage_context)

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

    # Wipe the persisted docstore too — it's a full rebuild, so any
    # previously cached nodes are about to be replaced wholesale.
    import shutil

    docstore_dir = _docstore_dir(embed_id)
    shutil.rmtree(docstore_dir, ignore_errors=True)
    docstore_dir.mkdir(parents=True, exist_ok=True)

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

    splitter = _build_text_splitter(embed_id)
    logger.info("Building embeddings for %d document(s) with '%s'...", len(documents), embed_id)
    _index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, transformations=[splitter]
    )
    _index_embed_id = embed_id

    try:
        storage_context.persist(persist_dir=str(docstore_dir))
        logger.info("Persisted docstore for hybrid retrieval (%d node(s)).", len(storage_context.docstore.docs))
    except Exception as e:
        logger.warning("Could not persist docstore (%s) — hybrid retrieval will be vector-only until the next build.", e)

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


def _build_hybrid_retriever(index: VectorStoreIndex, top_k: int):
    """Builds a vector + BM25 fusion retriever. Returns None (caller falls back to vector-only) if
    the optional BM25 dependency is missing, there are no cached nodes yet, or construction fails
    for any other reason — hybrid retrieval is a quality improvement, never a hard requirement.
    """
    try:
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever
    except ImportError:
        logger.debug("llama-index-retrievers-bm25 not installed — using vector-only retrieval.")
        return None

    node_count = len(index.docstore.docs)
    if node_count == 0:
        logger.debug(
            "No cached nodes in the docstore (e.g. right after a restart, before a rebuild) — "
            "using vector-only retrieval until the next 'Create Embeddings'."
        )
        return None

    try:
        vector_retriever = index.as_retriever(similarity_top_k=top_k)
        bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=top_k)
        return QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=top_k,
            num_queries=1,  # fuse the two retrieval methods only, no LLM-based query expansion
            mode="reciprocal_rerank",
            use_async=False,
        )
    except Exception as e:
        logger.warning("Could not build hybrid retriever (%s) — falling back to vector-only retrieval.", e)
        return None


def get_query_engine(top_k: int | None = None, response_mode: str | None = None):
    index = get_index()
    k = top_k or settings.similarity_top_k
    mode = response_mode or settings.response_mode

    if settings.enable_hybrid_retrieval:
        retriever = _build_hybrid_retriever(index, k)
        if retriever is not None:
            from llama_index.core.query_engine import RetrieverQueryEngine

            logger.debug("Using hybrid (vector + BM25) retrieval.")
            return RetrieverQueryEngine.from_args(retriever=retriever, response_mode=mode)

    return index.as_query_engine(similarity_top_k=k, response_mode=mode)
