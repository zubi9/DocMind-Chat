"""
Index management, backed by a persistent ChromaDB collection.

The original sandbox script built the index with
`VectorStoreIndex.from_documents(...)` and persisted it to a plain
`./storage` folder, which only supports "rebuild everything from scratch"
workflows. For a long-running API service we want to be able to add a
single new document or transcript without re-embedding the entire
knowledge base, so this module wires up llama-index's Chroma vector store
integration (which the sandbox imported but never actually used for the
build step) and keeps a single in-memory `VectorStoreIndex` instance that
wraps it.
"""
import logging

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import settings
from app.core.ingestion import load_documents_robust

logger = logging.getLogger("docmind.indexing")

_index: VectorStoreIndex | None = None
_chroma_client: chromadb.ClientAPI | None = None


def _get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _chroma_client


def _get_vector_store() -> ChromaVectorStore:
    client = _get_chroma_client()
    collection = client.get_or_create_collection(settings.chroma_collection_name)
    return ChromaVectorStore(chroma_collection=collection)


def get_index() -> VectorStoreIndex:
    """Returns the current in-memory index, building/loading it on first access."""
    global _index
    if _index is None:
        _index = load_or_create_index()
    return _index


def load_or_create_index() -> VectorStoreIndex:
    """Loads the index from the persisted Chroma collection (creating an empty one if needed)."""
    vector_store = _get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    collection_count = vector_store._collection.count()
    if collection_count > 0:
        logger.info("Loading existing index from Chroma collection (%d vectors).", collection_count)
        index = VectorStoreIndex.from_vector_store(vector_store)
    else:
        logger.info("No existing vectors found. Creating an empty index.")
        index = VectorStoreIndex.from_documents([], storage_context=storage_context)

    return index


def add_documents(documents: list[Document]) -> int:
    """Inserts new documents into the live index. Returns the number of documents inserted."""
    if not documents:
        return 0

    index = get_index()
    for doc in documents:
        index.insert(doc)

    logger.info("Inserted %d document(s) into the index.", len(documents))
    return len(documents)


def rebuild_index_from_scratch() -> int:
    """Wipes the Chroma collection and re-ingests everything from user_docs. Returns doc count."""
    global _index

    client = _get_chroma_client()
    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass  # collection may not exist yet

    vector_store = _get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = load_documents_robust()
    if not documents:
        logger.warning("No readable documents found in %s.", settings.user_docs_dir)
        _index = VectorStoreIndex.from_documents([], storage_context=storage_context)
        return 0

    logger.info("Rebuilding index from %d document(s)...", len(documents))
    _index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    logger.info("Index rebuilt successfully.")
    return len(documents)


def delete_document_vectors(filename: str) -> None:
    """Best-effort removal of a document's chunks from the vector store by its 'source' metadata.

    Note: this relies on llama-index having propagated the Document's
    `source` metadata onto every chunk/node stored in Chroma, which is the
    default behavior. It is not a strict guarantee across all llama-index
    versions, so a full `rebuild_index_from_scratch()` is the reliable
    fallback if vectors are left behind.
    """
    vector_store = _get_vector_store()
    try:
        vector_store._collection.delete(where={"source": filename})
    except Exception as e:
        logger.warning("Could not remove vectors for %s: %s", filename, e)


def get_query_engine(top_k: int | None = None, response_mode: str | None = None):
    index = get_index()
    return index.as_query_engine(
        similarity_top_k=top_k or settings.similarity_top_k,
        response_mode=response_mode or settings.response_mode,
    )
