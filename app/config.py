"""
Central configuration for DocMind Chat.

All values can be overridden via environment variables or a `.env` file
(see `.env.example`). This replaces the hard-coded constants scattered
throughout the original Colab notebook.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App metadata ---
    app_name: str = "DocMind Chat"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # --- Storage paths ---
    data_dir: str = "/app/data"
    user_docs_dir: str = "/app/data/user_docs"
    chroma_dir: str = "/app/data/chroma_db"
    chroma_collection_name: str = "docmind_collection"

    # --- Model backend: "huggingface" (in-process transformers, fp16/fp32)
    # or "ollama" (quantized GGUF served by a separate Ollama container —
    # see docker-compose.ollama.yml). This is the main lever for the OOM
    # issue on CPU: Ollama's llama.cpp runtime with 4-bit quantization
    # uses a fraction of the memory of raw HuggingFace weights. ---
    model_backend: str = "huggingface"  # "huggingface" | "ollama"

    # --- HuggingFace backend: embedding + LLM ---
    # NOTE: Mistral-7B raw fp16/fp32 weights are meaningfully heavier than
    # the smaller models also in the catalog (Phi-2, SmolLM2, etc.) — this
    # is the same class of workload that caused the OOM kills documented
    # in the troubleshooting notes. Budget 16GB+ container memory for
    # comfortable CPU inference, or switch to the Ollama backend, which
    # runs the same model family pre-quantized at a fraction of the
    # memory (see docker-compose.ollama.yml).
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_context_window: int = 8192
    llm_max_new_tokens: int = 256
    llm_device_map: str = "auto"

    # --- Ollama backend ---
    ollama_base_url: str = "http://ollama:11434"  # service name in docker-compose.ollama.yml
    ollama_llm_model: str = "mistral:7b-instruct-v0.2-q4_0"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_request_timeout_s: float = 180.0
    ollama_pull_timeout_s: float = 1800.0  # first pull of a model can take a while

    # --- Retrieval defaults ---
    similarity_top_k: int = 4
    response_mode: str = "compact"

    # --- Chunking ---
    # Target chunk size in tokens. This is a ceiling, not a fixed value:
    # at index-build time we cap it further to whatever the ACTIVE
    # embedding model can actually handle (see max_input_tokens in
    # model_catalog.json), because sending oversized chunks to a small
    # BERT-style embedding model either errors outright (Ollama) or
    # silently truncates most of the content (HuggingFace/
    # sentence-transformers) — the exact failure seen with all-minilm's
    # real 256-token limit. See app/core/indexing.py::_build_text_splitter.
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Hybrid retrieval (vector + BM25 keyword search, fused via
    # reciprocal rank fusion) — improves recall for exact terms/names/
    # numbers that pure embedding similarity can miss. Falls back to
    # plain vector search automatically if BM25 can't be built (e.g. the
    # optional dependency is missing, or there are no cached nodes yet
    # after a restart — see indexing.py). ---
    enable_hybrid_retrieval: bool = True

    # --- Supported ingestion file types ---
    supported_extensions: tuple = (".pdf", ".docx", ".txt", ".md", ".json")

    def ensure_directories(self) -> None:
        """Create the data directories if they don't already exist."""
        for path in (self.data_dir, self.user_docs_dir, self.chroma_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
