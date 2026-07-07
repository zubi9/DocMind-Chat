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

    # --- Embedding model ---
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- LLM ---
    llm_model_name: str = "microsoft/phi-2"
    llm_context_window: int = 2048
    llm_max_new_tokens: int = 256
    llm_device_map: str = "auto"

    # --- Retrieval defaults ---
    similarity_top_k: int = 4
    response_mode: str = "compact"

    # --- Supported ingestion file types ---
    supported_extensions: tuple = (".pdf", ".docx", ".txt", ".md")

    def ensure_directories(self) -> None:
        """Create the data directories if they don't already exist."""
        for path in (self.data_dir, self.user_docs_dir, self.chroma_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
