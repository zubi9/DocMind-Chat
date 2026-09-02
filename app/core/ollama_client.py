"""
Minimal client for the Ollama REST API (https://github.com/ollama/ollama/blob/main/docs/api.md).

Used for two things: checking which models are already pulled locally
(`list_local_models` / `is_pulled`), and pulling a new one on first use
with progress logged periodically (`pull_model`) rather than blocking
silently for however long the download takes.

Deliberately implemented with plain `requests` calls instead of the
`ollama` python package, to keep the Ollama-backend dependency footprint
small (this + `llama-index-llms-ollama` + `llama-index-embeddings-ollama`
is enough).
"""
import json
import logging
import time

import requests

from app.config import settings

logger = logging.getLogger("docmind.ollama_client")


class OllamaConnectionError(Exception):
    """Raised when the Ollama server can't be reached at all."""


class OllamaPullError(Exception):
    """Raised when pulling a model fails (bad tag, disk space, etc.)."""


def list_local_models(base_url: str | None = None) -> list[str]:
    """Returns the tags of every model currently pulled on the Ollama server."""
    url = f"{(base_url or settings.ollama_base_url).rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaConnectionError(
            f"Could not reach Ollama at {base_url or settings.ollama_base_url}: {e}"
        ) from e

    data = resp.json()
    return [m["name"] for m in data.get("models", [])]


def is_pulled(model_id: str, base_url: str | None = None) -> bool:
    """True if `model_id` (or a matching tag) is already pulled locally."""
    try:
        local = list_local_models(base_url)
    except OllamaConnectionError:
        return False

    if model_id in local:
        return True
    # Ollama normalizes "phi3:mini" == "phi3:mini" but a bare "phi3" tag
    # request should also match "phi3:latest" if that's what's pulled.
    if ":" not in model_id:
        return any(tag == f"{model_id}:latest" or tag.startswith(f"{model_id}:") for tag in local)
    return False


def pull_model(model_id: str, base_url: str | None = None, timeout: float | None = None) -> None:
    """Pulls `model_id` from the Ollama registry, streaming and logging progress.

    Ollama's /api/pull endpoint streams newline-delimited JSON status
    updates while it downloads and verifies layers. We log a line roughly
    every few seconds (not on every chunk, which can be dozens of updates
    per second) so a long first-time pull shows visible progress instead
    of going silent for minutes.
    """
    url = f"{(base_url or settings.ollama_base_url).rstrip('/')}/api/pull"
    timeout = timeout or settings.ollama_pull_timeout_s

    logger.info("Pulling Ollama model '%s' (this can take a while on first use)...", model_id)
    last_log = 0.0

    try:
        with requests.post(url, json={"name": model_id, "stream": True}, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise OllamaPullError(f"Failed to pull '{model_id}': {chunk['error']}")

                now = time.monotonic()
                if now - last_log > 3:
                    status = chunk.get("status", "")
                    completed = chunk.get("completed")
                    total = chunk.get("total")
                    if completed and total:
                        pct = 100 * completed / total
                        logger.info("Pulling '%s': %s (%.1f%%)", model_id, status, pct)
                    else:
                        logger.info("Pulling '%s': %s", model_id, status)
                    last_log = now
    except requests.RequestException as e:
        raise OllamaConnectionError(f"Could not reach Ollama at {base_url or settings.ollama_base_url}: {e}") from e

    logger.info("Pull complete: '%s'.", model_id)


def ensure_pulled(model_id: str, base_url: str | None = None) -> None:
    """Pulls `model_id` if it isn't already present locally. No-op otherwise."""
    if is_pulled(model_id, base_url):
        logger.info("Ollama model '%s' already pulled — loading from local cache.", model_id)
        return
    pull_model(model_id, base_url)
