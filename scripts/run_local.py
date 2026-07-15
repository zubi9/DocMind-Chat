#!/usr/bin/env python3
"""
Local dev runner — skips Docker entirely.

Creates (or reuses) a `.venv`, installs requirements.txt into it, points
data/model-cache paths at `./local_data` (so it never collides with your
Docker volume in `./data`), and launches `uvicorn --reload` directly on
your machine. Use this while iterating on backend code; switch back to
`docker compose up --build` when you want to verify the containerized
path (or before deploying).

Note: this still downloads/loads real model weights and needs the same
system packages the Dockerfile installs for OCR (tesseract-ocr,
poppler-utils) if you want PDF OCR fallback to work locally. Everything
else (upload, YouTube, web ingestion, embeddings, chat) works without
those.

Usage:
    python scripts/run_local.py
    python scripts/run_local.py --no-install   # skip the pip install step (faster restarts)
"""
import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT / ".venv"
LOCAL_DATA = ROOT / "local_data"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv() -> None:
    if VENV_DIR.exists():
        print(f"Using existing virtual environment at {VENV_DIR}")
        return
    print(f"Creating virtual environment at {VENV_DIR} ...")
    venv.EnvBuilder(with_pip=True).create(VENV_DIR)


def install_requirements() -> None:
    req = ROOT / "requirements.txt"
    print("Installing dependencies (skipped fast if already satisfied)...")
    subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "-q", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "-q", "-r", str(req)],
        check=True,
    )


def prepare_local_data_dirs() -> None:
    for sub in ("user_docs", "chroma_db", "hf_cache"):
        (LOCAL_DATA / sub).mkdir(parents=True, exist_ok=True)


def load_dotenv_into(env: dict) -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env.setdefault(key.strip(), value.strip())


def build_env() -> dict:
    env = os.environ.copy()
    load_dotenv_into(env)

    # Redirect data paths to a local folder instead of /app/data, so this
    # never collides with (or gets confused with) the Docker volume.
    env["DATA_DIR"] = str(LOCAL_DATA)
    env["USER_DOCS_DIR"] = str(LOCAL_DATA / "user_docs")
    env["CHROMA_DIR"] = str(LOCAL_DATA / "chroma_db")
    env["HF_HOME"] = str(LOCAL_DATA / "hf_cache")
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-install", action="store_true", help="Skip the pip install step.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("=== DocMind local dev runner (no Docker) ===")
    ensure_venv()
    if not args.no_install:
        install_requirements()
    else:
        print("Skipping dependency install (--no-install).")
    prepare_local_data_dirs()
    env = build_env()

    print(f"\nStarting FastAPI on http://localhost:{args.port}  (Ctrl+C to stop)")
    print(f"UI:    http://localhost:{args.port}/ui/")
    print(f"Docs:  http://localhost:{args.port}/docs")
    print(f"Data:  {LOCAL_DATA}\n")

    subprocess.run(
        [
            str(venv_python()), "-m", "uvicorn", "app.main:app",
            "--reload", "--host", "0.0.0.0", "--port", str(args.port),
        ],
        cwd=str(ROOT),
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
