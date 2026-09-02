#!/usr/bin/env python3
"""
Preflight check — run this before `docker compose up --build`.

Catches syntax errors, unused imports, broken Dockerfile COPY paths, and
frontend JS syntax errors in a couple of seconds, using nothing but the
Python standard library (+ node, if available, for a real JS syntax
check). No project dependencies need to be installed to run this.

Usage:
    python scripts/preflight_check.py
"""
import ast
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
errors: list[str] = []
warnings: list[str] = []


def _python_files():
    for base in ("app", "scripts"):
        base_dir = ROOT / base
        if base_dir.exists():
            yield from base_dir.rglob("*.py")


def check_python_syntax() -> None:
    for f in _python_files():
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"[syntax] {f.relative_to(ROOT)}: {e}")


def check_unused_imports() -> None:
    for f in _python_files():
        src = f.read_text()
        try:
            tree = ast.parse(src, filename=str(f))
        except SyntaxError:
            continue  # already reported by check_python_syntax
        names_used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = (alias.asname or alias.name).split(".")[0]
                    if name != "*" and name not in names_used:
                        warnings.append(f"[unused-import] {f.relative_to(ROOT)}: '{name}'")


def check_frontend_js() -> None:
    index_html = ROOT / "frontend" / "index.html"
    if not index_html.exists():
        errors.append("[frontend] frontend/index.html not found")
        return

    html = index_html.read_text()
    if "<script>" not in html:
        warnings.append("[frontend] No <script> block found in frontend/index.html")
        return

    js = html.split("<script>")[1].split("</script>")[0]
    if js.count("{") != js.count("}"):
        errors.append("[frontend] Unbalanced braces in <script> block")
    if js.count("(") != js.count(")"):
        errors.append("[frontend] Unbalanced parens in <script> block")

    if shutil.which("node"):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
            tmp.write(js)
            tmp_path = tmp.name
        result = subprocess.run(["node", "--check", tmp_path], capture_output=True, text=True)
        if result.returncode != 0:
            errors.append(f"[frontend] JS syntax error:\n{result.stderr.strip()}")
    else:
        warnings.append("[frontend] 'node' not found on PATH — skipped real JS syntax check (brace/paren balance only)")


def check_requirements_pinned() -> None:
    req_files = ["requirements.txt", "requirements-ollama.txt"]
    found_any = False
    for name in req_files:
        req = ROOT / name
        if not req.exists():
            continue
        found_any = True
        for line in req.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "==" not in line:
                warnings.append(f"[requirements] Unpinned dependency in {name}: '{line}' (pin a version for reproducible builds)")
    if not found_any:
        errors.append("[requirements] Neither requirements.txt nor requirements-ollama.txt found")


def check_dockerfile_paths() -> None:
    dockerfiles = ["Dockerfile", "Dockerfile.ollama"]
    found_any = False
    for name in dockerfiles:
        dockerfile = ROOT / name
        if not dockerfile.exists():
            continue
        found_any = True
        for line in dockerfile.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("COPY"):
                parts = stripped.split()
                if len(parts) >= 2:
                    src = parts[1]
                    if src not in (".", "./") and not (ROOT / src).exists():
                        errors.append(f"[docker] {name} COPYs a path that doesn't exist: '{src}'")
    if not found_any:
        errors.append("[docker] Neither Dockerfile nor Dockerfile.ollama found")


def check_compose_files() -> None:
    compose_files = ["docker-compose.yml", "docker-compose.ollama.yml"]
    for name in compose_files:
        path = ROOT / name
        if not path.exists():
            warnings.append(f"[docker] {name} not found")


def check_env_example() -> None:
    if not (ROOT / ".env.example").exists():
        warnings.append("[env] .env.example not found")


def main() -> None:
    check_python_syntax()
    check_unused_imports()
    check_frontend_js()
    check_requirements_pinned()
    check_dockerfile_paths()
    check_compose_files()
    check_env_example()

    if warnings:
        print(f"⚠ {len(warnings)} warning(s):\n")
        for w in warnings:
            print(f"  - {w}")
        print()

    if errors:
        print(f"✗ {len(errors)} error(s) found:\n")
        for e in errors:
            print(f"  - {e}")
        print("\nFix these before running `docker compose up --build`.")
        sys.exit(1)

    print("✓ All checks passed. Safe to `docker compose up --build`.")
    sys.exit(0)


if __name__ == "__main__":
    main()
