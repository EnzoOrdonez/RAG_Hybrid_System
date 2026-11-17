"""Verify that every repository path cited in docs/guides/project-map.md exists (F-017).

A citation is a backticked token that looks like a repository-relative path:
it contains "/", has no spaces, no drive letter and no glob, and either ends
with "/" (directory) or its last segment has an extension. An optional
":N" or ":N-M" suffix is checked against the file's line count. Local paths
ignored by git (data/, C:/CloudRAG/...) are written without backticks in the
map, so they are not checked here.

Exit code 0 only when every cited path exists. Offline and stdlib-only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP = Path("docs/guides/project-map.md")
TOKEN = re.compile(r"`([^`\s]+)`")
CITATION = re.compile(r"(?P<path>[\w.\-/]+?)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?")


def cited_paths(text: str) -> list[tuple[str, int | None]]:
    """Return (path, last cited line or None) for each path-like backticked token."""
    found: list[tuple[str, int | None]] = []
    for token in TOKEN.findall(text):
        match = CITATION.fullmatch(token)
        if not match:
            continue
        path = match["path"]
        if "/" not in path or path.startswith(("/", ".")) or ".." in path:
            continue
        if not (path.endswith("/") or "." in path.rsplit("/", 1)[-1]):
            continue
        last = match["end"] or match["start"]
        found.append((path, int(last) if last else None))
    return found


def check(text: str, root: Path) -> list[str]:
    errors: list[str] = []
    for path, line in cited_paths(text):
        target = root / path
        if not target.exists():
            errors.append(f"{path}: no existe")
        elif line is not None:
            total = len(target.read_text(encoding="utf-8").splitlines())
            if line > total:
                errors.append(f"{path}:{line}: el archivo tiene {total} líneas")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, default=None, help="mapa a verificar (por defecto docs/guides/project-map.md)")
    parser.add_argument("--root", type=Path, default=ROOT, help="raíz del repositorio")
    args = parser.parse_args(argv)
    text = (args.map or args.root / MAP).read_text(encoding="utf-8")
    errors = check(text, args.root)
    for error in errors:
        print(f"[check_project_map] {error}")
    print(f"[check_project_map] {len(cited_paths(text))} rutas citadas, {len(errors)} errores")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
