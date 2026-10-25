"""Verify the documentation harness (layer 3 of init.sh).

Checks, all offline and stdlib-only:
  1. CLAUDE.md exists and has fewer than 100 lines.
  2. The repository root holds no Markdown other than CLAUDE.md and README.md.
  3. Every relative Markdown link in the living docs resolves (file and, for
     `#adr-NNNN` anchors, the anchor).
  4. .harness/feature_list.json matches the schema in
     .harness/feature_list.schema.json (validated here without third-party deps).
  5. docs/archive/CLAUDE_legacy_backup.md keeps its recorded SHA-256 (immutability).
  6. Every non-blank line of that backup appears in at least one migrated file
     (no line of the previous CLAUDE.md was lost).

Exit code 0 only when every check passes. ADR-0014 in docs/decisions/adr-log.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP = Path("docs/archive/CLAUDE_legacy_backup.md")
BACKUP_SHA256 = "545d93526235db6feec84396b9b7891fbaba9f8ad91720dbd103efecea4fe7d4"
ROOT_MARKDOWN_ALLOWED = {"CLAUDE.md", "README.md"}
CLAUDE_MAX_LINES = 100
LIVING_DOCS = [
    "CLAUDE.md",
    "README.md",
    "docs/README.md",
    "docs/archive/README.md",
    ".harness/progress.md",
    "docs/architecture/*.md",
    "docs/decisions/*.md",
    "docs/guides/*.md",
    "docs/reports/*.md",
    "docs/specs/*.md",
]
COVERAGE_TARGETS = [
    "CLAUDE.md",
    ".harness/progress.md",
    "docs/decisions/session-log.md",
    "docs/decisions/adr-log.md",
    "docs/guides/guardrails.md",
    "docs/architecture/system-design.md",
]
FEATURE_KEYS = {"id", "category", "description", "steps", "status", "passes"}
FEATURE_STATUS = {"not_started", "active", "passing", "blocked"}
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_claude_md(root: Path) -> list[str]:
    path = root / "CLAUDE.md"
    if not path.is_file():
        return ["CLAUDE.md is missing"]
    n = len(_read(path).splitlines())
    return [] if n < CLAUDE_MAX_LINES else [f"CLAUDE.md has {n} lines (limit < {CLAUDE_MAX_LINES})"]


def check_clean_root(root: Path) -> list[str]:
    extra = sorted(p.name for p in root.glob("*.md") if p.name not in ROOT_MARKDOWN_ALLOWED)
    return [f"root Markdown outside the allowed set: {name}" for name in extra]


def _living_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in LIVING_DOCS:
        files.extend(sorted(root.glob(pattern)))
    return [f for f in files if f.is_file()]


def check_links(root: Path) -> list[str]:
    errors: list[str] = []
    for doc in _living_files(root):
        text = _read(doc)
        for target in LINK_RE.findall(text):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto:
                continue
            path_part, _, anchor = target.partition("#")
            dest = doc if not path_part else (doc.parent / path_part).resolve()
            rel = doc.relative_to(root)
            if not dest.exists():
                errors.append(f"{rel}: broken link -> {target}")
                continue
            if anchor and dest.is_file() and dest.suffix == ".md" and anchor.startswith("adr-"):
                if f'<a id="{anchor}"></a>' not in _read(dest):
                    errors.append(f"{rel}: missing anchor -> {target}")
    return errors


def validate_features(data: object) -> list[str]:
    if not isinstance(data, dict) or not isinstance(data.get("features"), list):
        return ['feature_list.json must be an object with a "features" list']
    errors: list[str] = []
    seen: set[str] = set()
    for i, feat in enumerate(data["features"]):
        where = f"features[{i}]"
        if not isinstance(feat, dict):
            errors.append(f"{where}: not an object")
            continue
        keys = set(feat)
        if keys != FEATURE_KEYS:
            errors.append(f"{where}: keys {sorted(keys)} != {sorted(FEATURE_KEYS)}")
            continue
        fid = feat["id"]
        if not isinstance(fid, str) or not re.fullmatch(r"F-\d{3}", fid):
            errors.append(f"{where}: id {fid!r} must match F-NNN")
        elif fid in seen:
            errors.append(f"{where}: duplicate id {fid}")
        else:
            seen.add(fid)
        for key in ("category", "description"):
            if not isinstance(feat[key], str) or not feat[key].strip():
                errors.append(f"{where}: {key} must be a non-empty string")
        steps = feat["steps"]
        if not isinstance(steps, list) or not steps or not all(isinstance(s, str) and s.strip() for s in steps):
            errors.append(f"{where}: steps must be a non-empty list of strings")
        if feat["status"] not in FEATURE_STATUS:
            errors.append(f"{where}: status {feat['status']!r} not in {sorted(FEATURE_STATUS)}")
        if not isinstance(feat["passes"], bool):
            errors.append(f"{where}: passes must be boolean")
        elif feat["passes"] != (feat["status"] == "passing"):
            errors.append(f"{where}: passes={feat['passes']} contradicts status={feat['status']!r}")
    return errors


def check_features(root: Path) -> list[str]:
    path = root / ".harness" / "feature_list.json"
    if not path.is_file():
        return [".harness/feature_list.json is missing"]
    if not (root / ".harness" / "feature_list.schema.json").is_file():
        return [".harness/feature_list.schema.json is missing"]
    try:
        data = json.loads(_read(path))
    except json.JSONDecodeError as exc:
        return [f".harness/feature_list.json is not valid JSON: {exc}"]
    return validate_features(data)


def check_backup(root: Path) -> list[str]:
    path = root / BACKUP
    if not path.is_file():
        return [f"{BACKUP} is missing"]
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return [] if digest == BACKUP_SHA256 else [f"{BACKUP} changed: sha256 {digest}"]


def check_coverage(root: Path) -> list[str]:
    backup = root / BACKUP
    if not backup.is_file():
        return []
    present: set[str] = set()
    for rel in COVERAGE_TARGETS:
        target = root / rel
        if target.is_file():
            present.update(line.rstrip() for line in _read(target).splitlines())
    missing = [
        line for line in _read(backup).splitlines() if line.strip() and line.rstrip() not in present
    ]
    return [f"line of the previous CLAUDE.md not found in migrated files: {m[:100]!r}" for m in missing]


CHECKS = [
    ("CLAUDE.md size", check_claude_md),
    ("clean root", check_clean_root),
    ("links", check_links),
    ("feature_list schema", check_features),
    ("backup immutability", check_backup),
    ("line coverage", check_coverage),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    failed = 0
    for name, check in CHECKS:
        errors = check(args.root.resolve())
        status = "OK" if not errors else f"FAIL ({len(errors)})"
        print(f"[check_docs] {name}: {status}")
        for err in errors[:50]:
            print(f"    - {err}")
        failed += bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
