"""Scan repository text files for high-confidence credential patterns.

The scanner deliberately avoids secret-bearing file types that must never be opened
by automation (for example ``.env``, ``*.key``, and ``*.pem``). Findings expose only
the first four characters of the matched credential.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import fnmatch
import json
import os
from pathlib import Path
import re
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SecretPattern:
    name: str
    regex: re.Pattern[str]
    secret_group: str | int = 0


@dataclass(frozen=True)
class Finding:
    pattern: str
    path: Path
    line: int
    preview: str


@dataclass(frozen=True)
class BaselineExclusion:
    category: str
    path_pattern: str
    reason: str

    def matches(self, path: Path) -> bool:
        return fnmatch.fnmatchcase(path.as_posix(), self.path_pattern)


PATTERNS = (
    SecretPattern("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    SecretPattern("github_personal_token", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    SecretPattern("sk_token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    SecretPattern(
        "private_key",
        re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----"),
    ),
    SecretPattern(
        "password_assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd)\b\s*[:=]\s*"
            r"(?:[\"'](?P<quoted>[^\"'\r\n]{8,256})[\"']|"
            r"(?P<bare>[^\s;,#}\]\r\n]{8,256}))"
        ),
        "password",
    ),
)

_BINARY_SUFFIXES = {
    ".7z",
    ".bin",
    ".bmp",
    ".docx",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".onnx",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".pyc",
    ".tar",
    ".webp",
    ".xlsx",
    ".zip",
}

_GENERATED_DIRECTORIES = {
    ".codex",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}

_PROTECTED_FILENAMES = {
    "credentials.json",
    "secrets.toml",
}


def _matched_secret(match: re.Match[str], pattern: SecretPattern) -> str:
    if pattern.secret_group != "password":
        return match.group(pattern.secret_group)
    return match.group("quoted") or match.group("bare")


def scan_lines(
    lines: Iterable[str],
    path: Path,
    patterns: Iterable[SecretPattern] = PATTERNS,
) -> list[Finding]:
    """Return redacted findings from an iterable of decoded lines."""

    active_patterns = tuple(patterns)
    findings: list[Finding] = []
    for line_number, line in enumerate(lines, start=1):
        for pattern in active_patterns:
            for match in pattern.regex.finditer(line):
                secret = _matched_secret(match, pattern)
                findings.append(
                    Finding(
                        pattern=pattern.name,
                        path=path,
                        line=line_number,
                        preview=secret[:4],
                    )
                )
    return findings


def scan_text(text: str, path: Path, patterns: Iterable[SecretPattern] = PATTERNS) -> list[Finding]:
    """Return redacted findings for one decoded text payload."""

    return scan_lines(text.splitlines(), path, patterns)


def scan_stream(
    stream: object,
    path: Path,
    patterns: Iterable[SecretPattern] = PATTERNS,
    *,
    chunk_size: int = 1024 * 1024,
    overlap_size: int = 512,
) -> list[Finding]:
    """Scan a text stream in bounded chunks, preserving cross-boundary matches."""

    active_patterns = tuple(patterns)
    findings: list[Finding] = []
    prefix = ""
    newlines_before_chunk = 0

    while chunk := stream.read(chunk_size):
        combined = prefix + chunk
        combined_start_line = newlines_before_chunk - prefix.count("\n") + 1
        for pattern in active_patterns:
            for match in pattern.regex.finditer(combined):
                if match.end() <= len(prefix):
                    continue
                secret = _matched_secret(match, pattern)
                findings.append(
                    Finding(
                        pattern=pattern.name,
                        path=path,
                        line=combined_start_line + combined.count("\n", 0, match.start()),
                        preview=secret[:4],
                    )
                )
        newlines_before_chunk += chunk.count("\n")
        prefix = combined[-overlap_size:]

    return findings


def should_exclude(relative_path: Path) -> bool:
    """Return whether a repository-relative path must not be inspected."""

    parts = relative_path.parts
    if ".git" in parts:
        return True
    if any(part in _GENERATED_DIRECTORIES for part in parts):
        return True
    if len(parts) >= 2 and parts[:2] == ("data", "models"):
        return True
    if relative_path.name.lower() in _PROTECTED_FILENAMES:
        return True
    if relative_path.name == ".env" or relative_path.name.startswith(".env."):
        return True
    if relative_path.suffix.lower() in {".key", ".pem"}:
        return True
    if relative_path.name.endswith("_llmjudge_blind.csv"):
        return True
    if relative_path.suffix.lower() in _BINARY_SUFFIXES:
        return True
    return False


def iter_repository_files(root: Path) -> Iterable[Path]:
    """Yield inspectable files without following directory symlinks."""

    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root)
        directories[:] = [
            name
            for name in directories
            if not should_exclude(relative_dir / name)
        ]
        for filename in filenames:
            path = current_path / filename
            if not should_exclude(path.relative_to(root)):
                yield path


def scan_repository(root: Path) -> list[Finding]:
    """Scan inspectable repository files and return all redacted findings."""

    findings: list[Finding] = []
    for path in iter_repository_files(root):
        try:
            with path.open("rb") as binary_file:
                if b"\x00" in binary_file.read(8192):
                    continue
            with path.open("r", encoding="utf-8-sig", errors="replace") as text_file:
                findings.extend(scan_stream(text_file, path.relative_to(root)))
        except OSError as exc:
            raise RuntimeError(f"No se pudo leer {path.relative_to(root)}: {exc}") from exc
    return findings


def load_baseline(path: Path) -> tuple[BaselineExclusion, ...]:
    """Load and validate a declared JSON baseline."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"No se pudo cargar el baseline {path}: {exc}") from exc
    if document.get("version") != 1:
        raise RuntimeError("El baseline debe declarar version=1")

    exclusions: list[BaselineExclusion] = []
    reasons_by_category: dict[str, str] = {}
    for index, item in enumerate(document.get("exclusions", []), start=1):
        try:
            exclusion = BaselineExclusion(
                category=item["category"].strip(),
                path_pattern=item["path"].strip(),
                reason=item["reason"].strip(),
            )
        except (KeyError, AttributeError) as exc:
            raise RuntimeError(f"Exclusión inválida en posición {index}") from exc
        if not all((exclusion.category, exclusion.path_pattern, exclusion.reason)):
            raise RuntimeError(f"Exclusión vacía en posición {index}")
        previous_reason = reasons_by_category.setdefault(exclusion.category, exclusion.reason)
        if previous_reason != exclusion.reason:
            raise RuntimeError(
                f"La categoría {exclusion.category!r} declara razones distintas"
            )
        exclusions.append(exclusion)
    if not exclusions:
        raise RuntimeError("El baseline no contiene exclusiones")
    return tuple(exclusions)


def apply_baseline(
    findings: Sequence[Finding],
    exclusions: Sequence[BaselineExclusion],
) -> tuple[list[Finding], Counter[str]]:
    """Split active findings from declared baseline matches."""

    active: list[Finding] = []
    excluded_counts: Counter[str] = Counter()
    for finding in findings:
        exclusion = next(
            (candidate for candidate in exclusions if candidate.matches(finding.path)),
            None,
        )
        if exclusion is None:
            active.append(finding)
        else:
            excluded_counts[exclusion.category] += 1
    return active, excluded_counts


def format_report(
    findings: Sequence[Finding],
    excluded_counts: Counter[str] | None = None,
    baseline_reasons: dict[str, str] | None = None,
) -> str:
    """Format aggregate counts and redacted locations without secret values."""

    counts = Counter(finding.pattern for finding in findings)
    lines = ["Conteo por patrón:"]
    for pattern in PATTERNS:
        lines.append(f"  {pattern.name}: {counts.get(pattern.name, 0)}")
    if excluded_counts:
        lines.append("Excluidos por baseline:")
        for category in sorted(excluded_counts):
            reason = (baseline_reasons or {}).get(category, "sin razón declarada")
            lines.append(
                f"  {excluded_counts[category]} excluidos por baseline: "
                f"{category} ({reason})"
            )
    lines.append("Hallazgos:")
    if not findings:
        lines.append("  ninguno")
    else:
        for finding in sorted(findings, key=lambda item: (str(item.path), item.line, item.pattern)):
            lines.append(
                f"  {finding.pattern} {finding.path}:{finding.line} "
                f"valor={finding.preview}..."
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="raíz del repositorio (por defecto, el padre de scripts/)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="baseline JSON versionado; las rutas declaradas se agregan pero no bloquean",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    findings = scan_repository(root)
    active_findings = findings
    excluded_counts: Counter[str] = Counter()
    baseline_reasons: dict[str, str] = {}
    if args.baseline:
        baseline_path = args.baseline
        if not baseline_path.is_absolute():
            baseline_path = root / baseline_path
        exclusions = load_baseline(baseline_path)
        active_findings, excluded_counts = apply_baseline(findings, exclusions)
        baseline_reasons = {
            exclusion.category: exclusion.reason for exclusion in exclusions
        }
    print(format_report(active_findings, excluded_counts, baseline_reasons))
    return 1 if active_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
