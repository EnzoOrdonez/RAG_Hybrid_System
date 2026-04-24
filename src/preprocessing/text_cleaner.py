"""
Text Cleaner - Post-parsing text cleanup for cloud documentation.
Removes boilerplate, normalizes whitespace, preserves technical content.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path

from src.ingestion.doc_parser import Document

logger = logging.getLogger(__name__)

BOILERPLATE_PATTERNS = [
    re.compile(r"Was this (page|helpful|article).*?(Yes|No).*?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"Did this page help you\?.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"Feedback.*?(thumbs up|thumbs down|helpful)", re.IGNORECASE),
    re.compile(r"^\s*Send feedback\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Stay organized with collections\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Save and categorize content based on your preferences\.?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"Edit this page on GitHub\.?", re.IGNORECASE),
    re.compile(r"View.*?on GitHub\.?", re.IGNORECASE),
    re.compile(r"Suggest an edit.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(Home|Docs)\s*[>/]\s*(.*?[>/]\s*){2,}.*$", re.MULTILINE),
    re.compile(r"^Last (updated|modified|reviewed):\s*.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Article\s*\|\s*\d{2}/\d{2}/\d{4}.*$", re.MULTILINE),
    re.compile(r"^(Previous|Next|Back to top)\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*On this page:?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^In this (article|section|topic):?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"(Accept|Reject)\s*(all\s*)?(cookies|tracking)", re.IGNORECASE),
    re.compile(r"^Javascript is disabled.*$", re.MULTILINE),
    re.compile(r"^Please refer to your browser.*$", re.MULTILINE),
    re.compile(r"^Choose a different version.*$", re.MULTILINE),
    re.compile(r"^\s*Share\s*(this\s*)?(page|article|post)?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*(Twitter|Facebook|LinkedIn|Copy link)\s*$", re.MULTILINE | re.IGNORECASE),
]


class TextCleaner:
    """Cleans text content from parsed documents."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        prep_cfg = self.config.get("preprocessing", {})
        self.normalize_unicode = prep_cfg.get("normalize_unicode", True)
        self.max_newlines = prep_cfg.get("max_consecutive_newlines", 2)
        self.min_para_length = prep_cfg.get("min_paragraph_length", 20)

    def clean_document(self, doc: Document) -> Document:
        """Clean a document's content in place."""
        doc.content = self.clean_text(doc.content)
        doc.word_count = len(doc.content.split())
        doc.char_count = len(doc.content)
        self._clean_sections(doc.sections)
        return doc

    def clean_text(self, text: str) -> str:
        """Apply all cleaning steps to text."""
        if not text:
            return text

        text = self._remove_boilerplate(text)
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        text = self._normalize_whitespace(text)
        text = self._limit_newlines(text)
        text = self._remove_duplicate_paragraphs(text)
        return text.strip()

    def _clean_heading(self, heading: str) -> str:
        """Clean section titles without applying paragraph-level cleanup."""
        if not heading:
            return heading
        heading = self._remove_boilerplate(heading)
        if self.normalize_unicode:
            heading = self._normalize_unicode(heading)
        heading = self._normalize_whitespace(heading)
        heading = heading.replace("\n", " ")
        heading = re.sub(r"\s+", " ", heading)
        return heading.strip()

    def _remove_boilerplate(self, text: str) -> str:
        for pattern in BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)
        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode, preserving technical symbols."""
        text = unicodedata.normalize("NFKC", text)
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
            "\u00a0": " ",
            "\u200b": "",
            "\ufeff": "",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize spaces and tabs, preserve newlines."""
        text = text.replace("\t", "    ")
        lines = [line.rstrip() for line in text.split("\n")]
        result = []
        in_code = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
            if not in_code:
                line = re.sub(r"  +", " ", line)
            result.append(line)
        return "\n".join(result)

    def _limit_newlines(self, text: str) -> str:
        pattern = r"\n{" + str(self.max_newlines + 1) + r",}"
        replacement = "\n" * self.max_newlines
        return re.sub(pattern, replacement, text)

    def _remove_duplicate_paragraphs(self, text: str) -> str:
        """Remove exact duplicate paragraphs within the same document."""
        paragraphs = text.split("\n\n")
        seen = set()
        unique = []
        for paragraph in paragraphs:
            stripped = paragraph.strip()
            if not stripped:
                unique.append(paragraph)
                continue
            if len(stripped) < self.min_para_length:
                unique.append(paragraph)
                continue
            if stripped in seen:
                continue
            seen.add(stripped)
            unique.append(paragraph)
        return "\n\n".join(unique)

    def _clean_sections(self, sections):
        """Recursively clean section contents and headings."""
        for section in sections:
            section.title = self._clean_heading(section.title)
            section.content = self.clean_text(section.content)
            self._clean_sections(section.subsections)

    def process_directory(self, input_dir: Path, output_dir: Path = None):
        """Clean all documents in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir

        json_files = list(input_dir.rglob("*.json"))
        cleaned = 0
        for json_file in json_files:
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                doc = Document(**data)
                doc = self.clean_document(doc)
                out_path = output_dir / json_file.relative_to(input_dir)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
                cleaned += 1
            except Exception as exc:
                logger.warning("Failed to clean %s: %s", json_file, exc)

        logger.info("Cleaned %d/%d documents", cleaned, len(json_files))
        return cleaned


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Clean preprocessed documents")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    project_root = Path(__file__).parent.parent.parent
    with open(project_root / "config/config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    cleaner = TextCleaner(config)
    cleaner.process_directory(Path(args.input), Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
