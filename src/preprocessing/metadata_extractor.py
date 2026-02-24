"""
Metadata Extractor - Extracts and enriches document metadata.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.ingestion.doc_parser import Document

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts enriched metadata from parsed documents."""

    # Patterns for detecting content characteristics
    CODE_INDICATORS = re.compile(
        r'(```|<code>|<pre>|\$ |>>> |#!/|import |from .+ import |'
        r'def |class |function |const |var |let |public |private )',
        re.IGNORECASE,
    )
    API_INDICATORS = re.compile(
        r'(endpoint|request|response|HTTP|GET|POST|PUT|DELETE|'
        r'API|REST|gRPC|status code|header|payload)',
        re.IGNORECASE,
    )
    TUTORIAL_INDICATORS = re.compile(
        r'(step \d|tutorial|walkthrough|getting started|'
        r'how to|in this guide|prerequisites|before you begin)',
        re.IGNORECASE,
    )

    def extract(self, doc: Document) -> Dict:
        """Extract enriched metadata from a document."""
        metadata = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "cloud_provider": doc.cloud_provider,
            "service_name": doc.service_name,
            "service_category": doc.service_category,
            "doc_type": doc.doc_type,
            "url_source": doc.url_source,
            "word_count": doc.word_count,
            "char_count": doc.char_count,
            "has_code": doc.has_code,
            "has_tables": doc.has_tables,
            "heading_count": doc.heading_count,
            "code_block_count": len(doc.code_blocks),
            "table_count": len(doc.tables),
            "section_count": len(doc.sections),
        }

        # Detect content characteristics
        content = doc.content
        metadata["code_density"] = self._compute_code_density(doc)
        metadata["reading_level"] = self._estimate_reading_level(content)
        metadata["languages_detected"] = self._detect_languages(doc)
        metadata["complexity"] = self._estimate_complexity(doc)

        return metadata

    def _compute_code_density(self, doc: Document) -> float:
        """Compute ratio of code to total content."""
        if doc.char_count == 0:
            return 0.0
        code_chars = sum(len(cb.code) for cb in doc.code_blocks)
        return round(code_chars / doc.char_count, 3)

    def _estimate_reading_level(self, text: str) -> str:
        """Simple reading level estimation."""
        words = text.split()
        if not words:
            return "unknown"
        sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
        avg_words_per_sentence = len(words) / sentences
        avg_word_length = sum(len(w) for w in words) / len(words)

        if avg_words_per_sentence > 25 or avg_word_length > 6:
            return "advanced"
        elif avg_words_per_sentence > 15 or avg_word_length > 5:
            return "intermediate"
        return "beginner"

    def _detect_languages(self, doc: Document) -> List[str]:
        """Detect programming languages from code blocks."""
        langs = set()
        for cb in doc.code_blocks:
            if cb.language:
                langs.add(cb.language.lower())
        return sorted(langs)

    def _estimate_complexity(self, doc: Document) -> str:
        """Estimate document complexity."""
        score = 0
        if doc.has_code:
            score += 1
        if doc.has_tables:
            score += 1
        if doc.heading_count > 10:
            score += 1
        if doc.word_count > 3000:
            score += 1
        if len(doc.code_blocks) > 5:
            score += 1

        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        return "low"

    def process_directory(self, input_dir: Path):
        """Process all documents and save enriched metadata."""
        input_dir = Path(input_dir)
        json_files = list(input_dir.rglob("*.json"))
        processed = 0
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                doc = Document(**data)
                meta = self.extract(doc)
                data["enriched_metadata"] = meta
                jf.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
                processed += 1
            except Exception as e:
                logger.warning("Failed to extract metadata from %s: %s", jf, e)

        logger.info("Extracted metadata for %d/%d documents", processed, len(json_files))
