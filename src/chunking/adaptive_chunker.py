"""
Adaptive Chunker - Main contribution of the thesis.
Combines hierarchical (heading-based) + semantic splitting with special
rules for technical documentation (code blocks, tables, numbered lists).

Algorithm:
  1st pass: Split by headings (H1-H4)
  2nd pass: Adjust sizes - subdivide large sections semantically, merge small ones
  Special rules: Atomic code blocks, atomic tables, heading path prepending
  Smart overlap: Between narrative chunks only, not across content types
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tiktoken

from src.ingestion.doc_parser import Chunk, Document, Section

logger = logging.getLogger(__name__)


class AdaptiveChunker:
    """Thesis-proposed chunker: hierarchical + semantic with domain rules."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        min_chunk_size: int = 50,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_name: str = "cl100k_base",
        prepend_heading_path: bool = True,
        preserve_code_blocks: bool = True,
        max_heading_depth: int = 4,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.prepend_heading = prepend_heading_path
        self.preserve_code = preserve_code_blocks
        self.max_depth = max_heading_depth
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # ============================================================
    # Main entry point
    # ============================================================

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Adaptive chunking: hierarchical first, then semantic refinement."""

        # PASS 1: Split by heading structure
        raw_sections = self._first_pass_headings(doc)

        # PASS 2: Adjust sizes (split large, merge small)
        adjusted = self._second_pass_adjust(raw_sections)

        # PASS 3: Apply special rules and build final chunks
        chunks = self._build_final_chunks(doc, adjusted)

        return chunks

    # ============================================================
    # PASS 1: Heading-based splitting
    # ============================================================

    def _first_pass_headings(self, doc: Document) -> List[dict]:
        """Split document by headings into candidate chunks."""
        if doc.sections:
            sections = []
            self._flatten_sections(doc.sections, [], sections)
            if sections:
                return sections

        # Fallback: split by heading patterns in raw text
        return self._split_by_heading_patterns(doc.content)

    def _flatten_sections(
        self, sections: List[Section], parent_path: List[str], result: list
    ):
        """Recursively flatten sections preserving hierarchy."""
        for section in sections:
            if section.level > self.max_depth:
                continue

            path = parent_path + [section.title]
            content = section.content.strip()

            # Extract code blocks and tables as separate items
            if content:
                parts = self._extract_atomic_units(content)
                for part in parts:
                    result.append({
                        "heading_path": path,
                        "content": part["text"],
                        "level": section.level,
                        "content_type": part["type"],
                    })

            if section.subsections:
                self._flatten_sections(section.subsections, path, result)

    def _split_by_heading_patterns(self, text: str) -> List[dict]:
        """Fallback: split raw text by Markdown heading patterns."""
        heading_re = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
        sections = []
        matches = list(heading_re.finditer(text))

        if not matches:
            # No headings at all
            parts = self._extract_atomic_units(text)
            return [
                {"heading_path": [], "content": p["text"], "level": 0, "content_type": p["type"]}
                for p in parts
            ]

        # Content before first heading
        if matches[0].start() > 0:
            pre_content = text[:matches[0].start()].strip()
            if pre_content:
                for p in self._extract_atomic_units(pre_content):
                    sections.append({
                        "heading_path": [],
                        "content": p["text"],
                        "level": 0,
                        "content_type": p["type"],
                    })

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            if content:
                for p in self._extract_atomic_units(content):
                    sections.append({
                        "heading_path": [title],
                        "content": p["text"],
                        "level": level,
                        "content_type": p["type"],
                    })

        return sections

    def _extract_atomic_units(self, text: str) -> List[dict]:
        """Extract code blocks and tables as atomic units, narrative as the rest."""
        parts = []
        # Split by code blocks
        code_re = re.compile(r'(```[\w]*\n.*?```)', re.DOTALL)
        segments = code_re.split(text)

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if seg.startswith("```"):
                parts.append({"text": seg, "type": "code"})
            else:
                # Check for tables
                table_re = re.compile(
                    r'(\|.+\|\n\|[-:\s|]+\|\n(?:\|.+\|\n?)+)',
                    re.MULTILINE,
                )
                table_parts = table_re.split(seg)
                for tp in table_parts:
                    tp = tp.strip()
                    if not tp:
                        continue
                    if tp.startswith("|") and "---" in tp:
                        parts.append({"text": tp, "type": "table"})
                    else:
                        parts.append({"text": tp, "type": "narrative"})

        return parts if parts else [{"text": text, "type": "narrative"}]

    # ============================================================
    # PASS 2: Size adjustment
    # ============================================================

    def _second_pass_adjust(self, sections: List[dict]) -> List[dict]:
        """Adjust chunk sizes: split large sections semantically, merge small ones."""
        adjusted = []

        for section in sections:
            tokens = self._token_count(section["content"])

            if tokens > self.chunk_size:
                # Split large sections
                if section["content_type"] == "narrative":
                    # Use semantic splitting for narrative text
                    sub_sections = self._semantic_split(
                        section["content"],
                        section["heading_path"],
                        section["level"],
                    )
                    adjusted.extend(sub_sections)
                elif section["content_type"] in ("code", "table"):
                    # Code/table blocks are atomic - keep even if large
                    adjusted.append(section)
                else:
                    adjusted.append(section)
            elif tokens < self.min_chunk_size:
                # Mark for merging
                section["_merge"] = True
                adjusted.append(section)
            else:
                adjusted.append(section)

        # Merge small consecutive sections at the same heading level
        merged = self._merge_small_sections(adjusted)
        return merged

    def _semantic_split(
        self, text: str, heading_path: List[str], level: int
    ) -> List[dict]:
        """Split a large narrative section at semantic boundaries."""
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [{"heading_path": heading_path, "content": text, "level": level, "content_type": "narrative"}]

        # Compute embeddings
        try:
            embeddings = self.model.encode(sentences, show_progress_bar=False)
        except Exception:
            # Fallback to token-based splitting
            return self._token_split(text, heading_path, level)

        # Find drop points in similarity
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-10
            ))
            similarities.append(sim)

        # Split at low-similarity points, respecting chunk_size
        result = []
        current_sentences = []
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sent_tokens = self._token_count(sentence)
            current_sentences.append(sentence)
            current_tokens += sent_tokens

            should_split = False
            if current_tokens >= self.chunk_size:
                should_split = True
            elif (
                i < len(similarities)
                and similarities[i] < self.similarity_threshold
                and current_tokens >= self.min_chunk_size
            ):
                should_split = True

            if should_split:
                chunk_text = "\n".join(current_sentences)
                result.append({
                    "heading_path": heading_path,
                    "content": chunk_text,
                    "level": level,
                    "content_type": "narrative",
                })
                current_sentences = []
                current_tokens = 0

        if current_sentences:
            chunk_text = "\n".join(current_sentences)
            result.append({
                "heading_path": heading_path,
                "content": chunk_text,
                "level": level,
                "content_type": "narrative",
            })

        return result

    def _token_split(
        self, text: str, heading_path: List[str], level: int
    ) -> List[dict]:
        """Fallback: split by tokens when semantic model unavailable."""
        tokens = self.tokenizer.encode(text)
        result = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = self.tokenizer.decode(tokens[start:end])
            result.append({
                "heading_path": heading_path,
                "content": chunk_text,
                "level": level,
                "content_type": "narrative",
            })
            start = end
        return result

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving numbered lists."""
        lines = text.split("\n")
        sentences = []
        current = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    sentences.append("\n".join(current))
                    current = []
                continue

            # Check for numbered list items (keep together)
            if re.match(r'^\d+[\.\)]\s', stripped):
                current.append(line)
                continue

            # Check for bullet points
            if re.match(r'^[-*+]\s', stripped):
                current.append(line)
                continue

            # Regular text - split by sentence endings
            if stripped.endswith(('.', '!', '?', ':')):
                current.append(line)
                sentences.append("\n".join(current))
                current = []
            else:
                current.append(line)

        if current:
            sentences.append("\n".join(current))

        return [s for s in sentences if s.strip()]

    def _merge_small_sections(self, sections: List[dict]) -> List[dict]:
        """Merge small consecutive sections at the same heading level."""
        if not sections:
            return sections

        merged = []
        i = 0
        while i < len(sections):
            current = sections[i]

            if current.get("_merge") and i + 1 < len(sections):
                # Try to merge with next section
                nxt = sections[i + 1]
                if nxt.get("_merge") or (
                    nxt.get("level", 0) == current.get("level", 0)
                    and nxt.get("content_type") == current.get("content_type")
                ):
                    combined_text = current["content"] + "\n\n" + nxt["content"]
                    if self._token_count(combined_text) <= self.chunk_size:
                        merged_section = {
                            "heading_path": current["heading_path"],
                            "content": combined_text,
                            "level": current["level"],
                            "content_type": current["content_type"],
                        }
                        # Check if we can merge more
                        sections[i + 1] = merged_section
                        i += 1
                        continue

            # Remove merge flag
            current.pop("_merge", None)
            merged.append(current)
            i += 1

        return merged

    # ============================================================
    # PASS 3: Build final chunks with context and overlap
    # ============================================================

    def _build_final_chunks(
        self, doc: Document, sections: List[dict]
    ) -> List[Chunk]:
        """Build final Chunk objects with heading path prepending and smart overlap."""
        chunks = []
        prev_type = None

        for i, section in enumerate(sections):
            text = section["content"]
            heading_path = section["heading_path"]
            content_type = section.get("content_type", "narrative")

            # Prepend heading path as context window
            if self.prepend_heading and heading_path:
                provider_prefix = doc.cloud_provider.upper()
                if doc.service_name:
                    provider_prefix += f" > {doc.service_name}"
                full_path = " > ".join([provider_prefix] + heading_path)
                text = f"[{full_path}] {text}"

            # Smart overlap: only between same-type narrative chunks
            if (
                i > 0
                and self.overlap > 0
                and content_type == "narrative"
                and prev_type == "narrative"
            ):
                prev_text = sections[i - 1]["content"]
                prev_tokens = self.tokenizer.encode(prev_text)
                overlap_tokens = prev_tokens[-self.overlap:]
                overlap_text = self.tokenizer.decode(overlap_tokens)
                text = overlap_text + " " + text

            token_count = self._token_count(text)

            chunk = Chunk(
                doc_id=doc.doc_id,
                text=text,
                token_count=token_count,
                cloud_provider=doc.cloud_provider,
                service_name=doc.service_name,
                doc_type=doc.doc_type,
                url_source=doc.url_source,
                section_hierarchy=heading_path,
                heading_path=" > ".join(heading_path) if heading_path else "",
                chunk_index=i,
                chunking_strategy="adaptive",
                chunk_size_config=self.chunk_size,
                has_code=content_type == "code" or "```" in text,
                has_table=content_type == "table",
                content_type=content_type,
            )
            chunks.append(chunk)
            prev_type = content_type

        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            try:
                all_chunks.extend(self.chunk_document(doc))
            except Exception as e:
                logger.warning("Failed to chunk doc %s: %s", doc.doc_id, e)
        return all_chunks


def process_directory(
    input_dir: Path, output_dir: Path, sizes: List[int] = None, overlap: int = 50
):
    sizes = sizes or [300, 500, 700]
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    documents = []
    for jf in input_dir.rglob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            documents.append(Document(**data))
        except Exception as e:
            logger.warning("Failed to load %s: %s", jf, e)

    if not documents:
        return

    for size in sizes:
        chunker = AdaptiveChunker(chunk_size=size, overlap=overlap)
        chunks = chunker.chunk_documents(documents)

        size_dir = output_dir / f"size_{size}"
        size_dir.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            path = size_dir / f"{chunk.chunk_id}.json"
            path.write_text(chunk.model_dump_json(indent=2), encoding="utf-8")

        logger.info("Adaptive chunker (size=%d): %d chunks", size, len(chunks))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive chunking (thesis proposal)")
    parser.add_argument("--input", default="data/processed", help="Input directory")
    parser.add_argument("--output", default="data/chunks/adaptive", help="Output directory")
    parser.add_argument("--sizes", default="300,500,700", help="Comma-separated chunk sizes")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap in tokens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sizes = [int(s) for s in args.sizes.split(",")]
    process_directory(Path(args.input), Path(args.output), sizes, args.overlap)


if __name__ == "__main__":
    main()
