"""
Hierarchical Chunker - Respects document heading structure (H1-H4).
Each section under a heading becomes a chunk candidate.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import tiktoken

from src.ingestion.doc_parser import Chunk, Document, Section

logger = logging.getLogger(__name__)


class HierarchicalChunker:
    """Splits documents respecting heading hierarchy."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        max_depth: int = 4,
        tokenizer_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_depth = max_depth
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Split document following heading structure."""
        if not doc.sections:
            # No headings: fall back to simple splitting
            return self._fallback_chunk(doc)

        # Flatten sections into (heading_path, content) pairs
        section_texts = []
        self._flatten_sections(doc.sections, [], section_texts)

        # Build chunks from sections
        chunks = self._build_chunks_from_sections(doc, section_texts)
        return chunks

    def _flatten_sections(
        self,
        sections: List[Section],
        parent_path: List[str],
        result: List[dict],
    ):
        """Recursively flatten sections into (heading_path, content) pairs."""
        for section in sections:
            if section.level > self.max_depth:
                continue

            path = parent_path + [section.title]
            content = section.content.strip()

            if content:
                result.append({
                    "heading_path": path,
                    "content": content,
                    "level": section.level,
                })

            if section.subsections:
                self._flatten_sections(section.subsections, path, result)

    def _build_chunks_from_sections(
        self, doc: Document, section_texts: List[dict]
    ) -> List[Chunk]:
        """Build chunks from flattened sections, merging small ones."""
        chunks = []
        current_text = ""
        current_path = []
        chunk_index = 0

        for section in section_texts:
            heading_path = section["heading_path"]
            content = section["content"]
            section_text = content

            candidate = current_text + "\n\n" + section_text if current_text else section_text
            candidate_tokens = self._token_count(candidate)

            if candidate_tokens <= self.chunk_size:
                # Merge with current
                current_text = candidate
                if not current_path:
                    current_path = heading_path
            else:
                # Emit current chunk
                if current_text.strip():
                    chunk = self._create_chunk(
                        doc, current_text.strip(), current_path, chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Check if section_text itself fits
                if self._token_count(section_text) <= self.chunk_size:
                    current_text = section_text
                    current_path = heading_path
                else:
                    # Section is too large - split by tokens
                    sub_chunks = self._split_large_section(
                        doc, section_text, heading_path, chunk_index
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    current_text = ""
                    current_path = []

        # Flush remaining
        if current_text.strip():
            chunk = self._create_chunk(
                doc, current_text.strip(), current_path, chunk_index
            )
            chunks.append(chunk)

        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks

    def _split_large_section(
        self, doc: Document, text: str, heading_path: List[str], start_index: int
    ) -> List[Chunk]:
        """Split an oversized section into chunks."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        idx = start_index

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = self.tokenizer.decode(tokens[start:end])
            chunk = self._create_chunk(doc, chunk_text, heading_path, idx)
            chunks.append(chunk)
            idx += 1
            start = end - self.overlap if end < len(tokens) else end

        return chunks

    def _create_chunk(
        self, doc: Document, text: str, heading_path: List[str], index: int
    ) -> Chunk:
        """Create a Chunk object."""
        path_str = " > ".join(heading_path) if heading_path else ""
        return Chunk(
            doc_id=doc.doc_id,
            text=text,
            token_count=self._token_count(text),
            cloud_provider=doc.cloud_provider,
            service_name=doc.service_name,
            doc_type=doc.doc_type,
            url_source=doc.url_source,
            section_hierarchy=heading_path,
            heading_path=path_str,
            chunk_index=index,
            chunking_strategy="hierarchical",
            chunk_size_config=self.chunk_size,
            has_code="```" in text,
            has_table="|" in text and "---" in text,
            content_type=self._detect_content_type(text),
        )

    def _detect_content_type(self, text: str) -> str:
        if "```" in text:
            non_code = text.split("```")
            code_len = sum(len(non_code[i]) for i in range(1, len(non_code), 2)) if len(non_code) > 1 else 0
            if code_len > len(text) * 0.5:
                return "code"
            return "mixed"
        return "narrative"

    def _fallback_chunk(self, doc: Document) -> List[Chunk]:
        """Fallback: split by paragraphs when no headings exist."""
        paragraphs = doc.content.split("\n\n")
        chunks = []
        current = ""
        idx = 0

        for para in paragraphs:
            candidate = current + "\n\n" + para if current else para
            if self._token_count(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunk = self._create_chunk(doc, current.strip(), [], idx)
                    chunks.append(chunk)
                    idx += 1
                current = para

        if current.strip():
            chunk = self._create_chunk(doc, current.strip(), [], idx)
            chunks.append(chunk)

        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
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
        chunker = HierarchicalChunker(chunk_size=size, overlap=overlap)
        chunks = chunker.chunk_documents(documents)

        size_dir = output_dir / f"size_{size}"
        size_dir.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            path = size_dir / f"{chunk.chunk_id}.json"
            path.write_text(chunk.model_dump_json(indent=2), encoding="utf-8")

        logger.info("Hierarchical chunker (size=%d): %d chunks", size, len(chunks))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical chunking")
    parser.add_argument("--input", default="data/processed", help="Input directory")
    parser.add_argument("--output", default="data/chunks/hierarchical", help="Output directory")
    parser.add_argument("--sizes", default="300,500,700", help="Comma-separated chunk sizes")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap in tokens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sizes = [int(s) for s in args.sizes.split(",")]
    process_directory(Path(args.input), Path(args.output), sizes, args.overlap)


if __name__ == "__main__":
    main()
