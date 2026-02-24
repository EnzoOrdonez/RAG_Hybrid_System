"""
Recursive Chunker - Splits by hierarchical separators.
Uses progressively finer separators: paragraph -> line -> sentence -> word.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import tiktoken

from src.ingestion.doc_parser import Chunk, Document

logger = logging.getLogger(__name__)


class RecursiveChunker:
    """Splits documents using recursive separators."""

    DEFAULT_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", " "]

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: Optional[List[str]] = None,
        tokenizer_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Split a document using recursive separators."""
        raw_chunks = self._recursive_split(doc.content, self.separators)

        # Merge small chunks and add overlap
        merged = self._merge_with_overlap(raw_chunks)

        chunks = []
        for i, text in enumerate(merged):
            token_count = self._token_count(text)
            chunk = Chunk(
                doc_id=doc.doc_id,
                text=text,
                token_count=token_count,
                cloud_provider=doc.cloud_provider,
                service_name=doc.service_name,
                doc_type=doc.doc_type,
                url_source=doc.url_source,
                chunk_index=i,
                chunking_strategy="recursive",
                chunk_size_config=self.chunk_size,
                has_code="```" in text,
                has_table="|" in text and "---" in text,
                content_type=self._detect_content_type(text),
            )
            chunks.append(chunk)

        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the separator hierarchy."""
        if not text.strip():
            return []

        # If text fits in one chunk, return it
        if self._token_count(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            # No more separators - force split by tokens
            return self._force_split(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(sep)
        if len(parts) == 1:
            # This separator didn't split anything, try the next
            return self._recursive_split(text, remaining_seps)

        results = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if self._token_count(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    results.append(current.strip())
                # If the part itself is too large, recursively split it
                if self._token_count(part) > self.chunk_size:
                    results.extend(self._recursive_split(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current.strip():
            results.append(current.strip())

        return results

    def _force_split(self, text: str) -> List[str]:
        """Force split text by token count when no separators work."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = self.tokenizer.decode(tokens[start:end])
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            start = end
        return chunks

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if not chunks or self.overlap <= 0:
            return chunks

        result = []
        for i, chunk in enumerate(chunks):
            if i > 0 and self.overlap > 0:
                # Get last N tokens of previous chunk
                prev_tokens = self.tokenizer.encode(chunks[i - 1])
                overlap_tokens = prev_tokens[-self.overlap:]
                overlap_text = self.tokenizer.decode(overlap_tokens)
                chunk = overlap_text + " " + chunk

            result.append(chunk)

        return result

    def _detect_content_type(self, text: str) -> str:
        code_indicators = text.count("```") + text.count("    ")
        if code_indicators > 3:
            return "code"
        elif code_indicators > 0:
            return "mixed"
        return "narrative"

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
        chunker = RecursiveChunker(chunk_size=size, overlap=overlap)
        chunks = chunker.chunk_documents(documents)

        size_dir = output_dir / f"size_{size}"
        size_dir.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            path = size_dir / f"{chunk.chunk_id}.json"
            path.write_text(chunk.model_dump_json(indent=2), encoding="utf-8")

        logger.info("Recursive chunker (size=%d): %d chunks", size, len(chunks))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Recursive chunking")
    parser.add_argument("--input", default="data/processed", help="Input directory")
    parser.add_argument("--output", default="data/chunks/recursive", help="Output directory")
    parser.add_argument("--sizes", default="300,500,700", help="Comma-separated chunk sizes")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap in tokens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sizes = [int(s) for s in args.sizes.split(",")]
    process_directory(Path(args.input), Path(args.output), sizes, args.overlap)


if __name__ == "__main__":
    main()
