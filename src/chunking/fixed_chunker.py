"""
Fixed-size Chunker - Baseline strategy.
Splits text into fixed token-size chunks with overlap.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import tiktoken

from src.ingestion.doc_parser import Chunk, Document

logger = logging.getLogger(__name__)


class FixedChunker:
    """Splits documents into fixed-size chunks by token count."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50, tokenizer_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Split a document into fixed-size chunks."""
        tokens = self.tokenizer.encode(doc.content)
        if not tokens:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunk = Chunk(
                doc_id=doc.doc_id,
                text=chunk_text,
                token_count=len(chunk_tokens),
                cloud_provider=doc.cloud_provider,
                service_name=doc.service_name,
                doc_type=doc.doc_type,
                url_source=doc.url_source,
                chunk_index=chunk_index,
                chunking_strategy="fixed",
                chunk_size_config=self.chunk_size,
                has_code="```" in chunk_text or "    " in chunk_text,
                has_table="|" in chunk_text and "---" in chunk_text,
                content_type=self._detect_content_type(chunk_text),
            )
            chunks.append(chunk)
            chunk_index += 1

            # Move forward with overlap
            start = end - self.overlap if end < len(tokens) else end

        # Set total_chunks
        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks

    def _detect_content_type(self, text: str) -> str:
        code_lines = sum(1 for line in text.split("\n") if line.strip().startswith(("```", "    ", ">>> ", "$ ")))
        total_lines = max(1, len(text.split("\n")))
        code_ratio = code_lines / total_lines

        if code_ratio > 0.5:
            return "code"
        elif code_ratio > 0.2:
            return "mixed"
        return "narrative"

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


def process_directory(
    input_dir: Path,
    output_dir: Path,
    sizes: List[int] = None,
    overlap: int = 50,
):
    """Process documents and generate chunks for given sizes."""
    sizes = sizes or [300, 500, 700]
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Load documents
    documents = []
    for jf in input_dir.rglob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            documents.append(Document(**data))
        except Exception as e:
            logger.warning("Failed to load %s: %s", jf, e)

    if not documents:
        logger.warning("No documents found in %s", input_dir)
        return

    for size in sizes:
        chunker = FixedChunker(chunk_size=size, overlap=overlap)
        chunks = chunker.chunk_documents(documents)

        size_dir = output_dir / f"size_{size}"
        size_dir.mkdir(parents=True, exist_ok=True)

        for chunk in chunks:
            path = size_dir / f"{chunk.chunk_id}.json"
            path.write_text(chunk.model_dump_json(indent=2), encoding="utf-8")

        logger.info("Fixed chunker (size=%d): %d chunks from %d documents", size, len(chunks), len(documents))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fixed-size chunking")
    parser.add_argument("--input", default="data/processed", help="Input directory")
    parser.add_argument("--output", default="data/chunks/fixed", help="Output directory")
    parser.add_argument("--sizes", default="300,500,700", help="Comma-separated chunk sizes")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap in tokens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sizes = [int(s) for s in args.sizes.split(",")]
    process_directory(Path(args.input), Path(args.output), sizes, args.overlap)


if __name__ == "__main__":
    main()
