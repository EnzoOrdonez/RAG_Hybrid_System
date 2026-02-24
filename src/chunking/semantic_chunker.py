"""
Semantic Chunker - Splits where semantic similarity between sentences drops.
Uses sentence-transformers for lightweight embedding during chunking.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import tiktoken

from src.ingestion.doc_parser import Chunk, Document

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Splits documents at semantic boundaries using embeddings."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        similarity_threshold: float = 0.5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Split document at semantic boundaries."""
        # Split into sentences
        sentences = self._split_sentences(doc.content)
        if len(sentences) <= 1:
            return self._single_chunk(doc)

        # Compute embeddings for all sentences
        embeddings = self.model.encode(sentences, show_progress_bar=False)

        # Compute similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-10
            )
            similarities.append(sim)

        # Find split points where similarity drops below threshold
        split_points = self._find_split_points(sentences, similarities)

        # Build chunks from split points
        chunks = self._build_chunks(doc, sentences, split_points)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving code blocks."""
        sentences = []
        current = []
        in_code = False

        for line in text.split("\n"):
            if line.strip().startswith("```"):
                in_code = not in_code
                current.append(line)
                if not in_code:  # End of code block
                    sentences.append("\n".join(current))
                    current = []
                continue

            if in_code:
                current.append(line)
                continue

            if not line.strip():
                if current:
                    sentences.append("\n".join(current))
                    current = []
                continue

            # Split by sentence-ending punctuation
            parts = []
            last = 0
            for i, char in enumerate(line):
                if char in ".!?" and i + 1 < len(line) and line[i + 1] == " ":
                    parts.append(line[last:i + 1])
                    last = i + 2
            if last < len(line):
                parts.append(line[last:])

            for part in parts:
                part = part.strip()
                if part:
                    if current:
                        current.append(part)
                    else:
                        current = [part]

        if current:
            sentences.append("\n".join(current))

        # Filter out empty sentences
        return [s for s in sentences if s.strip()]

    def _find_split_points(
        self, sentences: List[str], similarities: List[float]
    ) -> List[int]:
        """Find optimal split points respecting chunk size constraints."""
        split_points = [0]
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sent_tokens = self._token_count(sentence)
            current_tokens += sent_tokens

            # Check if we should split here
            should_split = False
            if current_tokens >= self.chunk_size:
                should_split = True
            elif (
                i < len(similarities)
                and similarities[i] < self.similarity_threshold
                and current_tokens >= self.chunk_size * 0.3  # Don't create tiny chunks
            ):
                should_split = True

            if should_split:
                split_points.append(i + 1)
                current_tokens = 0

        if split_points[-1] != len(sentences):
            split_points.append(len(sentences))

        return split_points

    def _build_chunks(
        self, doc: Document, sentences: List[str], split_points: List[int]
    ) -> List[Chunk]:
        """Build Chunk objects from split points."""
        chunks = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            text = "\n".join(sentences[start:end])

            if not text.strip():
                continue

            # Add overlap from previous chunk
            if i > 0 and self.overlap > 0:
                prev_start = split_points[i - 1]
                prev_sentences = sentences[prev_start:start]
                overlap_text = ""
                for s in reversed(prev_sentences):
                    candidate = s + "\n" + overlap_text if overlap_text else s
                    if self._token_count(candidate) <= self.overlap:
                        overlap_text = candidate
                    else:
                        break
                if overlap_text:
                    text = overlap_text + "\n" + text

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
                chunking_strategy="semantic",
                chunk_size_config=self.chunk_size,
                has_code="```" in text,
                has_table="|" in text and "---" in text,
                content_type="code" if "```" in text else "narrative",
            )
            chunks.append(chunk)

        for c in chunks:
            c.total_chunks = len(chunks)
        return chunks

    def _single_chunk(self, doc: Document) -> List[Chunk]:
        """Return the entire document as a single chunk."""
        if not doc.content.strip():
            return []
        return [Chunk(
            doc_id=doc.doc_id,
            text=doc.content,
            token_count=self._token_count(doc.content),
            cloud_provider=doc.cloud_provider,
            service_name=doc.service_name,
            doc_type=doc.doc_type,
            url_source=doc.url_source,
            chunk_index=0,
            total_chunks=1,
            chunking_strategy="semantic",
            chunk_size_config=self.chunk_size,
        )]

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
        chunker = SemanticChunker(chunk_size=size, overlap=overlap)
        chunks = chunker.chunk_documents(documents)

        size_dir = output_dir / f"size_{size}"
        size_dir.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            path = size_dir / f"{chunk.chunk_id}.json"
            path.write_text(chunk.model_dump_json(indent=2), encoding="utf-8")

        logger.info("Semantic chunker (size=%d): %d chunks", size, len(chunks))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic chunking")
    parser.add_argument("--input", default="data/processed", help="Input directory")
    parser.add_argument("--output", default="data/chunks/semantic", help="Output directory")
    parser.add_argument("--sizes", default="300,500,700", help="Comma-separated chunk sizes")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap in tokens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sizes = [int(s) for s in args.sizes.split(",")]
    process_directory(Path(args.input), Path(args.output), sizes, args.overlap)


if __name__ == "__main__":
    main()
