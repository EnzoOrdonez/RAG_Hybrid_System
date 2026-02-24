"""
Terminology Normalizer - Detects cloud terms and adds cross-provider metadata.
Does NOT replace text - ADDS metadata for query expansion.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

from src.ingestion.doc_parser import Chunk, Document

logger = logging.getLogger(__name__)


class TerminologyNormalizer:
    """Scans documents/chunks for cloud terminology and enriches metadata."""

    def __init__(self, mappings_path: str = "config/terminology_mappings.yaml"):
        project_root = Path(__file__).parent.parent.parent
        with open(project_root / mappings_path, encoding="utf-8") as f:
            self.mappings = yaml.safe_load(f)

        # Build lookup structures
        self.term_to_concept: Dict[str, str] = {}  # "S3" -> "object_storage"
        self.term_to_provider: Dict[str, str] = {}  # "S3" -> "aws"
        self.concept_to_terms: Dict[str, Dict[str, List[str]]] = {}
        self.acronyms: Dict[str, str] = self.mappings.get("acronyms", {})

        # Inverted index: term -> set of chunk_ids
        self.inverted_index: Dict[str, Set[str]] = {}

        self._build_lookups()

    def _build_lookups(self):
        """Build fast lookup tables from terminology mappings."""
        for category, concepts in self.mappings.items():
            if category == "acronyms":
                continue
            if not isinstance(concepts, dict):
                continue
            for concept_name, providers in concepts.items():
                if not isinstance(providers, dict):
                    continue
                self.concept_to_terms[concept_name] = {}
                for provider, terms in providers.items():
                    if not isinstance(terms, list):
                        continue
                    self.concept_to_terms[concept_name][provider] = terms
                    for term in terms:
                        term_lower = term.lower()
                        self.term_to_concept[term_lower] = concept_name
                        if provider != "generic":
                            self.term_to_provider[term_lower] = provider

        # Compile regex patterns for efficient matching (longest match first)
        all_terms = sorted(self.term_to_concept.keys(), key=len, reverse=True)
        # Escape for regex and build pattern
        escaped = [re.escape(t) for t in all_terms]
        if escaped:
            self.term_pattern = re.compile(
                r'\b(' + '|'.join(escaped) + r')\b',
                re.IGNORECASE,
            )
        else:
            self.term_pattern = None

    def normalize_document(self, doc: Document) -> Dict:
        """Analyze a document and return terminology metadata."""
        text = doc.content.lower()
        return self._analyze_text(text)

    def normalize_chunk(self, chunk: Chunk) -> Chunk:
        """Enrich a chunk with terminology metadata."""
        analysis = self._analyze_text(chunk.text.lower())
        chunk.normalized_terms = analysis["normalized_terms"]
        chunk.detected_siglas = analysis["detected_siglas"]
        chunk.cross_cloud_equivalences = analysis["cross_cloud_equivalences"]

        # Update inverted index
        for term in analysis["all_detected_terms"]:
            if term not in self.inverted_index:
                self.inverted_index[term] = set()
            self.inverted_index[term].add(chunk.chunk_id)

        return chunk

    def _analyze_text(self, text: str) -> Dict:
        """Analyze text for cloud terminology."""
        if not self.term_pattern:
            return {
                "normalized_terms": [],
                "detected_siglas": [],
                "cross_cloud_equivalences": {},
                "all_detected_terms": [],
            }

        # Find all matching terms
        matches = set()
        for m in self.term_pattern.finditer(text):
            matches.add(m.group(0).lower())

        # Map to concepts
        concepts = set()
        detected_siglas = []
        all_terms = []
        cross_cloud = {}

        for term in matches:
            all_terms.append(term)
            concept = self.term_to_concept.get(term, "")
            if concept:
                concepts.add(concept)
                # Build cross-cloud equivalences
                if concept in self.concept_to_terms:
                    equivs = {}
                    for prov, prov_terms in self.concept_to_terms[concept].items():
                        if prov != "generic":
                            equivs[prov] = prov_terms
                    if equivs:
                        cross_cloud[term] = []
                        for prov, prov_terms in equivs.items():
                            for pt in prov_terms:
                                if pt.lower() != term:
                                    cross_cloud[term].append(pt)

            # Check if it's an acronym
            upper_term = term.upper()
            if upper_term in self.acronyms:
                detected_siglas.append(upper_term)

        return {
            "normalized_terms": sorted(concepts),
            "detected_siglas": sorted(set(detected_siglas)),
            "cross_cloud_equivalences": cross_cloud,
            "all_detected_terms": all_terms,
        }

    def get_query_expansion(self, query: str) -> Dict:
        """Given a query, return expanded terms for cross-provider search."""
        analysis = self._analyze_text(query.lower())
        expansions = {}
        for concept in analysis["normalized_terms"]:
            if concept in self.concept_to_terms:
                all_terms = []
                for provider, terms in self.concept_to_terms[concept].items():
                    all_terms.extend(terms)
                expansions[concept] = list(set(all_terms))
        return expansions

    def get_inverted_index(self) -> Dict[str, List[str]]:
        """Return the inverted index of terms to chunk IDs."""
        return {k: sorted(v) for k, v in self.inverted_index.items()}

    def save_inverted_index(self, path: Path):
        """Save inverted index to disk."""
        index = self.get_inverted_index()
        Path(path).write_text(json.dumps(index, indent=2), encoding="utf-8")

    def process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Normalize all chunks and build inverted index."""
        for chunk in chunks:
            self.normalize_chunk(chunk)
        return chunks

    def process_directory(self, input_dir: Path):
        """Process all document JSON files in a directory."""
        input_dir = Path(input_dir)
        json_files = list(input_dir.rglob("*.json"))
        processed = 0
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                doc = Document(**data)
                meta = self.normalize_document(doc)
                # Attach terminology metadata to the document JSON
                data["terminology"] = {
                    "normalized_terms": meta["normalized_terms"],
                    "detected_siglas": meta["detected_siglas"],
                    "cross_cloud_equivalences": meta["cross_cloud_equivalences"],
                }
                jf.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
                processed += 1
            except Exception as e:
                logger.warning("Failed to normalize %s: %s", jf, e)

        logger.info("Normalized terminology in %d/%d documents", processed, len(json_files))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Normalize terminology in documents")
    parser.add_argument("--input", required=True, help="Input directory with processed docs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    normalizer = TerminologyNormalizer()
    normalizer.process_directory(Path(args.input))


if __name__ == "__main__":
    main()
