"""
Query Processor - Preprocessing, classification, and expansion of user queries.
Detects provider, expands acronyms, and classifies query type.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class QueryType:
    SINGLE_PROVIDER = "single_provider"
    CROSS_CLOUD = "cross_cloud"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"


class ProcessedQuery:
    """Result of query processing."""

    def __init__(
        self,
        original: str,
        bm25_query: str,
        semantic_query: str,
        query_type: str,
        detected_providers: List[str],
        detected_services: List[str],
        expanded_terms: List[str],
        provider_filter: Optional[List[str]] = None,
    ):
        self.original = original
        self.bm25_query = bm25_query
        self.semantic_query = semantic_query
        self.query_type = query_type
        self.detected_providers = detected_providers
        self.detected_services = detected_services
        self.expanded_terms = expanded_terms
        self.provider_filter = provider_filter

    def __repr__(self):
        return (
            f"ProcessedQuery(type={self.query_type}, "
            f"providers={self.detected_providers}, "
            f"bm25='{self.bm25_query[:60]}...')"
        )


class QueryProcessor:
    """Preprocesses queries for the retrieval pipeline."""

    PROVIDER_KEYWORDS = {
        "aws": ["aws", "amazon", "amazon web services"],
        "azure": ["azure", "microsoft azure", "microsoft"],
        "gcp": ["gcp", "google cloud", "google cloud platform"],
        "k8s": ["kubernetes", "k8s", "kubectl"],
        "cncf": ["cncf", "cloud native"],
    }

    PROCEDURAL_PATTERNS = [
        re.compile(r"\bhow\s+to\b", re.IGNORECASE),
        re.compile(r"\bstep[s]?\s+(to|for|by)\b", re.IGNORECASE),
        re.compile(r"\bcreate\b|\bset\s*up\b|\bconfigure\b|\bdeploy\b|\binstall\b", re.IGNORECASE),
        re.compile(r"\btutorial\b|\bguide\b|\bwalkthrough\b", re.IGNORECASE),
    ]

    CROSS_CLOUD_PATTERNS = [
        re.compile(r"\bcompare\b|\bcomparison\b|\bvs\.?\b|\bversus\b", re.IGNORECASE),
        re.compile(r"\bdifference[s]?\s+(between|of)\b", re.IGNORECASE),
        re.compile(r"\balternative[s]?\b|\bequivalent\b", re.IGNORECASE),
    ]

    def __init__(self, mappings_path: str = "config/terminology_mappings.yaml"):
        project_root = Path(__file__).parent.parent.parent
        mappings_file = project_root / mappings_path
        if mappings_file.exists():
            with open(mappings_file, encoding="utf-8") as f:
                self.mappings = yaml.safe_load(f)
        else:
            self.mappings = {}

        # Build service -> concept mapping
        self.service_to_concept: Dict[str, str] = {}
        self.concept_to_terms: Dict[str, Dict[str, List[str]]] = {}
        self.acronym_expansions: Dict[str, str] = self.mappings.get("acronyms", {})
        self._build_lookups()

    def _build_lookups(self):
        for category, concepts in self.mappings.items():
            if category == "acronyms" or not isinstance(concepts, dict):
                continue
            for concept_name, providers in concepts.items():
                if not isinstance(providers, dict):
                    continue
                self.concept_to_terms[concept_name] = providers
                for provider, terms in providers.items():
                    if isinstance(terms, list):
                        for term in terms:
                            self.service_to_concept[term.lower()] = concept_name

    def process(self, query: str) -> ProcessedQuery:
        """Full query processing pipeline."""
        # 1. Detect providers
        providers = self._detect_providers(query)

        # 2. Detect services / acronyms
        services = self._detect_services(query)

        # 3. Classify query type
        query_type = self._classify_query(query, providers)

        # 4. Expand terms for BM25
        expanded = self._expand_terms(query, services)

        # 5. Build BM25 query (expanded)
        bm25_query = self._build_bm25_query(query, expanded)

        # 6. Semantic query stays as-is (embedding captures semantics)
        semantic_query = query

        # 7. Determine provider filter
        provider_filter = self._get_provider_filter(query_type, providers)

        return ProcessedQuery(
            original=query,
            bm25_query=bm25_query,
            semantic_query=semantic_query,
            query_type=query_type,
            detected_providers=providers,
            detected_services=services,
            expanded_terms=expanded,
            provider_filter=provider_filter,
        )

    def _detect_providers(self, query: str) -> List[str]:
        q = query.lower()
        found = []
        for provider, keywords in self.PROVIDER_KEYWORDS.items():
            for kw in keywords:
                if kw in q:
                    found.append(provider)
                    break
        return found

    def _detect_services(self, query: str) -> List[str]:
        q = query.lower()
        found = []
        for term in self.service_to_concept:
            # Match whole word
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, q):
                found.append(term)
        # Also check acronyms
        for acronym in self.acronym_expansions:
            pattern = r'\b' + re.escape(acronym) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                if acronym.lower() not in [f.lower() for f in found]:
                    found.append(acronym)
        return found

    def _classify_query(self, query: str, providers: List[str]) -> str:
        # Check cross-cloud patterns
        for pattern in self.CROSS_CLOUD_PATTERNS:
            if pattern.search(query):
                return QueryType.CROSS_CLOUD

        # Multiple providers mentioned
        if len(providers) >= 2:
            return QueryType.CROSS_CLOUD

        # Check procedural
        for pattern in self.PROCEDURAL_PATTERNS:
            if pattern.search(query):
                return QueryType.PROCEDURAL

        # Single provider
        if len(providers) == 1:
            return QueryType.SINGLE_PROVIDER

        return QueryType.CONCEPTUAL

    def _expand_terms(self, query: str, detected_services: List[str]) -> List[str]:
        expanded = []
        for service in detected_services:
            concept = self.service_to_concept.get(service.lower())
            if concept and concept in self.concept_to_terms:
                for provider, terms in self.concept_to_terms[concept].items():
                    for term in terms:
                        if term.lower() != service.lower() and term not in expanded:
                            expanded.append(term)

        # Expand acronyms
        for acronym, expansion in self.acronym_expansions.items():
            if re.search(r'\b' + re.escape(acronym) + r'\b', query, re.IGNORECASE):
                if expansion not in expanded:
                    expanded.append(expansion)

        return expanded

    def _build_bm25_query(self, query: str, expanded_terms: List[str]) -> str:
        parts = [query]
        if expanded_terms:
            parts.append(" ".join(expanded_terms))
        return " ".join(parts)

    def _get_provider_filter(
        self, query_type: str, providers: List[str]
    ) -> Optional[List[str]]:
        if query_type == QueryType.SINGLE_PROVIDER and providers:
            return providers
        if query_type == QueryType.CROSS_CLOUD:
            return None  # Search all
        if query_type == QueryType.CONCEPTUAL:
            return None  # Search all + CNCF
        return None  # No filter
