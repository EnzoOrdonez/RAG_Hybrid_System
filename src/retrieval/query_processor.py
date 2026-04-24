"""
Query Processor - Preprocessing, classification, and selective expansion.
Detects provider, expands acronyms, and classifies query type.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

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

        self.service_to_concept: Dict[str, str] = {}
        self.concept_to_terms: Dict[str, Dict[str, List[str]]] = {}
        self.term_to_providers: Dict[str, Set[str]] = {}
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
                    if not isinstance(terms, list):
                        continue
                    for term in terms:
                        lower_term = term.lower()
                        self.service_to_concept[lower_term] = concept_name
                        if provider != "generic":
                            self.term_to_providers.setdefault(lower_term, set()).add(provider)

    def process(
        self,
        query: str,
        enable_query_expansion: bool = True,
        enable_terminology_normalization: bool = True,
    ) -> ProcessedQuery:
        """Full query processing pipeline."""
        keyword_providers = self._detect_providers(query)
        services = self._detect_services(query)
        service_providers = self._detect_service_providers(services)
        providers = self._merge_providers(keyword_providers, service_providers)
        query_type = self._classify_query(query, providers)

        expanded = self._expand_terms(
            query=query,
            detected_services=services,
            query_type=query_type,
            enable_query_expansion=enable_query_expansion,
            enable_terminology_normalization=enable_terminology_normalization,
        )

        bm25_query = self._build_bm25_query(query, expanded)
        semantic_query = query
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
            for keyword in keywords:
                if keyword in q:
                    found.append(provider)
                    break
        return found

    def _detect_services(self, query: str) -> List[str]:
        q = query.lower()
        found = []
        for term in self.service_to_concept:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, q):
                found.append(term)

        for acronym in self.acronym_expansions:
            pattern = r"\b" + re.escape(acronym) + r"\b"
            if re.search(pattern, query, re.IGNORECASE):
                if acronym.lower() not in [service.lower() for service in found]:
                    found.append(acronym)
        return found

    def _detect_service_providers(self, detected_services: List[str]) -> List[str]:
        providers = []
        for service in detected_services:
            service_providers = sorted(self.term_to_providers.get(service.lower(), set()))
            if len(service_providers) == 1:
                providers.append(service_providers[0])
        return providers

    def _merge_providers(self, keyword_providers: List[str], service_providers: List[str]) -> List[str]:
        merged = []
        for provider in keyword_providers + service_providers:
            if provider not in merged:
                merged.append(provider)
        return merged

    def _classify_query(self, query: str, providers: List[str]) -> str:
        for pattern in self.CROSS_CLOUD_PATTERNS:
            if pattern.search(query):
                return QueryType.CROSS_CLOUD

        if len(providers) >= 2:
            return QueryType.CROSS_CLOUD

        for pattern in self.PROCEDURAL_PATTERNS:
            if pattern.search(query):
                return QueryType.PROCEDURAL

        if len(providers) == 1:
            return QueryType.SINGLE_PROVIDER

        return QueryType.CONCEPTUAL

    def _expand_terms(
        self,
        query: str,
        detected_services: List[str],
        query_type: str,
        enable_query_expansion: bool,
        enable_terminology_normalization: bool,
    ) -> List[str]:
        if not enable_query_expansion:
            return []

        expanded = []
        allow_cross_provider_terms = (
            enable_terminology_normalization
            and query_type in (QueryType.CROSS_CLOUD, QueryType.CONCEPTUAL)
        )

        if allow_cross_provider_terms:
            for service in detected_services:
                concept = self.service_to_concept.get(service.lower())
                if concept and concept in self.concept_to_terms:
                    for terms in self.concept_to_terms[concept].values():
                        if not isinstance(terms, list):
                            continue
                        for term in terms:
                            if term.lower() != service.lower() and term not in expanded:
                                expanded.append(term)

        for acronym, expansion in self.acronym_expansions.items():
            if re.search(r"\b" + re.escape(acronym) + r"\b", query, re.IGNORECASE):
                if expansion not in expanded:
                    expanded.append(expansion)

        return expanded

    def _build_bm25_query(self, query: str, expanded_terms: List[str]) -> str:
        parts = [query]
        if expanded_terms:
            parts.append(" ".join(expanded_terms))
        return " ".join(parts)

    def _get_provider_filter(self, query_type: str, providers: List[str]) -> Optional[List[str]]:
        if query_type == QueryType.SINGLE_PROVIDER and providers:
            return providers
        if query_type == QueryType.CROSS_CLOUD:
            return None
        if query_type == QueryType.CONCEPTUAL:
            return None
        return None
