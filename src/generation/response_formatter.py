"""
Response Formatter - Formats LLM output with citations, confidence, and sources.

1. Parse citations [Source: provider/service/section] from generated text
2. Link each citation to original document URL
3. Generate "Sources" section at the end with links
4. Calculate confidence level:
   - HIGH: 3+ distinct citations support the response
   - MEDIUM: 1-2 citations
   - LOW: 0 citations (possible hallucination)
   - HONEST_DECLINE: LLM said "I cannot find sufficient information"
5. Detect if response includes code/configurations
6. Clean formatting: normalize whitespace, markdown
"""

import logging
import re
from typing import List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FormattedResponse(BaseModel):
    """Formatted and annotated LLM response."""
    text: str
    sources: List[dict]
    confidence: str  # HIGH, MEDIUM, LOW, HONEST_DECLINE, ERROR
    citation_count: int
    has_code: bool
    has_steps: bool
    original_length: int
    was_truncated: bool


# Patterns for honest declines
DECLINE_PATTERNS = [
    r"cannot find sufficient information",
    r"I cannot find enough information",
    r"not enough information in the.*context",
    r"the.*context does not.*contain",
    r"no relevant.*information.*found",
    r"I don't have.*information",
    r"based on the available documentation.*cannot",
    r"the provided context does not",
]

# Citation pattern: [Source: provider/service/section]
CITATION_PATTERN = re.compile(
    r'\[Source:\s*([^\]]+)\]',
    re.IGNORECASE,
)

# Code block detection
CODE_PATTERN = re.compile(r'```[\s\S]*?```|`[^`]+`')

# Step detection
STEP_PATTERN = re.compile(
    r'(?:step\s+\d|^\d+\.\s|\d+\)\s)',
    re.IGNORECASE | re.MULTILINE,
)


class ResponseFormatter:
    """Formats and annotates LLM responses."""

    MAX_OUTPUT_TOKENS = 2000

    def format(
        self,
        llm_text: str,
        retrieved_chunks: list,
        hallucination_report=None,
    ) -> FormattedResponse:
        """Format the LLM response with citations and confidence."""

        # Handle empty response
        if not llm_text or not llm_text.strip():
            return FormattedResponse(
                text="[Error: LLM returned an empty response]",
                sources=[],
                confidence="ERROR",
                citation_count=0,
                has_code=False,
                has_steps=False,
                original_length=0,
                was_truncated=False,
            )

        original_length = len(llm_text)

        # Check for honest decline
        is_decline = self._is_honest_decline(llm_text)

        # Parse citations
        citations = self._parse_citations(llm_text)

        # Build source list with context
        sources = self._build_sources(citations, retrieved_chunks)

        # Detect code and steps
        has_code = bool(CODE_PATTERN.search(llm_text))
        has_steps = bool(STEP_PATTERN.search(llm_text))

        # Truncate if too long
        was_truncated = False
        text = llm_text.strip()
        # Rough token estimate: ~4 chars per token
        if len(text) > self.MAX_OUTPUT_TOKENS * 4:
            text = text[: self.MAX_OUTPUT_TOKENS * 4]
            text += "\n\n[Response truncated due to length]"
            was_truncated = True

        # Clean up formatting
        text = self._clean_text(text)

        # Calculate confidence
        if is_decline:
            confidence = "HONEST_DECLINE"
        elif len(citations) >= 3:
            confidence = "HIGH"
        elif len(citations) >= 1:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return FormattedResponse(
            text=text,
            sources=sources,
            confidence=confidence,
            citation_count=len(citations),
            has_code=has_code,
            has_steps=has_steps,
            original_length=original_length,
            was_truncated=was_truncated,
        )

    def _is_honest_decline(self, text: str) -> bool:
        """Check if the LLM honestly declined to answer."""
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in DECLINE_PATTERNS)

    def _parse_citations(self, text: str) -> List[dict]:
        """Extract citation references from the text."""
        citations = []
        seen = set()
        for match in CITATION_PATTERN.finditer(text):
            raw = match.group(1).strip()
            if raw in seen:
                continue
            seen.add(raw)

            parts = raw.split("/")
            citation = {
                "raw": raw,
                "provider": parts[0].strip() if len(parts) > 0 else "",
                "service": parts[1].strip() if len(parts) > 1 else "",
                "section": "/".join(parts[2:]).strip() if len(parts) > 2 else "",
            }
            citations.append(citation)
        return citations

    def _build_sources(self, citations: list, chunks: list) -> List[dict]:
        """Build source list matching citations to retrieved chunks."""
        sources = []
        for cite in citations:
            source = {
                "provider": cite["provider"],
                "service": cite["service"],
                "section": cite["section"],
                "matched_chunk_id": None,
            }

            # Try to match to a retrieved chunk
            for chunk in chunks:
                chunk_prov = chunk.get("cloud_provider", "")
                chunk_svc = chunk.get("service_name", "")
                if (
                    cite["provider"].lower() == chunk_prov.lower()
                    and cite["service"].lower() == chunk_svc.lower()
                ):
                    source["matched_chunk_id"] = chunk.get("chunk_id")
                    break

            sources.append(source)
        return sources

    def _clean_text(self, text: str) -> str:
        """Clean and normalize the response text."""
        # Normalize multiple newlines
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        # Remove trailing whitespace per line
        lines = [line.rstrip() for line in text.split('\n')]
        return '\n'.join(lines)
