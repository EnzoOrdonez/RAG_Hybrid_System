"""
Hallucination Detector - NLI-based claim verification.

Three-step detection:
  1. CLAIM EXTRACTION: Split response into verifiable factual claims
  2. EVIDENCE MATCHING: Compare claims against retrieved chunks via NLI
  3. SCORING: Calculate faithfulness_score, hallucination_rate, rubric 1-5

NLI Model: cross-encoder/nli-deberta-v3-small (~200MB, fast)
Fallback: keyword matching if NLI model unavailable
"""

import logging
import re
import time
from typing import List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ClaimDetail(BaseModel):
    """Detail for a single extracted claim."""
    claim_text: str
    status: str  # "supported", "contradicted", "unsupported"
    evidence_chunk_id: Optional[str] = None
    nli_score: float = 0.0


class HallucinationReport(BaseModel):
    """Full hallucination detection report."""
    total_claims: int
    supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    faithfulness_score: float  # 0.0 - 1.0
    hallucination_rate: float  # 0.0 - 1.0
    suggested_rubric: int      # 1-5
    claim_details: List[ClaimDetail]
    processing_time_ms: float
    method: str = "nli"  # "nli" or "keyword_fallback"


# ============================================================
# Sentences to skip (not factual claims)
# ============================================================
SKIP_PATTERNS = [
    r"^based on",
    r"^according to",
    r"^the.*context",
    r"^in summary",
    r"^to summarize",
    r"^overall",
    r"^note that",
    r"^please note",
    r"^for more",
    r"^see also",
    r"^however",
    r"^additionally",
    r"^furthermore",
    r"^in conclusion",
    r"^\[source",
    r"^source:",
    r"^here",
    r"^let me",
    r"^i ",
    r"^you can",
    r"^you should",
    r"^the following",
    r"^below",
    r"^above",
]


class HallucinationDetector:
    """Detects hallucinations in LLM responses using NLI."""

    NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
    ENTAILMENT_THRESHOLD = 0.7
    CONTRADICTION_THRESHOLD = 0.7
    NLI_TIMEOUT = 30  # seconds

    def __init__(self, use_nli: bool = True):
        self._nli_model = None
        self._use_nli = use_nli
        self._nli_available = None

    @property
    def nli_model(self):
        """Lazy load NLI model."""
        if self._nli_model is None and self._use_nli:
            try:
                from sentence_transformers import CrossEncoder
                self._nli_model = CrossEncoder(
                    self.NLI_MODEL,
                    max_length=512,
                )
                self._nli_available = True
                logger.info("NLI model loaded: %s", self.NLI_MODEL)
            except Exception as e:
                logger.warning("NLI model unavailable, using keyword fallback: %s", e)
                self._nli_available = False
        return self._nli_model

    def check(
        self,
        response: str,
        retrieved_chunks: list,
    ) -> HallucinationReport:
        """
        Run full hallucination detection pipeline.

        Args:
            response: LLM-generated text
            retrieved_chunks: List of chunk dicts used for context

        Returns:
            HallucinationReport with claim-level details
        """
        start = time.perf_counter()

        # Edge cases
        if not response or not response.strip():
            return self._empty_report(0.0)

        if not retrieved_chunks:
            return self._empty_report(
                (time.perf_counter() - start) * 1000,
                hallucination_rate=1.0,
            )

        # Step 1: Extract claims
        claims = self._extract_claims(response)

        if not claims:
            elapsed = (time.perf_counter() - start) * 1000
            return HallucinationReport(
                total_claims=0,
                supported_claims=0,
                contradicted_claims=0,
                unsupported_claims=0,
                faithfulness_score=1.0,  # No claims = nothing to hallucinate
                hallucination_rate=0.0,
                suggested_rubric=5,
                claim_details=[],
                processing_time_ms=elapsed,
                method="none",
            )

        # Get chunk texts
        chunk_texts = []
        chunk_ids = []
        for c in retrieved_chunks:
            text = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            cid = c.get("chunk_id", "") if isinstance(c, dict) else getattr(c, "chunk_id", "")
            if text:
                chunk_texts.append(text)
                chunk_ids.append(cid)

        # Step 2: Evidence matching
        if self._use_nli and self.nli_model is not None:
            claim_details = self._nli_matching(claims, chunk_texts, chunk_ids)
            method = "nli"
        else:
            claim_details = self._keyword_matching(claims, chunk_texts, chunk_ids)
            method = "keyword_fallback"

        # Step 3: Scoring
        supported = sum(1 for c in claim_details if c.status == "supported")
        contradicted = sum(1 for c in claim_details if c.status == "contradicted")
        unsupported = sum(1 for c in claim_details if c.status == "unsupported")
        total = len(claim_details)

        faithfulness = supported / total if total > 0 else 0.0
        hallucination_rate = 1.0 - faithfulness

        # Rubric 1-5
        if faithfulness >= 0.95:
            rubric = 5
        elif faithfulness >= 0.80:
            rubric = 4
        elif faithfulness >= 0.50:
            rubric = 3
        elif faithfulness >= 0.20:
            rubric = 2
        else:
            rubric = 1

        elapsed = (time.perf_counter() - start) * 1000

        return HallucinationReport(
            total_claims=total,
            supported_claims=supported,
            contradicted_claims=contradicted,
            unsupported_claims=unsupported,
            faithfulness_score=round(faithfulness, 4),
            hallucination_rate=round(hallucination_rate, 4),
            suggested_rubric=rubric,
            claim_details=claim_details,
            processing_time_ms=round(elapsed, 1),
            method=method,
        )

    # ============================================================
    # Step 1: Claim Extraction
    # ============================================================

    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable factual claims from the response."""
        # Remove code blocks (not claims)
        text_clean = re.sub(r'```[\s\S]*?```', '', text)
        # Remove inline code
        text_clean = re.sub(r'`[^`]+`', '[CODE]', text_clean)
        # Remove citations
        text_clean = re.sub(r'\[Source:[^\]]*\]', '', text_clean)

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text_clean)

        claims = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Skip short sentences
            if len(sent.split()) < 4:
                continue
            # Skip non-claim patterns
            sent_lower = sent.lower()
            if any(re.match(p, sent_lower) for p in SKIP_PATTERNS):
                continue
            # Skip questions
            if sent.endswith("?"):
                continue
            # Skip list headers / bullets that are just labels
            if sent.endswith(":"):
                continue

            claims.append(sent)

        return claims

    # ============================================================
    # Step 2a: NLI-based evidence matching
    # ============================================================

    def _nli_matching(
        self,
        claims: List[str],
        chunk_texts: List[str],
        chunk_ids: List[str],
    ) -> List[ClaimDetail]:
        """Match claims against evidence using NLI model."""
        results = []

        for claim in claims:
            best_status = "unsupported"
            best_score = 0.0
            best_chunk_id = None

            # Build pairs for batch prediction
            pairs = [(chunk_text, claim) for chunk_text in chunk_texts]

            if not pairs:
                results.append(ClaimDetail(
                    claim_text=claim,
                    status="unsupported",
                    nli_score=0.0,
                ))
                continue

            try:
                # NLI model returns [contradiction, entailment, neutral] scores
                scores = self.nli_model.predict(
                    pairs,
                    batch_size=32,
                    show_progress_bar=False,
                )

                for i, score_set in enumerate(scores):
                    # score_set: [contradiction, entailment, neutral]
                    if hasattr(score_set, '__len__') and len(score_set) == 3:
                        contradiction_score = float(score_set[0])
                        entailment_score = float(score_set[1])
                    else:
                        # Single score (some models return just similarity)
                        entailment_score = float(score_set)
                        contradiction_score = 0.0

                    if entailment_score > self.ENTAILMENT_THRESHOLD and entailment_score > best_score:
                        best_status = "supported"
                        best_score = entailment_score
                        best_chunk_id = chunk_ids[i] if i < len(chunk_ids) else None
                    elif contradiction_score > self.CONTRADICTION_THRESHOLD and best_status != "supported":
                        if contradiction_score > best_score:
                            best_status = "contradicted"
                            best_score = contradiction_score
                            best_chunk_id = chunk_ids[i] if i < len(chunk_ids) else None

            except Exception as e:
                logger.warning("NLI prediction failed for claim, using keyword fallback: %s", e)
                # Fallback to keyword for this claim
                detail = self._keyword_match_single(claim, chunk_texts, chunk_ids)
                results.append(detail)
                continue

            results.append(ClaimDetail(
                claim_text=claim,
                status=best_status,
                evidence_chunk_id=best_chunk_id,
                nli_score=round(best_score, 4),
            ))

        return results

    # ============================================================
    # Step 2b: Keyword-based fallback
    # ============================================================

    def _keyword_matching(
        self,
        claims: List[str],
        chunk_texts: List[str],
        chunk_ids: List[str],
    ) -> List[ClaimDetail]:
        """Fallback: match claims using keyword overlap."""
        return [
            self._keyword_match_single(claim, chunk_texts, chunk_ids)
            for claim in claims
        ]

    def _keyword_match_single(
        self,
        claim: str,
        chunk_texts: List[str],
        chunk_ids: List[str],
    ) -> ClaimDetail:
        """Match a single claim via keyword overlap."""
        claim_words = set(
            w.lower() for w in re.findall(r'\b\w+\b', claim)
            if len(w) > 2
        )

        if not claim_words:
            return ClaimDetail(
                claim_text=claim,
                status="unsupported",
                nli_score=0.0,
            )

        best_overlap = 0.0
        best_chunk_id = None

        for i, chunk_text in enumerate(chunk_texts):
            chunk_words = set(
                w.lower() for w in re.findall(r'\b\w+\b', chunk_text)
                if len(w) > 2
            )
            if not chunk_words:
                continue

            overlap = len(claim_words & chunk_words) / len(claim_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk_id = chunk_ids[i] if i < len(chunk_ids) else None

        # Thresholds for keyword matching
        if best_overlap >= 0.5:
            status = "supported"
        elif best_overlap >= 0.2:
            status = "unsupported"  # Partial match -> uncertain
        else:
            status = "unsupported"

        return ClaimDetail(
            claim_text=claim,
            status=status,
            evidence_chunk_id=best_chunk_id,
            nli_score=round(best_overlap, 4),
        )

    # ============================================================
    # Helpers
    # ============================================================

    def _empty_report(self, elapsed_ms: float, hallucination_rate: float = 0.0) -> HallucinationReport:
        """Return an empty report for edge cases."""
        return HallucinationReport(
            total_claims=0,
            supported_claims=0,
            contradicted_claims=0,
            unsupported_claims=0,
            faithfulness_score=1.0 - hallucination_rate,
            hallucination_rate=hallucination_rate,
            suggested_rubric=1 if hallucination_rate > 0.5 else 5,
            claim_details=[],
            processing_time_ms=elapsed_ms,
            method="none",
        )
