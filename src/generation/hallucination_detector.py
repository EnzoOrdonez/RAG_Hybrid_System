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
    # "supported", "contradicted", "unsupported", "unsupported_no_evidence",
    # "not_a_claim" (format artifact; counted but excluded from denominator, N8/H1)
    status: str
    evidence_chunk_id: Optional[str] = None
    nli_score: float = 0.0


class HallucinationReport(BaseModel):
    """Full hallucination detection report."""
    total_claims: int
    supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    not_a_claim_claims: int = 0  # format artifacts (N8/H1); excluded from denominator
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


# ============================================================
# Format-artifact detection (ledger N8 / hypothesis H1, 2026-06-30).
#
# The sentence splitter in _extract_claims emits, alongside genuine
# factual claims, non-claim fragments that the NLI verifier then scores
# with high confidence: markdown ATX headers ("### Networking."),
# table-row fragments ("AWS IAM | Feature | VPC"), spans broken across an
# unbalanced ** marker, and documentation-coverage meta-comments ("the
# context does not mention X"). These are not verifiable assertions about
# the cloud services; counting them inflates the contradicted/unsupported
# tallies — in the 50-claim audit sample 6/8 artifact-shaped claims were
# labelled "contradicted". They are tagged status="not_a_claim": kept in
# claim_details for traceability but excluded from the faithfulness
# denominator. Filtering incoherent subclaims is the documented FActScore
# behaviour (Wanner et al. 2024). Prevalence (exp12_matrix): 10.8% of all
# extracted claims, model-asymmetric (mistral 2.3% .. gemma 23.0%).
# ============================================================
_ATX_HEADER_RE = re.compile(r'^\s*#{1,6}\s')
_META_COVERAGE_RE = re.compile(
    r'(?:'
    r'(?:the )?(?:context|documentation|provided (?:context|text|documentation)|passage) '
    r'(?:does not|doesn.t|do not)'
    r'|does not (?:mention|provide|specify|describe|address|state|contain|cover)'
    r'|is not (?:mentioned|provided|specified|described|addressed|stated|covered) in'
    r'|there is no (?:information|mention)\b'
    r'|no (?:information|mention) (?:about|on|regarding|for)\b'
    r'|insufficient (?:information|context|detail)'
    r'|not enough (?:information|detail|context)'
    r')',
    re.IGNORECASE,
)


def classify_artifact(claim: str) -> Optional[str]:
    """Return the format-artifact type of a claim, or None if it is a
    genuine candidate claim. Tagged claims are scored status="not_a_claim".

    Types: "atx_header", "table_row", "unbalanced_emph", "meta_coverage".
    Module-level so the offline v3 re-score reuses the identical rule.
    """
    s = (claim or "").strip()
    if not s:
        return None
    if _ATX_HEADER_RE.match(s):
        return "atx_header"
    if s.count('|') >= 2:
        return "table_row"
    if s.count('**') % 2 == 1 or s.count('__') % 2 == 1:
        return "unbalanced_emph"
    if _META_COVERAGE_RE.search(s):
        return "meta_coverage"
    return None


def decide_nli_status(
    contr_scores: List[float],
    ent_scores: List[float],
    ent_threshold: float = 0.7,
    contr_threshold: float = 0.7,
    variant: str = "v0",
    margin: float = 0.0,
):
    """Decide a claim's NLI status from the per-chunk contradiction and
    entailment probability arrays. Shared by the runtime detector and the
    offline v3 re-score so the decision rule is byte-identical.

    Honovich 2022 TRUE base rule (the supported side already carries the
    asymmetric guard ``max_ent > max_contr``). H2 (ledger N8): the
    contradicted side has NO symmetric guard — ``max_contr`` over 5 chunks
    crossing 0.7 is enough. Variants add one:
      * ``v0``        legacy: contradicted iff max_contr > contr_threshold.
      * ``va_margin`` contradicted iff max_contr > max_ent + margin.
      * ``vb_agree``  contradicted iff >=2 chunks exceed contr_threshold.
    Returns ``(status, score, idx)`` where idx indexes the arrays.
    """
    n = len(contr_scores)
    if n == 0:
        return "unsupported", 0.0, -1
    best_ent = max(ent_scores)
    bei = ent_scores.index(best_ent)
    best_contr = max(contr_scores)
    bci = contr_scores.index(best_contr)

    # Supported (unchanged across variants).
    if best_ent > ent_threshold and best_ent > best_contr:
        return "supported", best_ent, bei

    # Contradiction gate (variant-dependent).
    if variant == "va_margin":
        contr_fires = best_contr > contr_threshold and best_contr > best_ent + margin
    elif variant == "vb_agree":
        n_over = sum(1 for c in contr_scores if c > contr_threshold)
        contr_fires = best_contr > contr_threshold and n_over >= 2
    else:  # v0 (legacy)
        contr_fires = best_contr > contr_threshold

    if contr_fires:
        return "contradicted", best_contr, bci

    # Unsupported: surface the strongest signal for reporting.
    if best_ent >= best_contr:
        return "unsupported", best_ent, bei
    return "unsupported", best_contr, bci


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
        # H2 guard variant (ledger N8). Enzo confirmed **vb_agree** at the 2b
        # sub-gate (a single chunk crossing 0.7 marked 62% of claims
        # "contradicted" vs random chunks; >=2-chunk agreement cuts it to
        # 17.5%). Symmetric with H1 (not_a_claim) already being live. Note:
        # vb_agree needs >=2 chunks to fire contradiction — with the real RAG
        # top-5 contexts this is meaningful; single-chunk edge cases can never
        # be "contradicted".
        self.nli_variant = "vb_agree"
        self.nli_margin = 0.0

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
                # Offline fallback (N8): the HF tag may not be cached, but the
                # snapshot restored under data/models/nli-deberta-v3-small/
                # (curl --ssl-no-revoke; HF client TLS-blocked on this box) does
                # load. Try it before dropping to the keyword fallback.
                from pathlib import Path
                local = (Path(__file__).resolve().parent.parent.parent
                         / "data" / "models" / "nli-deberta-v3-small")
                try:
                    from sentence_transformers import CrossEncoder
                    self._nli_model = CrossEncoder(str(local), max_length=512)
                    self._nli_available = True
                    logger.info("NLI model loaded from local snapshot: %s", local)
                except Exception as e2:
                    logger.warning(
                        "NLI model unavailable, using keyword fallback: %s / %s", e, e2)
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
            # Operational definition under the RAG paradigm: without
            # evidence, no claim can be verified. Segments the response
            # so total_claims is comparable to RAG configs and tags
            # method="no_evidence" — distinct from method="none"
            # (empty response) so aggregators can identify this path.
            return self._no_evidence_report(
                response,
                (time.perf_counter() - start) * 1000,
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

        # Step 3: Scoring. not_a_claim (format artifacts, N8/H1) stay in
        # total/claim_details for traceability but leave the denominator.
        supported = sum(1 for c in claim_details if c.status == "supported")
        contradicted = sum(1 for c in claim_details if c.status == "contradicted")
        unsupported = sum(1 for c in claim_details if c.status == "unsupported")
        not_a_claim = sum(1 for c in claim_details if c.status == "not_a_claim")
        total = len(claim_details)
        effective = total - not_a_claim

        faithfulness = supported / effective if effective > 0 else 1.0
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
            not_a_claim_claims=not_a_claim,
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

    # Bullet-line detector used by _extract_claims to parse list-format
    # responses (q196-style) that the pre-2026-05 splitter collapsed
    # into a single unsplittable block. Matches leading whitespace +
    # bullet marker (`*`, `-`, `+`, `1.`, `2)`) + space.
    _BULLET_LINE_RE = re.compile(
        r'^(\s*)(?:[\*\-\+]|\d+[\.\)])\s+(.+)$'
    )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable factual claims from the response.

        Supports both prose and bullet-list formats. For bullet lists,
        header bullets (lines ending in ``:``) are recorded as context
        and attached to their sub-bullets so the resulting claims carry
        semantic grounding instead of becoming bare labels.

        Pre-2026-05 the splitter used only ``re.split(r'(?<=[.!?])\\s+', ...)``,
        which never segmented bullet lists because bullets typically do
        not end in ``.!?``. q196 (the only ``total_claims=0`` row in
        exp9_llm_only_no_rag) is the canonical example: its substantive
        list of VPC/Subnet/Security-Group mappings produced ``claims=[]``
        and a vacuous ``faithfulness_score=1.0``.

        Fix (this method): preprocess the text by walking lines,
        recording header bullets, and emitting each leaf bullet as a
        ``"<header>: <content>."`` sentence. Empty lines reset the
        header. Prose passes through unchanged.
        """
        # Remove code blocks (not claims)
        text_clean = re.sub(r'```[\s\S]*?```', '', text)
        # Remove inline code
        text_clean = re.sub(r'`[^`]+`', '[CODE]', text_clean)
        # Remove citations
        text_clean = re.sub(r'\[Source:[^\]]*\]', '', text_clean)

        # Bullet-list normalization
        lines = text_clean.split('\n')
        result_lines: List[str] = []
        current_header: Optional[str] = None
        prev_was_bullet = False
        for line in lines:
            m = self._BULLET_LINE_RE.match(line)
            if m:
                # Transitioning from prose to a bullet block — terminate
                # the last non-empty pre-bullet line with "." if it does
                # not already end in .!?, so the splitter does not merge
                # the intro paragraph with the first bullet content
                # (canonical bug: q196's "...across AWS, Azure, and GCP:"
                # was concatenated with the first VPC bullet, swallowed
                # by SKIP_PATTERNS r"^based on", and the bullet was lost).
                if not prev_was_bullet:
                    for i in range(len(result_lines) - 1, -1, -1):
                        if result_lines[i].strip():
                            if not result_lines[i].rstrip().endswith(('.', '!', '?')):
                                result_lines[i] = (
                                    result_lines[i].rstrip() + '.'
                                )
                            break
                indent, content = m.group(1), m.group(2).strip()
                # Strip surrounding markdown bold/italic for cleaner claims
                content_clean = re.sub(r'\*\*([^\*]+)\*\*', r'\1', content)
                content_clean = re.sub(r'__([^_]+)__', r'\1', content_clean)
                if content_clean.endswith(':'):
                    # Header bullet — record as context, do not emit
                    current_header = content_clean.rstrip(':').strip()
                else:
                    # Leaf bullet — attach header context if indented
                    if current_header and len(indent) > 0:
                        result_lines.append(
                            f"{current_header}: {content_clean}."
                        )
                    else:
                        result_lines.append(f"{content_clean}.")
                prev_was_bullet = True
            else:
                stripped = line.strip()
                if not stripped:
                    # Empty line resets header (new section)
                    current_header = None
                result_lines.append(line)
                prev_was_bullet = False
        text_clean = '\n'.join(result_lines)

        # Split into sentences (bullets now act as sentence boundaries
        # because each was rewritten to terminate in ".")
        sentences = re.split(r'(?<=[.!?])\s+', text_clean)

        claims = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Strip balanced markdown emphasis so genuine claims read
            # cleanly; an UNbalanced marker that survives is a broken
            # fragment, flagged later by classify_artifact (N8/H1).
            sent = re.sub(r'\*\*([^*]+)\*\*', r'\1', sent)
            sent = re.sub(r'__([^_]+)__', r'\1', sent)
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
        """
        Match each claim against all retrieved chunks using NLI and
        aggregate per the Honovich et al. 2022 *TRUE* rule:

            max_ent   = max(entailment_prob  over all chunks)
            max_contr = max(contradiction_prob over all chunks)

            if max_ent   > ENTAILMENT_THRESHOLD and max_ent > max_contr:
                supported   (evidence chunk = argmax_ent)
            elif max_contr > CONTRADICTION_THRESHOLD:
                contradicted (evidence chunk = argmax_contr)
            else:
                unsupported

        Closes audit §19.4 Flag 138. The previous implementation
        short-circuited on the first chunk whose entailment probability
        crossed the threshold and then refused to reconsider contradictions
        from later chunks (guarded by `best_status != "supported"`). With
        five retrieved chunks and a mis-calibrated NLI (pre-Flag 135) the
        probability that at least one chunk crossed the threshold by
        noise was ~0.93, producing near-universal "supported" labels
        that overstated faithfulness.
        """
        results = []

        for claim in claims:
            # Format artifacts never reach the verifier (N8/H1): tag and skip.
            art = classify_artifact(claim)
            if art:
                results.append(ClaimDetail(
                    claim_text=claim, status="not_a_claim",
                    evidence_chunk_id=None, nli_score=0.0,
                ))
                continue
            # Build pairs for batch prediction.
            pairs = [(chunk_text, claim) for chunk_text in chunk_texts]

            if not pairs:
                results.append(ClaimDetail(
                    claim_text=claim,
                    status="unsupported",
                    nli_score=0.0,
                ))
                continue

            try:
                # [contradiction_prob, entailment_prob, neutral_prob] per pair,
                # post-softmax (see Flag 135 fix in the previous commit).
                scores = self.nli_model.predict(
                    pairs,
                    batch_size=32,
                    show_progress_bar=False,
                    apply_softmax=True,
                )
            except Exception as e:
                logger.warning(
                    "NLI prediction failed for claim, using keyword fallback: %s",
                    e,
                )
                results.append(
                    self._keyword_match_single(claim, chunk_texts, chunk_ids)
                )
                continue

            contr_scores, ent_scores = [], []
            for score_set in scores:
                # score_set: [contradiction, entailment, neutral]
                if hasattr(score_set, "__len__") and len(score_set) == 3:
                    contr_scores.append(float(score_set[0]))
                    ent_scores.append(float(score_set[1]))
                else:
                    # Single-score models (rare): similarity == entailment.
                    ent_scores.append(float(score_set))
                    contr_scores.append(0.0)

            # Honovich 2022 TRUE rule + H2 guard variant (ledger N8).
            status, nli_score, chunk_idx = decide_nli_status(
                contr_scores, ent_scores,
                ent_threshold=self.ENTAILMENT_THRESHOLD,
                contr_threshold=self.CONTRADICTION_THRESHOLD,
                variant=self.nli_variant, margin=self.nli_margin,
            )

            best_chunk_id = (
                chunk_ids[chunk_idx]
                if 0 <= chunk_idx < len(chunk_ids)
                else None
            )

            results.append(ClaimDetail(
                claim_text=claim,
                status=status,
                evidence_chunk_id=best_chunk_id,
                nli_score=round(nli_score, 4),
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
        if classify_artifact(claim):
            return ClaimDetail(
                claim_text=claim, status="not_a_claim",
                evidence_chunk_id=None, nli_score=0.0,
            )
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
        """Return an empty report for edge cases (empty response)."""
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

    def _no_evidence_report(
        self,
        response: str,
        elapsed_ms: float,
    ) -> HallucinationReport:
        """Report for the LLM-only (no retrieval) case.

        Two sub-cases, both tagged method="no_evidence" so aggregators
        can identify this path (distinct from "none" / "error"):

        1. Response makes verifiable claims → each tagged
           "unsupported_no_evidence", faithfulness=0.0,
           hallucination_rate=1.0. Operational definition under the
           RAG paradigm: without evidence, no claim is verified.

        2. Response is an honest decline (claims extraction returns []
           because skip-patterns filter out "I cannot...", "Based on...",
           etc.) → faithfulness=1.0, hallucination_rate=0.0 vacuously,
           consistent with `check()` line 136-149 for the non-empty-
           chunks / no-claims case. The honest_decline signal is
           captured separately via ResponseFormatter.
        """
        claims = self._extract_claims(response)
        if not claims:
            return HallucinationReport(
                total_claims=0,
                supported_claims=0,
                contradicted_claims=0,
                unsupported_claims=0,
                faithfulness_score=1.0,
                hallucination_rate=0.0,
                suggested_rubric=5,
                claim_details=[],
                processing_time_ms=round(elapsed_ms, 1),
                method="no_evidence",
            )
        details = []
        for c in claims:
            if classify_artifact(c):
                details.append(ClaimDetail(
                    claim_text=c, status="not_a_claim",
                    evidence_chunk_id=None, nli_score=0.0,
                ))
            else:
                details.append(ClaimDetail(
                    claim_text=c, status="unsupported_no_evidence",
                    evidence_chunk_id=None, nli_score=0.0,
                ))
        total = len(details)
        not_a_claim = sum(1 for d in details if d.status == "not_a_claim")
        effective = total - not_a_claim
        return HallucinationReport(
            total_claims=total,
            supported_claims=0,
            contradicted_claims=0,
            unsupported_claims=total - not_a_claim,
            not_a_claim_claims=not_a_claim,
            faithfulness_score=0.0 if effective > 0 else 1.0,
            hallucination_rate=1.0 if effective > 0 else 0.0,
            suggested_rubric=1 if effective > 0 else 5,
            claim_details=details,
            processing_time_ms=round(elapsed_ms, 1),
            method="no_evidence",
        )
