"""
Smoke test for the format-artifact fix in hallucination_detector.py
(ledger N8 / hypothesis H1, 2026-06-30).

Validates, WITHOUT loading the NLI model (use_nli=False -> keyword path):
  1. classify_artifact() labels the four artifact shapes and returns None
     for a genuine factual claim.
  2. check() tags artifacts status="not_a_claim", keeps them in
     total_claims for traceability, and EXCLUDES them from the
     faithfulness denominator (faithfulness = supported / effective).
  3. The q196 bullet-list extraction is NOT regressed (substantive
     bullets still become claims).

Run: python scripts/audit/smoke_extractor_artifacts.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.generation.hallucination_detector import (  # noqa: E402
    HallucinationDetector,
    classify_artifact,
)


def test_classify_artifact_unit():
    cases = {
        "### Core Functionality and Networking.": "atx_header",
        "AWS IAM | Feature/Aspect | AWS VPC (Virtual Private Cloud)": "table_row",
        "AKS is supported in Public regions and **Azure.": "unbalanced_emph",
        "The context does not mention the pricing of this service.": "meta_coverage",
        "does not specify the supported regions for this resource.": "meta_coverage",
        "there is no information about cold start latency.": "meta_coverage",
        # genuine factual claims -> None
        "AWS Lambda supports Python, Node.js and Java runtimes.": None,
        "An S3 bucket name must be globally unique across all accounts.": None,
        # a legit negative factual claim about the service (NOT doc-coverage) -> None
        "The free tier does not support cross-region replication.": None,
    }
    for claim, expected in cases.items():
        got = classify_artifact(claim)
        assert got == expected, f"classify_artifact({claim!r}) = {got!r}, expected {expected!r}"
    print(f"PASS: classify_artifact unit ({len(cases)} cases)")


def test_check_excludes_artifacts_from_denominator():
    # Response: 2 genuine claims (overlap-supported by the chunk) + 4 artifacts.
    chunk = {
        "chunk_id": "c1",
        "text": ("Amazon S3 stores objects in buckets. A bucket name must be "
                 "globally unique across all AWS accounts. S3 supports "
                 "versioning and lifecycle policies for stored objects."),
    }
    response = (
        "A bucket name must be globally unique across all AWS accounts.\n"
        "S3 supports versioning and lifecycle policies for stored objects.\n"
        "### Core Functionality and Networking.\n"
        "AWS IAM | Feature/Aspect | AWS VPC (Virtual Private Cloud) here.\n"
        "AKS is supported in Public regions and **Azure.\n"
        "Cold start latency is not specified in the provided context.\n"
    )
    det = HallucinationDetector(use_nli=False)  # keyword path, no model
    rep = det.check(response, [chunk])
    assert rep.method == "keyword_fallback", rep.method
    print(f"  total={rep.total_claims} not_a_claim={rep.not_a_claim_claims} "
          f"supported={rep.supported_claims} faith={rep.faithfulness_score}")
    # The 4 artifacts must be tagged and excluded.
    assert rep.not_a_claim_claims == 4, (
        f"expected 4 not_a_claim, got {rep.not_a_claim_claims}; "
        f"claims={[(c.claim_text[:40], c.status) for c in rep.claim_details]}"
    )
    # Traceability: artifacts still counted in total_claims.
    assert rep.total_claims >= 6, rep.total_claims
    # Denominator excludes artifacts: effective = total - not_a_claim.
    effective = rep.total_claims - rep.not_a_claim_claims
    assert effective == 2, f"effective denominator {effective} != 2"
    # The 2 genuine claims overlap the chunk -> at least one supported,
    # so faithfulness reflects the effective denominator, not the inflated one.
    assert rep.supported_claims >= 1, rep.supported_claims
    assert abs(rep.faithfulness_score - rep.supported_claims / effective) < 1e-6
    print("PASS: check() tags 4 artifacts not_a_claim and excludes them from denominator")


def test_q196_bullets_not_regressed():
    # Bullet list with a header bullet + substantive leaf bullets (q196 shape).
    response = (
        "Here are the equivalent networking concepts across providers:\n"
        "* VPC mappings:\n"
        "  * AWS uses a Virtual Private Cloud to isolate resources.\n"
        "  * Azure provides a Virtual Network for the same isolation.\n"
        "  * GCP offers a VPC Network spanning all regions.\n"
    )
    det = HallucinationDetector(use_nli=False)
    claims = det._extract_claims(response)
    real = [c for c in claims if classify_artifact(c) is None]
    assert len(real) >= 3, f"q196 bullets regressed: only {len(real)} real claims: {claims}"
    print(f"PASS: q196 bullet extraction intact ({len(real)} real claims)")


if __name__ == "__main__":
    test_classify_artifact_unit()
    test_check_excludes_artifacts_from_denominator()
    test_q196_bullets_not_regressed()
    print("\nAll smoke cases passed for the N8/H1 artifact fix.")
