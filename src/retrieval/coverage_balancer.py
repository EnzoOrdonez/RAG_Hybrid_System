"""Provider-balanced selection of the final top-k (exp17).

Diagnosis that motivated this (exp17, 2026-07-24): of the 25 comparative cross-cloud
queries, only 7/25 retrieved ALL the requested providers inside the top-5 despite an
NDCG@5 of ~0.85. 18/25 lost a whole provider, which makes the comparison impossible to
ground -- the model either declines or answers about one side only. That is a
CONTENT-SELECTION failure, not a ranking-quality one, which is why better retrieval
never moved faithfulness (Tier A: rerank/top-k/order/lost-in-the-middle all null in
three verifiers) and why exp13's lexical expansion never touched it.

Re-selecting the same reranked pool with a per-provider quota lifted coverage 7/25 ->
25/25 and faithfulness in all three verifiers (HHEM +0.081, NLI base +0.045, small
+0.037; n=25, not significant, direction-consistent). The anti-gaming guards showed the
gain is genuine and not the exp16 pattern: declination fell 56%->32%, words rose
349->427, claims rose 11.5->14.9, verbatim overlap stayed flat (0.109->0.129).

This module is the single source of truth for that selection rule:
`scripts/build_balanced_retrieval_exp17.py` imports it (so exp17 stays reproducible
byte-for-byte) and `RAGPipeline` uses it for the deployable survey configuration.

Scope: the rule only makes sense when the question names several providers, so callers
must gate it on `query_type == "cross_cloud"`. It is deliberately NOT a general
diversity/MMR selector -- generalising coverage beyond cross-cloud is exp20, an open
question, not a shipped default.
"""

import math
import re
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

# Provider surface forms the runtime detector misses. QueryProcessor.PROVIDER_KEYWORDS
# has gcp = {'gcp', 'google cloud', 'google cloud platform'} with no bare 'google', so
# "Google Artifact Registry" and "Google Eventarc" resolve to no provider at all.
# Kept HERE rather than patched into QueryProcessor on purpose: that class also decides
# `query_type`, which run_generation_matrix.py feeds into the prompt router, so changing
# it would alter any re-run of the signed exp11/exp12 evidence.
_PROVIDER_ALIASES = {
    "aws": ("aws", "amazon"),
    "azure": ("azure", "microsoft"),
    "gcp": ("gcp", "google"),
}


def build_service_index(chunk_map: Mapping[str, dict]) -> Dict[str, str]:
    """{service_name_lower: provider} straight from the corpus that is actually indexed.

    Derived rather than hardcoded so it cannot drift from the built index: the K8s/CNCF
    corpus was dropped in the rebuild, yet QueryProcessor still detects 'k8s'/'cncf' as
    providers -- q189 ("EKS vs AKS vs GKE") resolves to ONLY k8s, a provider with zero
    chunks. Deriving the map from chunk_map makes that class of staleness impossible.
    """
    idx = {}
    for ch in chunk_map.values():
        svc, prov = ch.get("service_name"), ch.get("cloud_provider")
        if svc and prov:
            idx.setdefault(svc.lower(), prov)
    return idx


def corpus_providers(chunk_map: Mapping[str, dict]) -> Set[str]:
    """Providers with at least one chunk in the built index."""
    return {ch.get("cloud_provider") for ch in chunk_map.values() if ch.get("cloud_provider")}


def resolve_wanted_providers(question: str,
                             detected: Iterable[str],
                             service_index: Mapping[str, str],
                             present: Iterable[str]) -> List[str]:
    """Providers the question is actually comparing, restricted to what the corpus has.

    Three sources, unioned: the runtime detector, the alias table above, and any indexed
    SERVICE name occurring as a whole word in the question (EKS -> aws, GKE -> gcp).
    Providers with no chunks are dropped, so a stale keyword cannot claim a quota that
    no evidence can fill.

    Verified to reproduce the hand-labelled `cloud_providers` of all 25 exp17 queries,
    where the raw detector matched only 20/25 -- without which the deployed selection
    would differ from the one the pilot measured on 5 of its own queries.
    """
    present = set(present)
    q = question.lower()
    found = {p for p in detected if p in present}
    for prov, aliases in _PROVIDER_ALIASES.items():
        if prov in present and any(re.search(rf"\b{re.escape(a)}\b", q) for a in aliases):
            found.add(prov)
    for svc, prov in service_index.items():
        if prov in present and re.search(rf"\b{re.escape(svc)}\b", q):
            found.add(prov)
    # deterministic order (quota assignment must not depend on set iteration order)
    return sorted(found)


def balance(reranked_ids: Sequence[str],
            provider_of: Callable[[str], Optional[str]],
            wanted: Iterable[str],
            k: int = 5) -> List[str]:
    """Pick k ids covering each wanted provider, taken in reranked order.

    quota = ceil(k/|wanted|) per provider; any shortfall is filled from the best
    remaining ids of any provider. Relevance order is preserved within and across
    picks, so the only thing that changes versus `reranked_ids[:k]` is WHICH evidence
    is admitted -- never how it is arranged (Tier A showed arrangement is inert).

    A provider absent from the pool simply cannot be covered; the caller is expected to
    measure coverage against the providers actually present in the pool rather than
    against the requested set (see build_balanced_retrieval_exp17.py).

    Returns at most k ids; fewer only when the pool itself is smaller than k.
    """
    wanted = list(dict.fromkeys(wanted))  # unique, order-preserving
    if not wanted or k <= 0:
        return list(reranked_ids[:max(k, 0)])

    quota = math.ceil(k / len(wanted))
    per = {p: 0 for p in wanted}
    picked, pickset = [], set()
    for cid in reranked_ids:
        p = provider_of(cid)
        if p in per and per[p] < quota and len(picked) < k:
            picked.append(cid)
            pickset.add(cid)
            per[p] += 1
    if len(picked) < k:  # fill remainder from the best remaining, any provider
        for cid in reranked_ids:
            if cid not in pickset:
                picked.append(cid)
                pickset.add(cid)
                if len(picked) == k:
                    break
    return picked
