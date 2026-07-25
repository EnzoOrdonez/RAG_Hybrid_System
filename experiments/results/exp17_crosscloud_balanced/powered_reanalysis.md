# exp17 — higher-power reanalysis (same 25 queries, claim resolution)

GLMM = supported ~ arm + (1|query) (binomial VB, keeps pairing). Bootstrap = query-level cluster resample of paired micro-faithfulness diff (balanced - baseline), seed 42.

| Verifier | micro base | micro bal | diff | boot95 | boot p(1-sided) | GLMM OR | GLMM p(1-sided) |
|---|---|---|---|---|---|---|---|
| small | 0.2125 | 0.2473 | 0.0348 | [-0.0954, 0.1647] | 0.302 | 1.1512 | 0.1517 |
| base | 0.1672 | 0.2097 | 0.0424 | [-0.0389, 0.1195] | 0.1428 | 1.2385 | 0.05591 |
| hhem | 0.5714 | 0.6048 | 0.0334 | [-0.0609, 0.1326] | 0.2329 | 1.2519 | 0.02075 |

One-sided tests (H1: balanced > baseline). Claim-level conditional on a genuine claim; read with the per-query decline-aware arm_stats (declines drop here).
balanced also has MORE genuine claims than baseline (less decline) — a separate gain the conditional analysis does not capture.