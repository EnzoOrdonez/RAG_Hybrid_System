# exp19b — claim_selected vs baseline_repro (base)

n pareado **186** (descartadas por declinacion: 8; queries en fallback: 7)

| cantidad | valor |
|---|---|
| fidelidad baseline (pareada) | 0.1452 |
| fidelidad claim_selected (pareada) | 0.171 |
| **diferencia pareada** | **0.0258** |
| IC95 bootstrap | -0.0035 a 0.0562 |
| d_z | 0.1232 (negligible) |
| p (crudo) | 0.09958 |
| p_BH | 0.09958 |

**Familia BH declarada:** one contrast per verifier, so BH is the identity and p_BH == p_raw; declared explicitly rather than reported as if a correction had been applied. The three verifiers are triangulation, not a family.

**TOST (banda ±0.081):** p_TOST=0.00021 · IC90 [0.0004, 0.0513] · equivalente=True

queries whose draft asserted nothing keep the baseline top-5, so they contribute an exact zero difference and shrink the observable effect; the count is reported so the reader can see the dilution

exp18's selection bound motivated this arm and is NOT a denominator: it holds the answer fixed while a real selector changes it, so any percent-of-margin figure would be fabricated
