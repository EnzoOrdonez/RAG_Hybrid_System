# exp19b — claim_selected vs baseline_repro (hhem)

n pareado **186** (descartadas por declinacion: 8; queries en fallback: 7)

| cantidad | valor |
|---|---|
| fidelidad baseline (pareada) | 0.4562 |
| fidelidad claim_selected (pareada) | 0.5014 |
| **diferencia pareada** | **0.0451** |
| IC95 bootstrap | 0.0076 a 0.0817 |
| d_z | 0.1727 (negligible) |
| p (crudo) | 0.01798 |
| p_BH | 0.01798 |

**Familia BH declarada:** one contrast per verifier, so BH is the identity and p_BH == p_raw; declared explicitly rather than reported as if a correction had been applied. The three verifiers are triangulation, not a family.

**TOST (banda ±0.081):** p_TOST=0.03135 · IC90 [0.0134, 0.0768] · equivalente=True

queries whose draft asserted nothing keep the baseline top-5, so they contribute an exact zero difference and shrink the observable effect; the count is reported so the reader can see the dilution

exp18's selection bound motivated this arm and is NOT a denominator: it holds the answer fixed while a real selector changes it, so any percent-of-margin figure would be fabricated
