# exp19b — claim_selected vs baseline_repro (small)

n pareado **186** (descartadas por declinacion: 8; queries en fallback: 7)

| cantidad | valor |
|---|---|
| fidelidad baseline (pareada) | 0.2423 |
| fidelidad claim_selected (pareada) | 0.2401 |
| **diferencia pareada** | **-0.0021** |
| IC95 bootstrap | -0.044 a 0.0393 |
| d_z | -0.0073 (negligible) |
| p (crudo) | 0.94206 |
| p_BH | 0.94206 |

**Familia BH declarada:** one contrast per verifier, so BH is the identity and p_BH == p_raw; declared explicitly rather than reported as if a correction had been applied. The three verifiers are triangulation, not a family.

**TOST (banda ±0.081):** p_TOST=0.00015 · IC90 [-0.0375, 0.0332] · equivalente=True

queries whose draft asserted nothing keep the baseline top-5, so they contribute an exact zero difference and shrink the observable effect; the count is reported so the reader can see the dilution

exp18's selection bound motivated this arm and is NOT a denominator: it holds the answer fixed while a real selector changes it, so any percent-of-margin figure would be fabricated
