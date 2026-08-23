# Verificación offline — cifras de la fase de verano (Tier A / exp16 / exp17)
Fecha: 2026-08-23 · solo lectura sobre experiments/ · sin GPU ni LLM

Re-deriva cada cifra desde las probabilidades persistidas: si el pase GPU se hizo una vez y todo lo demás es re-agregación CPU (el diseño de la fase), esto debe dar exacto.

Experimentos descubiertos por forma del artefacto: exp15_ablation_tierA, exp16_anchored_decoding, exp17_crosscloud_balanced, exp18_evidence_ceiling, exp19b_anchored_selector

## exp15_ablation_tierA  (ancla: baseline_repro)
- [OK ] exp15_ablation_tierA/small: 281 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp15_ablation_tierA/small: 4 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp15_ablation_tierA/small: declared BH family == contrast count — declares 4, has 4
- [OK ] exp15_ablation_tierA/small: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 0/4 significativos (BH) · nivel ancla 0.3077
- [OK ] exp15_ablation_tierA/base: 281 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp15_ablation_tierA/base: 4 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp15_ablation_tierA/base: declared BH family == contrast count — declares 4, has 4
- [OK ] exp15_ablation_tierA/base: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 0/4 significativos (BH) · nivel ancla 0.2038
- [OK ] exp15_ablation_tierA/hhem: 281 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp15_ablation_tierA/hhem: 4 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp15_ablation_tierA/hhem: declared BH family == contrast count — declares 4, has 4
- [OK ] exp15_ablation_tierA/hhem: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 0/4 significativos (BH) · nivel ancla 0.4499
- [OK ] exp15_ablation_tierA: nivel HHEM del ancla en rango (carga verificada) — 0.4499

## exp16_anchored_decoding  (ancla: baseline_repro)
- [OK ] exp16_anchored_decoding/small: 157 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp16_anchored_decoding/small: 2 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp16_anchored_decoding/small: declared BH family == contrast count — declares 2, has 2
- [OK ] exp16_anchored_decoding/small: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 0/2 significativos (BH) · nivel ancla 0.2961
- [OK ] exp16_anchored_decoding/base: 157 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp16_anchored_decoding/base: 2 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp16_anchored_decoding/base: declared BH family == contrast count — declares 2, has 2
- [OK ] exp16_anchored_decoding/base: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 0/2 significativos (BH) · nivel ancla 0.2258
- [OK ] exp16_anchored_decoding/hhem: 157 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp16_anchored_decoding/hhem: 2 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp16_anchored_decoding/hhem: declared BH family == contrast count — declares 2, has 2
- [OK ] exp16_anchored_decoding/hhem: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 0/2 significativos (BH) · nivel ancla 0.4983
- [OK ] exp16_anchored_decoding: nivel HHEM del ancla en rango (carga verificada) — 0.4983

## exp17_crosscloud_balanced  (ancla: baseline)
- [OK ] exp17_crosscloud_balanced/small: 50 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp17_crosscloud_balanced/small: 1 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp17_crosscloud_balanced/small: declared BH family == contrast count — declares 1, has 1
- [OK ] exp17_crosscloud_balanced/small: declared anchor == actual anchor — declares baseline, is baseline
    · 0/1 significativos (BH) · nivel ancla 0.199
- [OK ] exp17_crosscloud_balanced/base: 50 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp17_crosscloud_balanced/base: 1 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp17_crosscloud_balanced/base: declared BH family == contrast count — declares 1, has 1
- [OK ] exp17_crosscloud_balanced/base: declared anchor == actual anchor — declares baseline, is baseline
    · 0/1 significativos (BH) · nivel ancla 0.1514
- [OK ] exp17_crosscloud_balanced/hhem: 50 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp17_crosscloud_balanced/hhem: 1 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp17_crosscloud_balanced/hhem: declared BH family == contrast count — declares 1, has 1
- [OK ] exp17_crosscloud_balanced/hhem: declared anchor == actual anchor — declares baseline, is baseline
    · 0/1 significativos (BH) · nivel ancla 0.4772
- [OK ] exp17_crosscloud_balanced: nivel HHEM del ancla en rango (carga verificada) — 0.4772

## exp18_evidence_ceiling  (ancla: baseline_repro)
- [OK ] exp18_evidence_ceiling/small: 625 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp18_evidence_ceiling/small: 3 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp18_evidence_ceiling/small: declared BH family == contrast count — declares 3, has 3
- [OK ] exp18_evidence_ceiling/small: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 1/3 significativos (BH) · nivel ancla 0.2415
- [OK ] exp18_evidence_ceiling/base: 625 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp18_evidence_ceiling/base: 3 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp18_evidence_ceiling/base: declared BH family == contrast count — declares 3, has 3
- [OK ] exp18_evidence_ceiling/base: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 1/3 significativos (BH) · nivel ancla 0.1418
- [OK ] exp18_evidence_ceiling/hhem: 625 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp18_evidence_ceiling/hhem: 3 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp18_evidence_ceiling/hhem: declared BH family == contrast count — declares 3, has 3
- [OK ] exp18_evidence_ceiling/hhem: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 1/3 significativos (BH) · nivel ancla 0.4638
- [OK ] exp18_evidence_ceiling: nivel HHEM del ancla en rango (carga verificada) — 0.4638

## exp19b_anchored_selector  (ancla: baseline_repro)
- [OK ] exp19b_anchored_selector/small: 369 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp19b_anchored_selector/small: 1 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp19b_anchored_selector/small: declared BH family == contrast count — declares 1, has 1
- [OK ] exp19b_anchored_selector/small: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 0/1 significativos (BH) · nivel ancla 0.2398
- [OK ] exp19b_anchored_selector/base: 369 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp19b_anchored_selector/base: 1 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp19b_anchored_selector/base: declared BH family == contrast count — declares 1, has 1
- [OK ] exp19b_anchored_selector/base: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 0/1 significativos (BH) · nivel ancla 0.1421
- [OK ] exp19b_anchored_selector/hhem: 369 faithfulness cells re-aggregated from raw probs — exact
- [OK ] exp19b_anchored_selector/hhem: 1 paired contrasts recomputed (Wilcoxon/d_z/BH) — exact
- [OK ] exp19b_anchored_selector/hhem: declared BH family == contrast count — declares 1, has 1
- [OK ] exp19b_anchored_selector/hhem: declared anchor == actual anchor — declares baseline_repro, is baseline_repro
    · 1/1 significativos (BH) · nivel ancla 0.4572
- [OK ] exp19b_anchored_selector: nivel HHEM del ancla en rango (carga verificada) — 0.4572

## Resultado
**Todas las verificaciones pasaron.** Las cifras titulares de la fase de verano se reproducen desde los artefactos committeados, sin GPU.