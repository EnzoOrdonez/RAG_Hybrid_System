# RESULTADOS_RESUMEN — Nota 3 (2026-06-11)

**Estado:** retrieval (exp11) COMPLETO; matriz de fidelidad (exp12) **COMPLETA**
(4 modelos × 4 escenarios × 194 q, temp=0); **exp13 (expansión cross-cloud) COMPLETO**.
Insumo para A.3/A.1.

---

## 1. Retrieval (exp11_retrieval194_fullrerank) — FINAL

NDCG@5 graded por oráculo (mitigación de circularidad: el reranker del pipeline =
ms-marco; oráculo independiente = bge-reranker-large):

| Sistema | ms-marco (circular) | bge (independiente) |
|---|---|---|
| Léxico (BM25) | 0,552 | 0,442 |
| Denso (BGE) | 0,649 | 0,624 |
| Híbrido pre-rerank (RRF) | 0,668 | 0,603 |
| Híbrido post-rerank | **0,995** | **0,740** |

- **Híbrido(post) > Denso: real y significativo** con oráculo independiente
  (d_z=+0,45, p_BH<0,001), pero la magnitud estaba **inflada por circularidad**
  (d_z=+1,38 con ms-marco; NDCG 0,995 ≈ techo por construcción).
- **Híbrido(pre, solo fusión RRF) ≈ Denso** (n.s. en ambos oráculos). **La ventaja del
  híbrido es la etapa de reranking, no la fusión RRF.**
- Orden Híbrido > Denso > Léxico **estable** entre oráculos.

**Delta del fix del reranker (D12, exp11 vs exp10):** corregir la truncación a 200 chars
(reranker ahora ve texto completo, igual que el oráculo) llevó al híbrido post-rerank a
NDCG@5≈0,995 **bajo el oráculo compartido** — es decir, el fix hace que reranker y oráculo
sean la misma función: la "subida" es **tautológica**, no una mejora de calidad. El número
honesto es 0,740 (oráculo independiente). Esto era invisible en exp10 (oráculo único = el
propio reranker, sobre texto truncado).

## 2. Veredicto exp7 / "+16,8 %" — RETIRADO

Probado por código: los dos brazos de exp7 corrieron RRF con la query **cruda** en ambas
piernas (`hybrid_retriever.py:52`); la expansión nunca llegó a BM25 y `terminology_normalization`
no se lee en retrieval. **Ambos brazos recuperaron lo mismo** → el +16,75 % (0,2916 vs 0,2498)
es ruido de generación + NLI descalibrado sobre n=29, **no** una ganancia de
expansión/normalización. Refuerza Flag 142 (candidato a remover del abstract).

**exp13 (expansión real, fix D11, 25 q cross-cloud, granite determinista) — CONFIRMA que la
expansión no aporta:**
- Con D11 la expansión **sí** llega a BM25: cambió el retrieval en **7/25** consultas
  (exp7 cambiaba 0 — era un no-op).
- Bajo oráculo independiente (bge), ON es **direccionalmente PEOR**: NDCG@5 0,852→0,820,
  recall@5 0,863→0,789, avg_score 0,287→0,269 — **no significativo tras BH** (d_z≈−0,4,
  p_BH=0,13). La expansión añade términos a BM25 que perturban sin mejorar.
- Fidelidad: OFF 0,175 ≈ ON 0,174 (d_z≈0, n.s.).
- **Veredicto doble y definitivo:** el "+16,8 %" no solo era un no-op (exp7); con la
  implementación correcta, la expansión cross-cloud **no mejora el retrieval (tiende a
  perjudicarlo levemente) ni la fidelidad**. El claim queda **retirado**.

## 3. Fidelidad (exp12, COMPLETO — 4 modelos) — **métrica v2 (N5, 2026-06-11)**

**Corrección de medición (N5):** la métrica publicada hasta 06-11 promediaba TODA respuesta
con claims, mezclando declinaciones honestas (cuyos textos explicativos también generan
claims) con respuestas reales, con sesgo asimétrico por modelo (hundía a granite, inflaba a
qwen). Además el flag `is_honest_decline` mislabelea en ambas direcciones (mistral ~18 % de
rechazos no marcados entre "contestadas"; ~60 % de los "declines" de granite/qwen son
respuestas parciales largas). v2 reclasifica a nivel análisis: `pure_decline` (rechazo en
los primeros 300c) / `hedged_partial` (hedge tardío) / `answered`.

**Tabla 6 v2 — `faithfulness_answered` PRIMARIA** (excluye solo pure_decline; n entre paréntesis):

| Escenario | Granite 4.1 | Gemma 4 E4B | Mistral 7B | Qwen 3.5 9B |
|---|---|---|---|---|
| Sin RAG | 0,000* (189) | 0,000* (191) | 0,000* (193) | 0,000* (17†) |
| RAG léxico | 0,234 (75) | 0,381 (60) | 0,245 (114) | 0,246 (42) |
| RAG denso | 0,255 (85) | 0,356 (65) | 0,275 (132) | 0,246 (51) |
| RAG híbrido | 0,293 (87) | 0,312 (60) | 0,282 (122) | 0,293 (47) |

`*` Sin RAG = 0 por construcción (N3). `†` qwen sin_rag: 177/194 respuestas VACÍAS (N6).
Sensibilidad bajo 4 denominadores: `output/tables/nota3/tabla6_sensibilidad_denominador`.
La métrica v1 se conserva solo como sensibilidad etiquetada (sens_c).

**Tabla 6 v3 — `faithfulness_answered` corregida (small, N8: artefactos de formato excluidos del
denominador + guarda de contradicción vb_agree):**

| Escenario | Granite 4.1 | Gemma 4 E4B | Mistral 7B | Qwen 3.5 9B |
|---|---|---|---|---|
| RAG léxico | 0,235 (75) | **0,443** (60) | 0,246 (114) | **0,361** (42) |
| RAG denso | 0,247 (85) | **0,425** (65) | 0,281 (132) | **0,351** (51) |
| RAG híbrido | 0,299 (87) | **0,363** (60) | 0,288 (122) | **0,354** (47) |

v2 (arriba) queda como *superseded documentado* (postura N8: ambas). Delta = exclusión de artefactos
(H1): granite/mistral ~igual (±0,01); **gemma/qwen suben +0,05..0,11** (tenían 20-23 % de artefactos
de formato que inflaban su denominador con contradicciones-basura). Fuente:
`faithfulness_metrics_v3_small.json`; tabla `output/tables/nota3/tabla6_fidelidad_v3`.

**Tabla 6 v4 — CITABLE (small, N9): v3 + exclusión de respuestas 100%-artefactos** (`genuine==0`,
59/1798, que en v3 entraban como faithfulness=1,0 vacuo; mismo trato que method='none', Flag 137):

| Escenario | Granite 4.1 | Gemma 4 E4B | Mistral 7B | Qwen 3.5 9B |
|---|---|---|---|---|
| RAG léxico | 0,235 (75) | 0,413 (57) | 0,246 (114) | 0,255 (36) |
| RAG denso | 0,247 (85) | 0,377 (60) | 0,281 (132) | 0,265 (45) |
| RAG híbrido | 0,299 (87) | 0,317 (56) | 0,288 (122) | 0,294 (43) |

Granite/mistral idénticos a v3 (sus vacuas eran declinaciones ya excluidas); gemma baja −0,03..−0,05
y qwen −0,06..−0,11 respecto de v3 — parte del alza v2→v3 de gemma/qwen venía de los flips 0→1,0
vacuos, no solo de limpiar el denominador. Fuente: `faithfulness_metrics_v4_small.json`; tabla
`output/tables/nota3/tabla6_fidelidad_v4`. Detalle y decisión: ledger **N9**.

Re-stats v2 (pareado por INTERSECCIÓN de no-excluidas en ambos brazos; n por par; familias BH):
- **"El método de retrieval NO mueve la fidelidad" SE SOSTIENE bajo la métrica corregida:**
  ningún par RAG-vs-RAG significativo tras BH en ningún modelo. PERO ahora con matiz honesto:
  granite es monótono léxico→denso→híbrido (0,234→0,255→0,293; mejor par híbrido-vs-léxico
  d_z=+0,29, p_BH=0,11, n=53) y mistral también (0,245→0,275→0,282, n.s.); gemma tiende al
  REVÉS (0,381→0,312). Potencia limitada: n pareado 20–63 en granite/gemma/qwen ([n<60] en
  el JSON). La excepción v1 (denso>léxico en qwen) NO sobrevive a la métrica corregida.
- **RAG ≫ Sin RAG se sostiene** para granite/gemma/mistral (d_z 0,86–1,09, p_BH<0,001).
  Para **qwen es NO testeable** bajo v2 (n pareado 5–7 por las vacías de sin_rag, N6).
- **Entre modelos:** bajo v2 (small, artefactos incluidos) todo n.s. — pero eso era **artefacto del
  contaminante de formato**. Corrigiendo el instrumento (v3, artefactos excluidos + vb_agree), bajo el
  **mismo verificador small hay 2/18 pares significativos** (denso granite-vs-mistral d_z=+0,42
  p_BH=0,014 n=75; léxico gemma-vs-mistral d_z=−0,45 p_BH=0,044 n=49); bajo base, 6/18. Las diferencias
  entre modelos estaban **enmascaradas** por el sesgo asimétrico de artefactos (gemma 23 % / qwen 20 %
  / mistral 2 %). **Esto corrige el "todo n.s." de N5.** Lo que además es altamente significativo es el
  **comportamiento de declinación** (McNemar v2: mistral declina mucho menos que el resto, p≈0,000). El
  trade-off declinación ↔ fidelidad sigue siendo un hallazgo. **[Corregido en N9 — framing B
  (decisión de Enzo, 02/07):** las filas vacuas sostenían parte de esos pares. Excluyéndolas (v4)
  y aplicando el MISMO estándar de doble verificador con el que se defiende el hallazgo central,
  **0/18 pares entre-modelos significativos bajo los dos verificadores**: cada verificador marca
  1/18 par —distinto— (small: denso granite-vs-mistral d_z=+0,42 p_BH=0,014 n=75; base: léxico
  gemma-vs-granite d_z=−0,53 p_BH=0,038 n=42) que NO se sostiene al cruzar verificadores —
  resultado frágil atribuible al instrumento (cf. TRUE, arXiv:2204.04991; Verifying the
  Verifiers, arXiv:2506.13342), no hallazgo. El 0/12 RAG-vs-RAG se mantiene bajo v3 y v4;
  réplicas H5 en curso solo como nota de refuerzo.**]**

**Desglose de claims (instrumento, `tabla_claims_desglose`):** % contradicted casi insensible
al escenario y al modelo (27–39 %), y donde varía, sube con MEJOR contexto (gemma léxico→híbrido
26,8→34,5 %) — evidencia de contradicciones espurias del verificador (ver Instrumento, abajo).
**N8:** con la guarda `vb_agree` (≥2 chunks deben cruzar el umbral) la banda se desploma —
base-rate sintético de falso-contradicted 62 %→17,5 % vs chunks aleatorios; q085 26→5 contradicted —
confirmando que gran parte de esa banda era ruido de `max` sobre 5 chunks sin guarda simétrica.

**Auditoría del instrumento (N5):**
- q085 (granite-híbrido): 28/28 claims "contradicted" a prob ~0,99 en una respuesta procedural
  citada y fundamentada ("Open your web browser and go to the Azure portal" = contradicted
  0,986). Reproducido bit-exacto. El lado contradicted usa max sobre 5 chunks con umbral 0,7
  SIN guarda simétrica (`hallucination_detector.py:413-424`).
- Muestra de 50 claims para juicio humano: `output/audit/claim_audit_sample.{csv,md}`
  (20 contradicted incl. q085, 20 unsupported, 10 supported; pendiente de revisión manual).
- **Segundo verificador (nli-deberta-v3-base, fp16) — COMPLETADO** (re-score íntegro de los 12
  configs RAG, 0 mismatches de claims; `faithfulness_rescore__nli-base.json`):
  - Ablación de formato (small sin prefijo "Header:"): **negativa limpia** — kappa 0,87,
    Spearman 0,944, mismos órdenes por modelo, q085 idéntico. El formato sintetizado de claims
    NO es el artefacto.
  - Verificador base vs small: kappa claim-level **0,411** (las contradicciones de small migran
    a unsupported bajo base); Spearman de medias por config 0,825 (publicada) pero **0,559
    (n.s.) en la primaria**; el orden léxico/denso/híbrido **cambia en 3 de 4 modelos**.
    Niveles absolutos más bajos bajo base (granite primaria 0,29→0,21).
  - **q085 sigue 28/28 contradicted bajo base** → el artefacto vive en el PROCEDIMIENTO
    (max-contradicción sobre 5 chunks largos, umbral 0,7, sin guarda simétrica) + desajuste de
    dominio, no en la capacidad del modelo ni el formato.
  - **Espejo de la familia B bajo verificador base (primaria, pareado, BH): TODO n.s.**
    (mejor |d_z|=0,38). El veredicto "método de retrieval n.s." queda apoyado bajo
    2 verificadores × 4 denominadores.
  - Implicación para el paper: la fidelidad NLI se reporta como **instrumento-relativa**
    (contrastes, no niveles); detalle en `output/audit/rescore_v2_summary.md`.

## 4. Sorpresas

1. **La superioridad del híbrido NO propaga a la fidelidad** (métodos de RAG n.s.). Hallazgo
   honesto y central: el aporte del híbrido es de ranking, no de fidelidad de respuesta.
   **Re-confirmado bajo la métrica corregida v2 (N5) y bajo v3 (N8): retrieval n.s. en fidelidad
   con LOS DOS verificadores (base 0/12, small 0/12)** — con el matiz de monotonía no
   significativa de granite/mistral hacia el híbrido (§3).
2. **Circularidad del oráculo infla masivamente** el retrieval del híbrido (0,995 → 0,740).
3. **Determinismo desigual a temp=0:** granite y qwen3.5 deterministas; **gemma y mistral NO**
   (no-determinismo de kernel Ollama). Headline = granite (determinista + RAG-tuned).
4. **Sesgo de extracción de claims entre modelos — cuantificado (N8):** además del volumen (gemma
   produce ~la mitad de claims extraíbles, n_eff≈100, que granite ≈189), **10,8 % de los claims son
   artefactos de formato** (headers `###` / filas de tabla / negrita rota / meta-comentario), y el
   sesgo es asimétrico: gemma 23 % · qwen 20 % · granite 8 % · mistral 2 %. No solo confunde niveles:
   **enmascaró diferencias reales entre modelos** (v3 las revela, §3). Corregido excluyéndolos del
   denominador (`not_a_claim`).
5. **granite declina 62–66 %** vs mistral 20–25 %: el comportamiento de declinación domina la
   varianza de fidelidad.

## 5. Latencias (p50, gen) — preliminar
gemma ~38 s (26 tok/s) · mistral ~34–45 s (10–14 tok/s) · granite ~65–89 s (verboso, NLI
1,2–1,9 s por muchos claims) · qwen3.5 ~239 s (cuello). Tabla completa:
`output/tables/nota3/latency__exp12_matrix.md`.

## 6. Puntos que un jurado atacaría (y mitigación)

| Ataque | Mitigación en esta ronda |
|---|---|
| Circularidad del oráculo (reranker = juez) | Oráculo independiente (bge) + ranking pre-rerank; orden estable; número honesto 0,740 reportado |
| Sesgo de pooling (relevancia solo sobre lo recuperado) | Documentado; NDCG graded sin umbral + sensibilidad de percentil |
| "Sin RAG" fidelidad = 0 por construcción | Anotado (N3); se reporta answered/decline, no como fidelidad comparable |
| Sesgo de extracción de claims entre modelos | Reportado n_eff + decline por modelo; comparación entre modelos con caveat |
| 2/4 modelos no deterministas a temp=0 | Documentado por modelo; headline = modelo determinista (granite) |
| Sin qrels humanos ni respuestas gold | Limitación declarada; oráculos múltiples como proxy; ventana de anotación pendiente |
| Fidelidad ≠ corrección | NLI mide groundedness en el contexto, no veracidad factual; explícito |

## 7. Pendiente
- Reescritura A.3/A.1 a partir de este resumen (figuras f1-f4 ya generadas:
  `output/figures/nota3/`; Tabla 4 en `output/tables/nota3/`).

## 8. Cierre operativo (2026-06-11, N7)
- exp13 re-medido bajo métrica v2: **"expansión OFF≈ON" SE SOSTIENE** (0,285 vs 0,324,
  n=10, n.s.) — veredicto N4 intacto.
- Repo publicado: rama + tag **`nota3-evidencia-2026-06-11`** en origin; suite pytest verde;
  D11 y set 194 + bitácora versionados; README sin números huérfanos; checkpoints crudos
  versionados; demo con caché segregado. Detalle completo: ledger **N7**.
- Fuera del repo (humano): 50 claims, A.3 v7/A.1, figuras draw.io, SUS/B.4, video, actas.

Artefactos: `output/tables/nota3/` (oracle_stability, tabla6_fidelidad_v2/v3/v4 + sensibilidad +
clasificación v2 + claims_desglose, tabla5 v2, latency); `experiments/results/exp{11,12,13}_*`
(+ `faithfulness_metrics_v2.json`, `_v3*.json`, `_v4*.json`, `faithfulness_rescore*`); `output/audit/`
(muestra de 50 claims); `paper/audit_findings_cc_addenda.md` (N1–N9).
