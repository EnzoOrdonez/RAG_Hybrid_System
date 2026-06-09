# RESULTADOS_RESUMEN — Nota 3 (PRELIMINAR, 2026-06-09)

**Estado:** retrieval (exp11) COMPLETO; matriz de fidelidad (exp12) con 3 de 4 modelos
completos (granite4.1, gemma4:e4b, mistral:7b); **qwen3.5:9b en curso** (~25-40 h restantes);
**exp13 (expansión) pendiente** (requiere GPU libre tras qwen). Refrescar al completar qwen.
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
expansión/normalización. Refuerza Flag 142 (candidato a remover del abstract). La expansión
real (fix D11) se evaluará limpiamente en exp13.

## 3. Fidelidad (exp12, PRELIMINAR — 3 modelos)

Fidelidad NLI media (respuestas con claims verificables) y declinación honesta:

| Escenario | Granite 4.1 | Gemma 4 E4B | Mistral 7B |
|---|---|---|---|
| Sin RAG | 0,000* | 0,000* | 0,000* |
| RAG léxico | 0,170 (decl 66 %) | 0,331 (55 %) | 0,222 (25 %) |
| RAG denso | 0,193 (62 %) | 0,322 (51 %) | 0,258 (21 %) |
| RAG híbrido | 0,202 (62 %) | 0,268 (53 %) | 0,256 (24 %) |

`*` Sin RAG = 0 por construcción (sin contexto no hay claim que verificar; N3).

Estadística pareada (familias BH por RQ; Wilcoxon+d_z+bootstrap; McNemar para binarias):
- **RAG ≫ Sin RAG** en fidelidad: TODOS los modelos, d_z 0,67–0,95, p_BH<0,001. RAG aporta
  groundedness frente al LLM solo.
- **Entre métodos de RAG (léxico/denso/híbrido): n.s.** dentro de cada modelo (d_z ~0,02–0,23).
  **El método de retrieval NO se traduce en mayor fidelidad de generación** — la ventaja del
  híbrido vive en la *calidad de retrieval* (§1), no aguas abajo.
- **Entre modelos:** gemma ≥ mistral ≥ granite en fidelidad de respuestas contestadas (varios
  sig tras BH), pero **McNemar de declinación es altamente significativo** en todos los pares.

## 4. Sorpresas

1. **La superioridad del híbrido NO propaga a la fidelidad** (métodos de RAG n.s.). Hallazgo
   honesto y central: el aporte del híbrido es de ranking, no de fidelidad de respuesta.
2. **Circularidad del oráculo infla masivamente** el retrieval del híbrido (0,995 → 0,740).
3. **Determinismo desigual a temp=0:** granite y qwen3.5 deterministas; **gemma y mistral NO**
   (no-determinismo de kernel Ollama). Headline = granite (determinista + RAG-tuned).
4. **Sesgo de extracción de claims entre modelos:** gemma produce ~la mitad de claims
   extraíbles (n_eff≈100) que granite (≈189) → la fidelidad entre modelos está confundida por
   estilo de salida.
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
- Completar qwen3.5:9b (exp12) → refrescar §3/§5 y Tablas 5/6 (la columna Qwen ya está en las tablas).
- exp13: expansión ON vs OFF (30 q cross-cloud, fix D11) con generación determinista (GPU libre tras qwen).
- Export final + figuras; reescritura A.3/A.1.

Artefactos: `output/tables/nota3/` (oracle_stability, tabla6_fidelidad, tabla6_declinacion,
tabla5_modelo_principal, latency); `experiments/results/exp{11,12}_*`;
`paper/audit_findings_cc_addenda.md` (N1/N2/N3).
