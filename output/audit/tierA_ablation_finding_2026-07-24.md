# Tier A — ablación de contexto (exp15_ablation_tierA): la fidelidad NO responde a las perillas de recuperación

**Fecha:** 2026-07-24 · **Fase:** verano, Tier A · **Estado:** para revisión de Enzo
**Regla:** report-before-prose. NO cambia cifras firmadas (exp15 nuevo). No tocar prosa A.3 sin OK.
**Evidencia:** `experiments/results/exp15_ablation_tierA/{results.json, faithfulness_rows__{small,base}__vb_agree.json, arm_stats__{small,base}.{json,md}}`

## Diseño
5 brazos × 60 queries (subset estratificado seed 42), granite4.1:8b temp 0 seed 42, generación nueva
(exp15). El contexto viene de los ids firmados de exp11 (`exp11_retrieval194_fullrerank`), **transformado
sin re-recuperar**:

| Brazo | Transform | Hipótesis que testea |
|---|---|---|
| `baseline_repro` | identidad (ids híbrido full-rerank) | ancla / reproduce exp12 hibrido |
| `reranker_off` | ids RRF pre-rerank | ¿el reranking cross-encoder sube la fidelidad? |
| `final_top_k_3` | top-3 en vez de top-5 | ¿menos contexto = menos/más fidelidad? |
| `context_reversed` | orden invertido | sensibilidad al orden |
| `context_lost_middle` | permutación (relevante al centro) | lost-in-the-middle |

Contraste **pareado within-session por query_id** vs `baseline_repro` (Wilcoxon + d_z + bootstrap seed 42,
familia BH de 4). Decline-aware: pares con fidelidad None se descartan; vacuous=1.0. Métrica = rescore
vb_agree τ0.7 (mirror v4). Passes desacoplados (G generación, N scoring) por protocolo H5.

## Resultado — NULO ROBUSTO: ninguna perilla de recuperación mueve la fidelidad

**0/4 brazos significativos bajo NLI small Y bajo NLI base (robusto small∩base). Todos los d_z
despreciables (|d_z| ≤ 0.19).**

| Brazo | det3x | NLI small: Δ (d_z, p_BH) | NLI base: Δ (d_z, p_BH) | Veredicto |
|---|---|---|---|---|
| reranker_off | ✓ | −0.058 (−0.14, 0.60) | −0.050 (−0.15, 0.82) | n.s.; tiende **abajo** en ambos |
| final_top_k_3 | ✗ H5 | −0.019 (−0.06, 0.60) | +0.029 (+0.09, 0.53) | n.s.; signo inconsistente |
| context_reversed | ✓ | +0.043 (+0.19, 0.60) | +0.047 (+0.19, 0.53) | n.s.; leve arriba en ambos |
| context_lost_middle | ✗ H5 | +0.027 (+0.10, 0.60) | −0.005 (−0.02, 0.53) | n.s.; signo inconsistente |

Nivel `baseline_repro`: 0.308 (small) / 0.204 (base) sobre 60/60.

### Lectura
1. **La fidelidad medida por NLI es insensible a las perillas de recuperación.** Quitar el reranker,
   recortar a top-3, invertir el orden o mandar lo relevante al centro **no cambia** la fidelidad de
   granite de forma detectable. Esto **aísla el enigma central al lado de la GENERACIÓN**: incluso
   degradando la composición del contexto, el generador no ancla mejor ni peor.
2. **Lost-in-the-middle: NO respaldado.** Los signos se contradicen entre small (+0.027) y base
   (−0.005), ambos despreciables. A este tamaño de contexto (top-5), granite no muestra la degradación
   clásica en fidelidad.
3. **El reranking no aporta fidelidad.** `reranker_off` tiende ligeramente **abajo** en ambos
   verificadores (−0.05..−0.06, n.s.) — o sea, ni siquiera hay una tendencia a que reordenar mejore el
   anclaje. Converge con el hallazgo central: las ganancias de calidad de recuperación (NDCG@5 0.74 vs
   0.44) no se traducen en fidelidad.

## Deriva de entorno H5 (baseline_repro julio vs exp12 hibrido junio, NLI small, 59 q)
- Deriva media **+0.033** (Wilcoxon p=0.083, **n.s.**), |deriva por-query| media 0.087, **corr r=0.858**.
- **39% (23/59) de queries reproducen la fidelidad EXACTA**; el resto deriva (un caso con swing total
  en respuesta de pocos claims).
- **Implicación de diseño (importante):** la deriva (+0.033) es del mismo orden que los efectos de brazo
  (reranker_off −0.058, reversed +0.047) → los contrastes de brazo **tienen que ser pareados
  within-session vs baseline_repro** (lo son), NUNCA contra junio. La decisión de relajar el gate
  (advertir+warmup en vez de abortar) queda **validada**: el ancla deriva poco (+0.03, n.s.) y la
  estructura por-query se conserva (r=0.86).

## Triangulación instrumental HHEM — el nulo de Tier A SÍ es robusto al instrumento
Dado que Tier 3 mostró que el NLI enmascara efectos (HHEM revela granite híb>léx, 1/12), re-scoreé los 5
brazos con HHEM (carga verificada: baseline_repro nivel 0.450 = exp12 granite hibrido HHEM 0.40-0.44, no
basura). **HHEM coincide con NLI: 0/4, todos despreciables** (context_reversed d_z 0.21 pequeño, p_BH 0.96).

| Brazo | NLI small p_BH | NLI base p_BH | HHEM p_BH | Δ HHEM (d_z) |
|---|---|---|---|---|
| reranker_off | 0.60 | 0.82 | 0.96 | +0.007 (0.02) |
| final_top_k_3 | 0.60 | 0.53 | 0.96 | −0.006 (−0.01) |
| context_reversed | 0.60 | 0.53 | 0.96 | +0.060 (0.21) |
| context_lost_middle | 0.60 | 0.53 | 0.96 | −0.025 (−0.09) |

Nivel HHEM por brazo: baseline 0.450, reranker_off 0.438, top_k_3 0.438, reversed 0.469, lost_middle 0.424.

### Contraste clave con Tier 3 (esto es lo publicable)
| | Tier 3 (entre-escenarios: léxico/denso/híbrido) | Tier A (transforms del MISMO pool híbrido) |
|---|---|---|
| 0/N bajo NLI | 0/12 | 0/4 |
| Bajo HHEM | **1/12** (granite híb>léx CRUZA sig) | **0/4** (sigue nulo) |
| ¿Robusto al instrumento? | **NO** | **SÍ** |

**Lectura mecanística:** la fidelidad responde (débil, granite, solo-HHEM) a **QUÉ documentos** selecciona
el *método* de recuperación (híbrido vs léxico, NDCG 0.74 vs 0.44), pero **NO** a cómo se arregla un pool
ya recuperado — reranking, top-k, orden y lost-in-the-middle son nulos en los TRES instrumentos. O sea: el
efecto pequeño que existe es de **selección de contenido**, no de **ordenamiento/reranking/recorte**.
Esto acota fuerte el diagnóstico y descarta lost-in-the-middle y el reranking como palancas de fidelidad.

## Implicación para A.3/LACCI (report-before-prose)
- Refuerza el hallazgo central desde la generación: "mejor recuperación ≠ mejor fidelidad" se sostiene
  incluso **degradando** el contexto (reranker off, lost-middle). El cuello de botella no es el orden
  ni el reranking del contexto.
- Candidato a **discusión/limitaciones**, no cambio de cifras: la ablación de contexto es nula bajo NLI;
  pendiente confirmación bajo HHEM. Lost-in-the-middle no observado a top-5.
- NO tocar prosa sin OK frase por frase.

## Pendiente
- ~~HHEM sobre los 5 brazos~~ **HECHO** (arriba): 0/4, nulo robusto al instrumento.
- deberta-large como 3.er voto (8/12 parcial, resumible) — bajo valor, el nulo ya triangula 3 instrumentos.
- Gold humano arbitra el nivel absoluto (NLI 0.30 vs HHEM 0.45).
