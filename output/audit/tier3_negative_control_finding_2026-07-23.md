# Tier 3 — Hallazgo mayor: la "baja fidelidad" es en gran parte artefacto del instrumento NLI

**Fecha:** 2026-07-23 · **Fase:** verano, Tier 3 (verificador de fidelidad) · **Estado:** para revisión de Enzo
**Regla aplicada:** report-before-prose. Esto NO cambia ninguna cifra publicada; **contextualiza** la
interpretación del 0,30 de fidelidad del A.3/LACCI. No tocar prosa sin OK frase por frase.

## Qué se hizo
Control negativo de validez de constructo (reproduce el método de `h2_variant_eval.json` que eligió
`vb_agree`): 400 claims genuinos emparejados cada uno con **5 chunks ALEATORIOS no relacionados**
(seed 42, excluyendo los chunks propios), scoreados con los verificadores. Un verificador válido casi
nunca debería etiquetar texto no relacionado como *contradicted* (NLI) o *grounded* (HHEM).
Datos: `experiments/results/exp15_ablation_nli/negative_control_{pairs,scores,rates}.json`.

## Resultado
| Verificador | Falso-positivo en texto ALEATORIO | En datos reales (retrieved) |
|---|---|---|
| NLI small (runtime, vb_agree) | **falso-contradicted 0,237** | fidelidad ≈ 0,30 |
| NLI base (vb_agree) | **falso-contradicted 0,215** | fidelidad ≈ 0,30 |
| NLI (v0 legacy) | 0,54–0,60 | — |
| **HHEM-2.1** (grounding, τ=0,5) | **falso-grounded 0,010** | grounding ≈ 0,99 |

- Los verificadores NLI (deberta entrenado en NLI general) etiquetan **~22 %** de pares aleatorios
  no relacionados como "contradicted". Es un falso-positivo sistemático de contradicción: el instrumento
  inventa contradicciones sobre texto que solo es irrelevante.
- HHEM-2.1 (modelo de grounding RAG-específico, familia ORTOGONAL) tiene **~1 %** de falso-grounded en
  aleatorio (mean 0,002) Y en datos reales da grounding ≈ 0,99 → **discrimina nítidamente** (no es lenient
  global) y ve los claims **como respaldados por la evidencia recuperada**.

## Interpretación (para discutir)
El enigma central de la tesis — "mejor recuperación no mejora la fidelidad (≈0,30)" — se re-enmarca:
el **≈0,30 es en gran parte artefacto del instrumento NLI**, que (a) sobre-dispara contradicción (~22 %
falso en aleatorio) y (b) sub-acredita entailment en verificación de claims sobre documentación técnica.
Un verificador de grounding construido para la tarea (HHEM) considera que las respuestas RAG **sí están
ancladas** en la evidencia (~0,99). Converge con Tier 3-A: small es el verificador runtime, el más
ruidoso (2,5× más frágil al umbral, sobre-contradice 1,8× vs base) y con 128 falso-contradicted.

**Cautela / pendiente (rigor):**
1. HHEM ≈0,99 en datos reales roza el techo → puede NO discriminar entre escenarios (varianza baja).
   Eso NO invalida el punto sobre el NIVEL (0,99 vs 0,30), pero sí implica que HHEM quizá tampoco muestre
   el efecto retrieval→fidelidad (por techo, no por instrumento). Cuantificar con la corrida completa
   (en curso).
2. Falta el **gold humano** (`claim_audit_sample_v4`, N≈200, pendiente de anotar) para arbitrar
   objetivamente: ¿tiene razón HHEM (claims grounded) o el NLI (contradicted)? La selección del
   instrumento se ancla en gold + este control negativo, NUNCA en el contraste downstream (anti-p-hacking).
3. HHEM tokeniza con límite flan-t5 (512); chunks largos se procesan sin truncar (T5 sin límite duro).

## Qué NO cambia
Ninguna cifra firmada. El 0,30 sigue siendo el valor bajo el instrumento NLI publicado. Este hallazgo es
material para una sección de **Limitaciones / discusión del instrumento** y para la recomendación de
verificador de las encuestas — sujeto a confirmación con el gold y OK explícito de Enzo antes de prosa.

## Siguiente
Corrida completa HHEM + deberta-large (en curso, ~5-6 h) → ensemble sweep con los 4 verificadores +
control negativo (selección provisional) → esperar gold humano para selección definitiva → re-medición
pre-registrada del enigma central con el instrumento elegido (viejo vs nuevo lado a lado).
