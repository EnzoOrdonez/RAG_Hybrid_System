# Acuerdo triple de jueces — referencia humana vs dos jueces LLM ciegos

> Generado por [Kimi Work] — 2026-08-30.
> Tercer pilar de la validación humana: además del auto-acuerdo (retest) y la
> adjudicación, dos jueces LLM independientes anotaron los mismos 200 ítems en
> condiciones de cegamiento total (sin ver juicios humanos ni del otro LLM).

## Diseño

- **Ítems**: 200 juicios claim–condición sobre **150 claims únicos** del gold v4 —
  etapa A (150 claims, 1 chunk) y etapa B (un subconjunto pareado de 50 de esos
  claims, con los 5 chunks completos). La κ global mezcla dos condiciones de
  evidencia; por eso se presentan primero los resultados por condición y la global
  queda como resumen secundario.
- **Jueces**:
  - *Humano* (Enzo): `claim_audit_sample_v4.csv` / `claim_audit_sample_v4_stageB.csv`
    (columna `juicio_humano`, versión fusionada post-adjudicación).
  - *LLM-1* (Codex): `claim_audit_sample_v4_llmjudge_blind.csv` /
    `claim_audit_sample_v4_stageB_llmjudge_blind.csv` (columna `juicio_llm`).
  - *LLM-2* (Kimi): `claim_audit_sample_v4_kimi_blind.json`
    (150 ítems A anotados el 2026-08-29; 50 ítems B anotados el 2026-08-30,
    siempre desde `gold_v4_blind_items.json`, sin acceso a los CSV fusionados).
- **Emparejamiento** por `(query_id, claim)`; se cuentan solo pares con juicio
  válido en ambos lados (de ahí n=197/200 en algunos pares: hay claims duplicados
  en la muestra que colapsan la clave).
- **Métricas**: acuerdo crudo y κ de Cohen, en dos granularidades:
  3 clases (correcto / incorrecto / dudoso) y binaria (correcto vs no-correcto),
  que es la que usan las comparaciones con HHEM/anchor del estudio.

## Resultados

### Etapa A (150 ítems, evidencia = 1 chunk)

| Par | n | Acuerdo 3 clases | κ₃ | Acuerdo binario | κ₂ |
|---|---|---|---|---|---|
| Humano – Kimi | 148 | 36,5 % | +0,061 | 49,3 % | +0,102 |
| Humano – Codex | 150 | 39,3 % | +0,095 | 48,7 % | +0,076 |
| **Kimi – Codex** | 148 | **84,5 %** | **+0,581** | **91,9 %** | **+0,692** |

### Etapa B (50 ítems, evidencia = 5 chunks)

| Par | n | Acuerdo 3 clases | κ₃ | Acuerdo binario | κ₂ |
|---|---|---|---|---|---|
| Humano – Kimi | 49 | 61,2 % | +0,176 | 71,4 % | +0,322 |
| Humano – Codex | 50 | 68,0 % | +0,204 | 72,0 % | +0,234 |
| **Kimi – Codex** | 49 | **81,6 %** | **+0,585** | **83,7 %** | **+0,622** |

### Global (A + B, sin solape de claves)

| Par | n | Acuerdo 3 clases | κ₃ | Acuerdo binario | κ₂ |
|---|---|---|---|---|---|
| Humano – Kimi | 197 | 42,6 % | +0,138 | 54,8 % | +0,204 |
| Humano – Codex | 200 | 46,5 % | +0,166 | 54,5 % | +0,171 |
| **Kimi – Codex** | 197 | **83,8 %** | **+0,667** | **89,8 %** | **+0,754** |

### Distribuciones por juez

| Juez | Etapa A (C/I/D) | Etapa B (C/I/D) |
|---|---|---|
| Humano | 88 / 44 / 18 | 39 / 4 / 7 |
| Codex | 27 / 109 / 14 | 37 / 13 / 0 |
| Kimi | 20 / 124 / 6 | 31 / 17 / 2 |

## Lectura

1. **Los dos jueces LLM son altamente concordantes entre sí bajo el protocolo fijado,
   pero sistemáticamente más estrictos que el humano.** Kimi–Codex alcanzan κ₂ = 0,754
   (89,8 % de acuerdo binario) sin haberse visto; esta concordancia intermodelo no
   establece corrección ni estabilidad ante otros prompts, modelos o ejecuciones. Ambos
   marcan como `incorrecto` la gran mayoría de lo que el humano acepta (etapa A: humano
   59 % correcto vs LLMs 13–18 %). La inspección cualitativa sugiere que parte de la
   discrepancia se concentra en claims multi-parte parcialmente respaldados, meta-claims
   autorreferentes y fragmentos de encabezado/enlace; esta atribución no fue evaluada
   cuantitativamente.
2. **La evidencia ampliada acerca al humano y a los LLM.** En el subconjunto pareado de
   50 claims (cinco chunks), el acuerdo binario humano–LLM sube de ~49 % a ~72 % y κ₂
   humano–Kimi pasa de 0,102 a 0,322. Es consistente con la sensibilidad A→B ya
   reportada (18/50 flips, 12 hacia `correcto`): con un solo chunk se subestima el
   soporte; al ampliar la evidencia visible los criterios convergen.
3. **Consecuencia para el estudio**: bajo este protocolo, los jueces LLM no son
   intercambiables con la referencia humana —presentan distribuciones y criterios
   sistemáticamente distintos—; esto no implica que todos los desacuerdos sean errores
   del LLM ni convierte al humano en ground truth (su estabilidad intra-anotador es
   55 %). La triangulación es evidencia de validez de constructo: la evaluación depende
   tanto del juez como de la evidencia visible.
4. **Nota de integridad**: los juicios LLM quedaron fijados *antes* de conocer
   cualquier juicio humano (cegamiento verificado por construcción: los archivos
   ciegos no contienen `juicio_humano`). Las divergencias no se "corrigieron"
   post-hoc; se reportan tal cual.

*Fin del reporte — [Kimi Work] 2026-08-30.*
