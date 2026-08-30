# exp18 — por que 759 claims no los soporta ningun chunk del pool

Sobre 188 queries y 2053 claims genuinos. tau=0.5, soft_tau=0.3. **Pre-registrado** en el docstring del script antes de mirar salida; DESCRIPTIVO, fuera de la familia BH.

## H1 — ¿el modelo se desancla al avanzar, tras señalar que le falta contexto?

**H1 v1 retirada, nunca computada:** «claims despues del marcador vs antes» es degenerada por construccion — `pure_decline` is DEFINED as a marker inside the first 300 chars, so the 'before' stratum is empty. Observed marker offsets over 84 decline-prefixed answers: p50=0, max=253 chars.

Diferencia-en-diferencias sobre el gradiente intra-respuesta (2.ª mitad − 1.ª mitad de los claims). 49 respuestas con prefijo de declinacion vs 72 `answered`; 38 excluidas por <4 claims.

| grupo | gradiente |
|---|---|
| `answered` (control) | 0.0834 |
| **prefijo de declinacion** | **0.1275** |
| **DiD** | **0.0442** (IC95 -0.0694 a 0.1601) |

IC95 que cruza 0 = el gradiente no distingue a los dos grupos, y la hipotesis de memoria parametrica **no** queda apoyada por esta via.

## Estratos candidatos (NO son veredictos)

| estrato | n | % | mejor score medio |
|---|---|---|---|
| `d_threshold_artifact` | 123 | 16.2 % | 0.4501 |
| `a_synthesis_cand` | 58 | 7.6 % | 0.3581 |
| `b_parametric_cand` | 178 | 23.4 % | 0.1651 |
| `c_unattributed_cand` | 400 | 52.7 % | 0.1893 |

(a) synthesis and (c) hallucination cannot be separated by a grounding model -- that is what the human gold is for. These are strata for annotation. A verdict here would be the overclaim this phase keeps catching.

Muestra estratificada para anotacion humana: `output/audit/unsupported_claims_sample.csv` (40 claims).