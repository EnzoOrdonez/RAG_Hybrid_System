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

## Muestra de calibracion humana

Muestreo estratificado con seed 42: `output/audit/unsupported_claims_sample_v2.csv` (40 claims; 10 por estrato). Cada fila incluye `stratum_size` e `inclusion_prob`; el peso Horvitz-Thompson es `1 / inclusion_prob`.

**n efectivo de Kish = 27.4**.

Las columnas `human_verdict` y `human_notes` se dejan vacias para juicio humano.

### Ejemplos representativos

Dos por estrato: los scores `best_over_pool` mas cercanos a la mediana del estrato; empates resueltos reproduciblemente con seed 42.

| estrato | mediana | query | claim_idx | score | claim |
|---|---:|---|---:|---:|---|
| `d_threshold_artifact` | 0.4492 | q065 | 23 | 0.4492 | Click Create database to provision your RDS instance.. |
| `d_threshold_artifact` | 0.4492 | q091 | 8 | 0.4507 | Enable recommended out-of-the-box alert rules by navigating to your AKS cluster in the Azure portal.. |
| `a_synthesis_cand` | 0.3567 | q128 | 9 | 0.3572 | A VPC is essential when you need to isolate and manage multiple types of AWS services (e.g., EC2, RDS) securely within a private network space. |
| `a_synthesis_cand` | 0.3567 | q167 | 12 | 0.3572 | Both services offer robust scaling mechanisms tailored to their respective use cases—VMs for compute-intensive applications and Blob Storage for large-scale data storage and access. |
| `b_parametric_cand` | 0.1601 | q025 | 0 | 0.1599 | Therefore, I cannot provide details about AWS Lambda pricing tiers from this source. |
| `b_parametric_cand` | 0.1601 | q177 | 8 | 0.1604 | Mitigation strategies include provisioned concurrency, which pre-warms instances to reduce start-up time.. |
| `c_unattributed_cand` | 0.1815 | q072 | 14 | 0.1816 | Ensure you plan accordingly if you need detailed descriptions of resources.. |
| `c_unattributed_cand` | 0.1815 | q107 | 1 | 0.1813 | Enable Stackdriver Logging and Monitoring: Go to Logging under Operations Suite or directly access Cloud Logging.. |