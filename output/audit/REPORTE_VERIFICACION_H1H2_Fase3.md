# Reporte de verificación — H1/H2 + Fase 3 (N8, 2026-06-30)

Reporte solicitado por Enzo. Detalle, no resumen. Todas las cifras reproducidas en `py 3.14` (GPU).
**Nada pusheado.** Evidencia firmada exp1-13 intacta; recálculos en archivos `_v3` nuevos.

## 1. Intérprete y modelo NLI — qué se corrigió

**Intérprete.** El repo tiene 4 Python (`py --list`). El `python` del PATH es 3.11 **sin stack ML**
(por eso el primer `pytest` saltó los 3 tests NLI). El intérprete con el stack es:
`C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe` — **py 3.14.3, torch 2.10.0+cu126,
CUDA=True (RTX 3060)**, sentence-transformers, numpy/scipy/statsmodels. Toda la parte NLI/stats usa
ESE intérprete (ruta completa; el launcher `py -V:3.14` flaqueaba).

**Causa del fallo de `small`.** NO era un `HF_HOME`/caché distinto. El modelo `small` simplemente
**no estaba en ningún lado**: ni en la caché HF (`~/.cache/huggingface`) ni como snapshot local.
El `base` sí existía como snapshot local (`data/models/nli-deberta-v3-base/`); el `small` no. Con
`HF_HUB_OFFLINE=1`, `CrossEncoder('cross-encoder/nli-deberta-v3-small')` busca en la caché HF, no
encuentra nada → `OSError`. Además el cliente HF está bloqueado por interceptación TLS en esta
máquina (documentado en N5).

**Corrección (descarga one-time).** `huggingface_hub.snapshot_download(...)` se colgó en el handshake
TLS (timeout 3 min). Fallback: `curl --ssl-no-revoke` de cada archivo del repo HF a
`data/models/nli-deberta-v3-small/`:
`config.json (1052 B), model.safetensors (567.605.820 B ≈ 142M params fp32), tokenizer.json (8.656.624 B),
tokenizer_config.json (1346 B), spm.model (2.464.616 B), special_tokens_map.json (301 B),
added_tokens.json (26 B)`. Luego se apuntó a la ruta local: `rescore_nli_v3.model_path('small')` y un
fallback en `HallucinationDetector.nli_model` (si el tag HF no resuelve offline, carga el snapshot local).

**Verificación de que carga y calibra (no keyword_fallback).**
- Carga offline OK. `id2label = {0:contradiction, 1:entailment, 2:neutral}` — **idéntico a base**
  (importante: `decide_nli_status` asume `score[0]=contr, score[1]=ent`).
- Softmax sano: par entailment ("The capital of France is Paris." / "Paris is the capital of France.")
  → `[contr 0.0, ent 0.998, neu 0.002]`; par no-relacionado ("S3 stores objects…" / "Kubernetes was
  created by Microsoft.") → `[contr 1.0, ent 0.0, neu 0.0]`.
- `pytest` bajo 3.14: **7 passed / 1 skipped** (antes 5/3): los 2 tests de calibración por-detector ahora
  corren contra el modelo real. (El 1 skip carga el tag HF directo, sin pasar por el fallback.)
- **Prueba dura de que corrió el NLI y no el keyword_fallback:** el re-score v3-small tiene
  `contradicted = 1696`. `_keyword_match_single` SOLO emite `supported`/`unsupported`, jamás
  `contradicted` → un contradicted>0 prueba que el cross-encoder corrió. (El re-score además carga
  `CrossEncoder` directo y llama `model.predict(apply_softmax=True)` + `decide_nli_status`; nunca llama
  `detector.check()`, así que el keyword_fallback es estructuralmente imposible en esa ruta.)

## 2. Fase 3 — qué verificador corrió en cada pasada (NO mezclar)

Hubo DOS corridas de Fase 3. Se distinguen explícitamente:

| Corrida | Cuándo | Verificador | retrieval RAG-vs-RAG | entre-modelos | Estado |
|---|---|---|---|---|---|
| **#1 (inicial)** | ANTES de restaurar small | **BASE** (small no disponible) | 0/12 n.s. | **6/18 sig** | **REEMPLAZADA** como primaria |
| **#2 (final)** | DESPUÉS de restaurar small | **SMALL** (el del paper) | 0/12 n.s. | **2/18 sig** | **PRIMARIA** |

- El primer "se sostiene / entre-modelos 6/18" que reporté corrió con **BASE**, porque el small aún no
  cargaba. Se marca explícitamente como **reemplazado** por la corrida #2.
- **Primario = SMALL** (mismo verificador que la Tabla 6 publicada) → entre-modelos **2/18**.
- **BASE queda como cruce de auditoría** → entre-modelos 6/18.
- No se combinan en una sola cifra: cada número lleva su verificador.

Control que aísla H1 del verificador (mismo verificador, artefactos IN vs OUT):
- small: publicado-v2 (artefactos IN) 0/18 → v3-small (artefactos OUT) **2/18**.
- base:  N5-base (artefactos IN) 0/18 → v3-base (artefactos OUT) **6/18**.
- Cruzado: v2-small (IN) 0/18 vs N5-base (IN) 0/18 → cambiar small↔base con artefactos NO crea
  significancia. ⇒ el flip lo produce **la exclusión de artefactos (H1)**, no el verificador.

## 3. H2 — tabla de trade-off COMPLETA (todas las variantes)

Verificador **base**, 1690 claims genuinos (195 artefactos excluidos), muestra estratificada +
q085 + base-rate sintético n=200. `contr/supp/unsup` = conteos de claims.

| variante | contr | supp | unsup | flips (v0→otra) | q085 (28 claims) | base-rate falso-contr (vs 5 chunks aleatorios) |
|---|---|---|---|---|---|---|
| v0 (legacy) | 391 | 383 | 916 | — | 26 contr | **62.0 %** |
| va_margin δ=0.05 | 351 | 383 | 956 | 40 | 19 contr / 7 unsup | 59.0 % |
| va_margin δ=0.10 | 339 | 383 | 968 | 52 | 18 contr / 8 unsup | 57.5 % |
| **vb_agree (≥2 chunks)** | **133** | 383 | 1174 | 258 | **5 contr / 21 unsup** | **17.5 %** |

Lecturas:
- `supp = 383` constante en las 4 variantes ⇒ **la fidelidad (supported/denominador) es H2-invariante**;
  H2 solo reordena `contradicted ↔ unsupported`, no toca el numerador.
- `va_margin` casi no mueve el ruido (62→57-59 %); **`vb_agree` lo desploma (62→17,5 %)** y desinfla
  q085 (26→5). **Elegida: `vb_agree`.** El falso-contradicted residual 17,5 % es la cota del propio
  verificador en dominio técnico (limitación declarada).

## 4. Fase 3 FINAL — números re-corridos (small primario, base cruce)

`faithfulness_answered` primaria (pareado por intersección de no-declinadas; familias BH; `[n<60]` =
potencia limitada). Fuentes: `faithfulness_metrics_v3_small.json` (primario),
`faithfulness_metrics_v3.json` (base, cruce).

### 4.1 SMALL (PRIMARIO) — retrieval-vs-fidelidad (RAG-vs-RAG): 0/12 significativos

| par | n | d_z | p_BH | sig |
|---|---|---|---|---|
| gemma denso-vs-híbrido | 32 | −0.22 | 0.446 | no |
| gemma denso-vs-léxico | 32 | +0.33 | 0.123 | no |
| gemma híbrido-vs-léxico | 32 | +0.24 | 0.178 | no |
| granite denso-vs-híbrido | 63 | +0.18 | 0.240 | no |
| granite denso-vs-léxico | 51 | +0.00 | 0.752 | no |
| granite híbrido-vs-léxico | 53 | −0.31 | 0.084 | no |
| mistral denso-vs-híbrido | 109 | −0.07 | 0.446 | no |
| mistral denso-vs-léxico | 99 | −0.09 | 0.446 | no |
| mistral híbrido-vs-léxico | 93 | −0.01 | 0.932 | no |
| qwen denso-vs-híbrido | 31 | −0.38 | 0.084 | no |
| qwen denso-vs-léxico | 20 | −0.15 | 0.603 | no |
| qwen híbrido-vs-léxico | 20 | +0.03 | 0.902 | no |

### 4.2 SMALL (PRIMARIO) — entre-modelos: 2/18 significativos

Significativos:
| par | n | d_z | p_BH |
|---|---|---|---|
| denso: granite-vs-mistral | 75 | +0.42 | 0.0136 |
| léxico: gemma-vs-mistral | 49 | −0.45 | 0.0442 |

Los otros 16 pares n.s. (rango d_z −0.36..+0.44, p_BH 0.014..0.605). El más cercano no-sig:
denso granite-vs-qwen d_z=+0.44 p_BH=0.164 n=39.

### 4.3 SMALL — RAG ≫ sin-RAG (control de que RAG aporta)
Significativo para granite/gemma/mistral en los 3 escenarios (d_z −0.86..−1.17, p_BH<0.001,
n 59-132). **qwen NO testeable** (n=5-7 por 177/194 respuestas vacías de qwen sin_rag, N6).

### 4.4 BASE (CRUCE DE AUDITORÍA) — entre-modelos: 6/18 significativos

| par | n | d_z | p_BH |
|---|---|---|---|
| denso: granite-vs-qwen | 39 | +0.60 | 0.008 |
| denso: mistral-vs-qwen | 46 | +0.49 | 0.021 |
| léxico: gemma-vs-granite | 45 | −0.61 | 0.008 |
| léxico: gemma-vs-mistral | 49 | −0.53 | 0.008 |
| léxico: granite-vs-qwen | 34 | +0.49 | 0.022 |
| léxico: mistral-vs-qwen | 38 | +0.50 | 0.022 |

Base retrieval RAG-vs-RAG: 0/12 (igual que small).

### 4.5 Niveles absolutos (primaria) — v2-small publicado → v3-small corregido
| modelo | léxico | denso | híbrido |
|---|---|---|---|
| granite | 0.234→0.235 | 0.255→0.247 | 0.293→0.299 |
| gemma | 0.381→**0.443** | 0.356→**0.425** | 0.312→**0.363** |
| mistral | 0.245→0.246 | 0.275→0.281 | 0.282→0.288 |
| qwen | 0.246→**0.361** | 0.246→**0.351** | 0.293→**0.354** |
granite/mistral ~igual (±0.01); gemma/qwen suben +0.05..0.11 (les sacamos 20-23 % de artefactos).

### 4.6 ¿El "se sostiene" original sigue siendo cierto? — explícito

- **"El método de retrieval NO mueve la fidelidad" (RAG-vs-RAG n.s.): SÍ, SE SOSTIENE.** 0/12 bajo
  **small (primario) Y base (cruce)**. Robusto bajo ambos verificadores. Resultado, no predicción.
- **"Entre-modelos TODO n.s." (afirmado en N5): NO, YA NO ES CIERTO.** Bajo small (primario) hay
  **2/18** pares significativos; bajo base, 6/18. El "todo n.s." era artefacto del contaminante de
  formato asimétrico (gemma 23 % / qwen 20 % / mistral 2 %), que enmascaraba diferencias reales.
  **Este punto queda corregido** (N5 → N8). El "se sostiene" que reporté al inicio para ESTE punto
  corrió con base y era, además, provisional; queda reemplazado por small 2/18.
- **Réplica entre verificadores de los 2 pares significativos bajo small (peso para el A.3):**
  - **léxico gemma-vs-mistral: significativo bajo LOS DOS verificadores** (small d_z=−0,45
    p_BH=0,044; base d_z=−0,53 p_BH=0,008) → evidencia robusta, **peso alto**.
  - **denso granite-vs-mistral: significativo SOLO bajo small** (small d_z=+0,42 p_BH=0,014); bajo
    base es marginal-no-sig (d_z=+0,26 p_BH=0,053) → evidencia **más débil**, reportar con cautela /
    menor peso.

**Neto:** el hallazgo central de la tesis (mejor retrieval ≠ más fiel) se sostiene con números y bajo
2 verificadores; la afirmación secundaria "entre-modelos todo n.s." se corrige a "2/18 sig (small)".

## Estado del repo (para contexto, sin acción)
`main` = `ad23f74` (14 commits sin push). `pre-corpus-rebuild-2026-05-21` congelada en `66cd0cf`.
Ledger N8 en `paper/audit_findings_cc_addenda.md`. **PARADO antes de Fase 5/push por indicación de Enzo.**
