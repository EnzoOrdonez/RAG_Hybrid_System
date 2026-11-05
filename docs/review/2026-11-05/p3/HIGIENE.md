# P3: higiene del repositorio público (base 3bb9ca8)

Rama `claude/p3-higiene`. No se reescribe historia y no se toca `scripts/study_operator/` ni `config/`.

## R-03: huecos del escáner de secretos [EJECUTADO]

**Cambio en `scripts/scan_secrets.py`** (fuera de la zona congelada).
- **7 patrones nuevos:**
  - `google_oauth_token` (`ya29.`);
  - `google_api_key` (`AIza` + 35);
  - `huggingface_token` (`hf_` + 34 o más);
  - `github_fine_grained_token` (`github_pat_22_59`);
  - `github_app_or_oauth_token` (`gh[ousr]_` + 36);
  - `slack_token` (`xox[abprs]-`);
  - `gcp_service_account_key_id` (`"private_key_id": <40 hex>`).
- **Material de clave:** `.pem`, `.key`, `.env`, `.env.*`, `credentials.json` y `secrets.toml` siguen **sin abrirse**, como manda el guardrail de `.env`. Pero ahora, si git los lista (versionados, o sin versionar y no ignorados), se informa su **presencia** como `key_material_file` (línea 0, sin vista previa). Recorriendo el disco con `--root` también se informan.

**Tests:** `tests/test_p3_scan_secrets.py`, 19 tests.
- Hay 7 positivos (detecta el patrón y redacta a 4 caracteres), 9 negativos casi iguales, 2 de presencia (con git y recorriendo el disco, comprobando que nunca se abre el archivo) y 1 del informe.
- Contra el escáner de 3bb9ca8: **10 fallan** (7 positivos, 2 de presencia y 1 del informe) y los 9 negativos pasan. Es lo esperado: los negativos protegen contra falsos positivos.
- **Capa 1 con los patrones nuevos sobre todo el repo:** 0 hallazgos (exit 0) y sin ampliar el baseline.

## R-16: inventario de identificadores [EJECUTADO]

- **Script:** [inventory_identifiers.py](inventory_identifiers.py), sobre `git ls-tree 3bb9ca8` (1 de cada archivo versionado). Salida: [inventory_3bb9ca8.json](inventory_3bb9ca8.json), con valores truncados a 4 caracteres.
- **Exclusiones:** IPs privadas y de documentación (RFC 5737), y correos de `example.*` o `noreply`.

| Categoría | Apariciones | Archivos | En datos de corpus o salidas de LLM | Sev. |
|---|---|---|---|---|
| correo | 137 | 47 | 121 (36 archivos) | S2 |
| IPv4 pública | 123 | 31 | 117 (26) | S3: en código y tests es el rango de IAP de Google, 35.235.240.0/20, que es público |
| ID de proyecto GCP | 75 | 43 | 2 | S2 |
| ruta de usuario de Windows | 68 | 46 | 47 | S3 |
| ruta `/home` o `/Users` | 57 | 26 | 51 | S3 |
| correo de cuenta de servicio | 23 | 12 | 21 | S2: es un nombre, no una credencial |
| host `.sslip.io` | 5 | 4 | 0 | S3: todos son 203.0.113.x (TEST-NET) en tests |
| tokens (`ya29`, `AIza`, `hf_`, `github`) | 0 | 0 | 0 | — |

- **`docs/review/**`:** **0 apariciones** en `origin/claude/review-2026-11-02` y en `origin/claude/review-i8` (ejecutado sobre ambas).

**Identificadores fuera del corpus** (archivo:línea; valor truncado en el JSON):

| Sev. | Dónde | Qué | Plan, sin reescribir historia |
|---|---|---|---|
| **S2** | `titulacion/emails_lewis_y_gyt.md:40,80,86` | correspondencia con correos de terceros (asesor y oficina) | Mover fuera del repo público y declararlo (ADR-0011). Lo decide Enzo |
| S2 | `scripts/study_operator/deployment.py:125` (Caddyfile, ACME) | correo institucional del alumno | Zona congelada. Parametrizarlo en el próximo commit de operador, junto con la imagen |
| S2 | `scripts/study_operator/cpu_build.py:23`, `tests/test_study_operator_failover_owned.py:12` | nombre de la cuenta de servicio | Pasar a configuración (zona congelada) y usar uno ficticio en el test |
| S2 | 43 archivos (23 de código y config, 12 de tests, 6 de docs) | ID de proyecto | Una sola constante (`checked_config`) y un ID ficticio en los tests; ya estaba en R-09/R-16 |
| S3 | `docs/Paper_IEEE_RAG_Hibrido_LACCI_v9.tex:21,27`, `paper/overleaf_ready/main.tex:39,45`, `README_OVERLEAF.md:30` | correos de los autores (uno personal de Gmail) | Normal en un paper; confirmar que Enzo acepta publicar el personal |
| S3 | `.harness/feature_list.json:74`, `paper/summer_ablation_log.md:2423` | correo de contacto de IEEE | Público; no requiere acción |
| S3 | 12 docs con `C:\Users\<usuario>`, como `docs/reports/i7-gpu-restore-01-audit.md` | nombre de usuario de Windows | Sustituir por `%USERPROFILE%` en los docs nuevos; los ledgers append-only no se tocan |
| S3 | datos de corpus y caches de LLM (`experiments/**`, `data/llm_cache/**`) | ejemplos de la documentación pública y texto generado | Sin acción: es contenido de terceros o evidencia firmada inmutable |
| S3 | `output/audit/descriptive_cis.json` (salida de `compute_descriptive_cis.py`) | incrusta rutas absolutas al regenerarse (visto en P1) | Escribir rutas relativas en un `_v2` |

## No verificado
- La historia de git: solo se barrió `3bb9ca8` y las dos ramas de revisión, no los commits antiguos. Un secreto borrado en el pasado seguiría en la historia.
- Binarios `.docx` y `.pdf`.
- Si el correo personal de los autores está consentido.
