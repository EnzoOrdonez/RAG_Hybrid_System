# Progreso 2026-11-05: fases P1-P5 (base 3bb9ca8)

Cada fase vive en su rama, creada desde `3bb9ca8`, y cada entrada se escribe después del push. Capas = `bash init.sh` sin `PLAYWRIGHT_BROWSERS_PATH`. Las ramas no se fusionan ni abren PR.

## P1 determinismo: `claude/p1-determinismo` @ `c60aa15` (05:50-05:58 UTC)
- **Capas:** 1=0, 2=0, 3=0.
- **Inventario:** 41 archivos con estadística o aleatoriedad. Se ejecutaron 31 scripts con PYTHONHASHSEED 0, 7 y 42, cada uno en su worktree.
  - 17 deterministas bit a bit, contando `compute_exp15_nli_sweep` con un tope de 900 s: solo difieren los tiempos impresos.
  - **1 no determinista: `compute_exp17_powered.py`** (`:46` y `:84`, orden de un `set`).
  - 13 no verificados: entradas no versionadas, modelos de HF, la guarda de evidencia, telemetría de Windows.
- **Arreglo:** el parche 005 (qids ordenados y salida a `_v2` con `guard_write`) y `tests/test_p1_determinism.py`: 4 tests que fallan sin el arreglo y pasan con él. El arnés sobre `c60aa15` da exp17 = **determinista**.
- **Efecto en exp17** (n_boot = 10000; firmado → arreglado): p del bootstrap small 0,302 → 0,2906, base 0,1428 → 0,1418, hhem 0,2329 → 0,2375. La GLMM no cambia. Nada cruza 0,05, y el valor firmado no se reproduce con ninguna de las semillas 0, 7 o 42.
- **No verificado:** los 13 de arriba, 6 scripts excluidos (LLM, red o telemetría), Windows y Python 3.14.3.
- **No cometido:** `powered_reanalysis_v2` en `experiments/`, porque toca una cifra entregada y la decisión es de Enzo.

## P2 trazabilidad: `claude/p2-trazabilidad` @ `add1abd` (05:58-06:02 UTC)
- **Capas:** 1=0, 2=0, 3=0.
- **Barrido automático por valor:** camera-ready 69/85, README 47/56, results-summary 110/113, summer-benchmarks 141/141, main.tex 8/8.
- **Verificación a mano:** retrieval del camera-ready 16/16 (exp11 bge-indep, n = 194); fidelidad 12/12 (exp12 `faithfulness_metrics_v4_small`); d_z y corpus correctos.
- **Discrepancias:**
  - **D1 (S1):** el abstract de `main.tex` (borrador) afirma el +16,8 % retirado y d = 0,626 de forma ambigua.
  - **D2 (S2):** la Tabla 6 de `results-summary.md` es la v2, no la del camera-ready (v4_small).
  - **D3 (S3):** las latencias del README no tienen fuente en el repo.
  - **D4 (S3):** «HHEM 0.23» pasaría a 0,24 con P1.
- **No verificado:** la prosa de A.3 y de la ficha (no están en el repo), 76 `docs/*.md` no barridos y los registros fuera del repo.

## P3 higiene: `claude/p3-higiene` @ `0545339` (06:04-06:08 UTC)
- **Capas:** 1=0, 2=0, 3=0.
- **R-03:**
  - 7 patrones nuevos: Google OAuth y API, Hugging Face, GitHub fine-grained y app/OAuth, Slack y `private_key_id`.
  - Los archivos de clave se informan por presencia, sin abrirse.
  - `tests/test_p3_scan_secrets.py`: 19 tests; con el escáner de 3bb9ca8 fallan 10.
  - El repo queda en 0 hallazgos sin ampliar el baseline.
- **R-16:** 488 apariciones en 3bb9ca8 (correo 137, IPv4 pública 123, ID de proyecto 75, ruta de Windows 68, `/home` 57, cuenta de servicio 23, sslip 5), con 0 tokens. La mayoría está en datos de corpus o salidas de LLM; `docs/review/**` tiene 0.
  - **S2 fuera del corpus:** correspondencia en `titulacion/` con correos de terceros, el correo ACME en `deployment.py:125` (zona congelada), la cuenta de servicio y el ID de proyecto en 43 archivos.
  - **Plan:** sin reescribir historia.
- **No verificado:** la historia de git anterior, los binarios `.docx`/`.pdf` y el consentimiento del correo personal en los `.tex`.

## P4 salud de tests: `claude/p4-tests` @ `23bd15b` (06:09-06:47 UTC)
- **Capas:** 1=0, 2=0, 3=0.
- **F-016:** la causa son los plazos del fixture `supervisor()`: preparación de 3 s y llamada de 0,1 s. Bajo carga, el arranque del worker sube a unos 10 s y el viaje de una llamada llega a 112 ms.
  - **Antes:** 30 de 30 fallan con 128 procesos de carga.
  - **Después:** 0 de 10 fallan con la misma carga (archivo completo).
  - El arreglo está solo en el test; `scripts/` no se toca.
- **Mutación, antes → después:**
  - `retrieval_metrics`: 46/63 → 57/63.
  - `hybrid_index` (RRF): 0/66 → 32/66 (fusión 32/36).
  - `hybrid_retriever`: 0/22 → 18/22.
  - `hallucination_metrics`: 0/12 → 12/12.
  - `statistical_analysis` (60 sitios): 8/60 → 37/60.
  - Los 42 tests nuevos están en `tests/test_p4_mutation_killers.py`.
- **Hallazgos:**
  - **P4-H1 (S2):** `_linear_fusion` (`hybrid_index.py:149`) desempata según PYTHONHASHSEED; 4 órdenes con 4 semillas. Afecta a exp3.
  - **P4-H2 (S3):** `paired_comparison` con diferencias nulas devuelve p = NaN.
- **No verificado:** la carga real de la suite, Windows y `hallucination_detector` (necesita modelos).

## P5 docs: `claude/p5-docs` @ `82fd492` (06:36-06:43 UTC)
- **Capas:** 1=0, 2=0, 3=0, ejecutadas en el checkout principal. En un worktree, `test_fresh_refuses_legacy_output_and_checkout` falla por entorno (R-39).
- **`CLAUDE.md:37`:** CI no llama a `init.sh`; corre las tres capas por separado y solo en `main`/`summer/*`.
- **Runbook de Fase B y receta de imagen (I8-08):** nota sobre la rama `fix/interview-readiness`, remota y local en HEAD, y `$run`/`$evidence` como marcadores de una raíz y un recibo nuevos.
- **`adr-log.md`:** filas 0020-0026 en el índice (fecha de alta en git; estado no declarado en el archivo) y sección para ADR-0019.
- **No verificado:** el resto de runbooks no se ha revisado.
