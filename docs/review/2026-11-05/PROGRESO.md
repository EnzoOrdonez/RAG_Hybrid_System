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
