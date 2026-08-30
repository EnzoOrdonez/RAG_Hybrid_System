# Sondas de ruido de runtime — NO es evidencia experimental

> **Sección de Claude Code — 2026-08-21 13:55 (hora local).**

Este directorio está **fuera de `experiments/results/`** a propósito. Nada de aquí lo descubre
`verify_summer_offline.py` ni `tests/test_scored_arms_complete.py` (los dos recorren
`experiments/results/` un nivel y filtran por forma de artefacto), ni lo protege ni lo bloquea
`src/utils/signed_evidence.py`. **No entra en ninguna familia BH, ningún TOST y ninguna cifra
citable.** Mide el instrumento, no el sistema.

## `state_A_2026-08-21.json`

Las **70 primeras queries** del brazo `baseline_repro` de exp19b, generadas el 2026-08-21 entre
las 07:25 y las 08:48 con `granite4.1:8b`, temp 0, seed 42, caché off, sobre los contextos
baseline congelados de exp18. La corrida se cortó en la query 79 (proceso terminado desde
fuera) y **no se reanudó**: ver la entrada 26 de `paper/summer_ablation_log.md`.

**Por qué no se borró.** Como checkpoint era una trampa: reanudarlo habría mezclado dos estados
del generador dentro de un mismo brazo, y por eso `run_exp19b_generation.py` ahora lo rechaza
(no lleva `session_fingerprint`). Pero como **muestra** vale: son 70 generaciones ya pagadas en
un estado de runtime concreto —el «estado A»— con contextos y prompts fijados. Regenerar las
mismas queries tras reiniciar Ollama da un pareado del efecto de cambiar de estado de runtime
por la mitad del coste de GPU. Moverlo aquí cierra la trampa y conserva la muestra.

**Lo que NO es:** un brazo, ni una corrida parcial de exp19b que alguien pueda completar. La
corrida real de exp19b arranca de cero con `--no-resume`.
