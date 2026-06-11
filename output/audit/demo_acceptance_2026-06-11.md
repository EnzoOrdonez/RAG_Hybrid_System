# Demo estable — evidencia de criterios de aceptación (F6, 2026-06-11)

Hardware: RTX 3060 Laptop 6 GB. Modelo demo: `llama3.1:8b-instruct-q4_K_M`.
Camino medido = el de la UI corregida: `CUDA_VISIBLE_DEVICES=""` (aux en CPU),
warm-up + `keep_alive=30m`, `query_stream` (512 tokens), NLI diferido.

## Antes (UI original, mismo hardware, mismo día)

| medición | valor |
|---|---|
| e2e por consulta (tibia) | 120 s (llama3.1), 77 s (gemma4:e4b) |
| primeros tokens visibles | = e2e (spinner bloqueante, sin streaming) |
| GPU para Ollama | 38 % GPU / 62 % CPU (≈6 tok/s) — embedder+reranker+NLI en CUDA dentro del proceso UI |
| keep_alive | 5 min (default) → recarga fría tras pausa |
| NLI | síncrono dentro del request (+1,4–6,6 s) |
| "se corta" | sin excepción en 4 mediciones: colapso UX (120 s en blanco + reruns apilados) |

## Después (fixes F6)

Ráfaga de 4 consultas (proceso tibio):

| consulta | e2e | primer token tras enviar | TTFT post-retrieval |
|---|---|---|---|
| Q1 (incluye cargas perezosas) | 83,3 s | 33,1 s | 4,7 s |
| Q2 | 50,6 s | 4,9 s | 3,1 s |
| Q3 | 40,5 s | 4,2 s | 1,9 s |
| CROSS (cross-cloud) | **44,2 s** | 4,1 s | 3,2 s |

Sesión simulada de **36 min** (6 consultas, pausa de 12 min + 4×5 min): **0 fallos**.

| consulta | pausa previa | e2e | primer token | NLI (CPU, diferido) |
|---|---|---|---|---|
| S1 | 0 | 81,0 s* | 28,4 s* | 6,4 s |
| S2 | **12 min** | 35,1 s | **5,3 s** | 3,1 s |
| S3 | 5 min | 18,4 s | 4,4 s | 1,6 s |
| S4 | 5 min | 27,0 s | 4,1 s | 0,9 s |
| S5 | 5 min | 55,5 s | 8,9 s | 2,6 s |
| S6 | 5 min | 22,6 s | 4,8 s | 1,2 s |

`*` S1 = primera consulta del proceso (carga perezosa del reranker en CPU, una vez
por proceso). Para el video: 1 consulta de calentamiento antes de grabar.

## Criterios

| criterio | resultado |
|---|---|
| Primeros tokens < 5 s tras retrieval | ✓ (TTFT 1,9–4,7 s) |
| 3 consultas seguidas sin recarga fría | ✓ (keep_alive 30m; `ollama ps UNTIL 29 min`) |
| Consulta tras pausa de 10 min sin fallo | ✓ (12 min: 35,1 s e2e, token a 5,3 s) |
| Sesión 30 min sin un solo corte | ✓ (36 min, 6/6 OK) |
| Cross-cloud < 45 s e2e | ✓ (44,2 s) |
| README "Cómo lanzar la demo" | ✓ |

Nota: medido con una instancia Streamlit antigua reteniendo aún ~3,5 GB de VRAM;
tras reiniciarla con el código nuevo, los tiempos solo pueden mejorar.
