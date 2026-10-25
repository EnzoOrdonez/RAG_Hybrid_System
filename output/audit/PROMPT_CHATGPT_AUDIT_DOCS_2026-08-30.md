# PROMPT MAESTRO — ChatGPT work — Auditoría de consistencia de documentación (2026-08-30)

Eres auditor externo de documentación de un repositorio de investigación cuyo paper fue
**aceptado en IEEE LACCI 2026** (camera-ready ya certificado por PDF eXpress, Paper ID
2026305869). El repo quedará como artefacto público reproducible del paper, así que la
documentación debe estar **pareja y sin afirmaciones desactualizadas**.

## Alcance (solo lectura, NO modifiques nada)

Revisa estos archivos del repo:

- `README.md`
- `CLAUDE.md`
- `MODELS.md`
- `REPRODUCE.md`
- `SUMMER_RESULTS.md`
- `RESULTADOS_RESUMEN.md`
- `NOTA3_NEXT_STEPS.md`
- `CITATION.cff`
- `docs/ESTADO_PROYECTO_UNIFICADO_2026-08-06.md`
- `docs/FICHA_EXPERIMENTAL.md`
- `docs/SECCION_VALIDACION_HUMANA.md`
- `docs/LIMITACIONES_Y_TRABAJO_FUTURO.md`
- `docs/TRACEABILITY_nota3.md`
- `docs/APP_VS_EXPERIMENTO.md`

## Hechos recientes contra los que debes contrastar

- El repo tiene **19 experimentos versionados** (carpetas exp3..exp19b + exp8b en
  `experiments/results/`) más sondas en `experiments/probes/`.
- Existe un **gold humano v4** completado y adjudicado (150 claims, 200 juicios
  claim–condición, taxonomía de 40; κ mejor verificador = 0,30 ponderada; jueces LLM
  κ₂=0,754 entre sí, 0,17–0,20 vs humano; intra-anotador 55 %).
- exp19b cerrado: Δ HHEM +0,0451 (IC95 [0,0076; 0,0817]), TOST dentro de ±0,081.
- Paper aceptado; v9 camera-ready en `docs/Paper_IEEE_RAG_Hibrido_LACCI_v9.tex`;
  PDF certificado `docs/2026305869.pdf`.
- Rama de trabajo mergeada a `main`; CI en GitHub Actions verde.
- Cifras canónicas del paper: corpus 24.481 chunks (AWS 6.366 / Azure 9.606 / GCP 8.509);
  NDCG@5 oráculo independiente 0,740 (0,995 circular); 194 queries; 4 LLMs
  (Granite 4.1 8B, Gemma 4 E4B, Mistral 7B, Qwen 3.5 9B).

## Qué reportar

Por cada archivo, una de tres etiquetas: **OK** / **DESACTUALIZADO** / **CONTRADICE**, y
para cada hallazgo: cita la línea o fragmento exacto, por qué está mal y qué debería decir
(no necesitas redactar el reemplazo completo, basta la corrección concreta). Busca
específicamente:

1. Números de experimentos / conteos desactualizados ("12 experiments", "exp9..13", etc.).
2. Cifras que contradigan las canónicas de arriba.
3. Afirmaciones de "pendiente / futuro / por hacer" que ya estén cumplidas.
4. Enlaces internos a archivos que ya no existan o renombrados.
5. Instrucciones de reproducción rotas o que mencionen rutas/ramas inexistentes.

## Formato de salida

- Veredicto general primero (¿la documentación está pareja? sí/no con matices).
- Tabla o lista por archivo con sus hallazgos.
- Lista final priorizada: qué corregir primero si solo hubiera tiempo para 3 cosas.
- NO propongas reestructurar el repo ni nuevos experimentos. Solo consistencia documental.
