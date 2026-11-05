# Registro de la defensa LACCI: problemas, estado y solución (2026-11-05)

Inventario de los puntos débiles de las preguntas probables de la defensa, cruzado con el repositorio. Estados:

- **[VERIFICADO]:** la fuente está en el repo y se comprobó.
- **[REGISTRADO]:** ya constaba en un ledger o ADR.
- **[NUEVO]:** se registra aquí y en `paper/correction_log.md` N10.
- **[SIN FUENTE]:** la cifra se usa en el material oral pero no está en el repo. No citar hasta versionarla.

## Defectos del camera-ready

| # | Problema | Estado | Evidencia | Solución |
|---|---|---|---|---|
| 1 | BM25 `k_1=1.5` en el paper; se usó 1,2 | **[NUEVO] [VERIFICADO]** N10-1 | `...v9.tex:67` frente a `config/config.yaml:100` y `bm25_index.py:50` | Declarar la errata; ninguna cifra cambia. Decidir si se pide errata a IEEE. Sin análisis de sensibilidad de k1. |
| 2 | El ledger decía «No hay erratas que corregir» | **[NUEVO]** N10-1 | `summer_ablation_log.md:2392` | Corregido por N10-1, sin editar el ledger (append-only). |
| 3 | El abstract («does not translate») es más fuerte que la evidencia | **[NUEVO]** N10-2 | `...v9.tex:33`; κ 0,30 | En la defensa: «con este verificador». El cambio de prosa requiere OK frase por frase. |
| 4 | Ejemplos «common across models» que vienen de Llama 3.1 | **[NUEVO] [VERIFICADO]** N10-3 | `exp9_llm_only_no_rag/_handoff/qualitative_examples.md`; 0 coincidencias en `exp12_matrix/*sin_rag*` | Decir que son ilustrativos de una corrida anterior. No se midió la tasa de error factual de los cuatro modelos sin RAG. |
| 5 | Oráculo `bge-reranker-large` de la misma familia que el denso (BAAI) | **[NUEVO]** N10-4 | `audit_findings.md:99` lo llama «otra familia» | Admitir el posible sesgo a favor del denso. Las etiquetas humanas lo resolverían. |
| 6 | No se probó denso + reordenador | **[NUEVO]** N10-4 | Tabla II: solo RRF y RRF + reordenador | Limitar la conclusión a «en nuestro diseño». |

## Limitaciones ya registradas

| Problema | Estado | Dónde |
|---|---|---|
| Oráculo circular (0,995) | [REGISTRADO] | `audit_findings.md` §oráculo |
| Fidelidad 0 sin RAG por construcción | [REGISTRADO] N3 | `results-summary.md:83` |
| Qwen sin RAG: 177/194 vacías | [REGISTRADO] N6 | `audit_findings_cc_addenda.md:238`. Causa (modo de razonamiento): no verificada |
| «RAG-tuned» frente a «capacidad para RAG» de Granite | [REGISTRADO] | `adr-log.md:46` |
| κ 0,30-0,36 del verificador, un anotador | [REGISTRADO] | `summer-benchmarks.md:88` |
| Sensibilidad al umbral | [REGISTRADO, parcial] | `summer-benchmarks.md:360`: con τ = 0,5 (fase de verano, no el 0,7 del paper), 123 de 759 afirmaciones sin respaldo (16 %) quedan a menos de 0,1 del umbral. El 0,7 del paper no se evaluó. |

## Cifras del material oral sin fuente en el repo [SIN FUENTE]

No aparecen en ninguna rama remota (búsqueda del 2026-11-05). Pueden estar en `C:\CloudRAG`, que no es accesible desde aquí.

| Cifra | Usada en | Solución |
|---|---|---|
| Granite híbrido: 102 de 107 «declinaciones» con afirmaciones | Pregunta «Is decline really declining?» | Versionar el script y la salida (`_vN`) o no citarla |
| Fidelidad de 0,299 a 0,207 incluyendo las declinaciones | Ídem | Ídem. El 0,299 sí está en `...v9.tex:95` y en `ESTADO_PROYECTO_UNIFICADO_2026-08-06.md:61` |
| 44 de 169 preguntas de un solo proveedor sin documentación propia | Pregunta «Why does Granite decline so much?» | Ídem (`COBERTURA_CORPUS` no está en el repo) |
| Panel de expertos para las 194 preguntas | Pregunta «How were the 194 queries built?» | Registrar el plan (`MATERIAL_ORAL` no está en el repo) o decir solo «planned» |

## Respuestas cortas que conviene ajustar

- **k1:** la respuesta está bien. Añadir solo si preguntan: «it's logged as an erratum in our correction log».
- **Oráculo:** no decir «a different model family». Decir «a different model from the pipeline's reranker; same organization as the dense encoder».
- **Ejemplos sin RAG:** la versión corta ya dice «earlier Llama 3.1 run». Ahora está verificado (N10-3).
