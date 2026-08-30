# Segunda opinión — claim "entre-modelos" (fidelidad v4)

**Fecha:** 2026-07-03 · **Alcance:** verificación de solo lectura (nada modificado; este .md queda
untracked, sin commit). **Verificador:** Claude Code, sesión independiente de Cowork.
**Entorno:** intérprete `pythoncore-3.14-64`, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONHASHSEED=42` (solo re-análisis estadístico; sin generación LLM, sin modelos cargados).

**Documentos evaluados (además del repo):**
`D:\...\Nota_3\A3_InformeTecnico_Nota3_V8.docx` (2026-07-03 20:26) y
`D:\...\Nota_3\Paper_IEEE_RAG_Hibrido_Nota3_LACCI.tex` (2026-07-03 20:49) — leídos, no tocados.

---

## Punto 1 — Canonicidad de los archivos: **SÍ**

Los dos JSON son los artefactos v4 firmados del cierre N9, sin versiones posteriores.

| Evidencia | Comando | Salida |
|---|---|---|
| Commit que los introduce | `git log --follow -- faithfulness_metrics_v4.json` (ídem `_small`) | único commit v4: **`e671f67`** "data(nota3): faithfulness v4 - exclude vacuous all-artifact rows (N9)" |
| Sin modificaciones posteriores | `git log e671f67..HEAD -- <ambos>` | vacío |
| Contenidos == tag de cierre | `git diff --quiet nota3-N9-cierre-2026-07-02 -- <ambos>` | **IDENTICAL to tag** |
| Tags que los contienen | `git tag --contains e671f67` | `nota3-N9-cierre-2026-07-02` y `nota3-N9-v4-2026-07-02` |
| Metadata interna (ambos) | lectura JSON | `metric: "faithfulness_answered (v4_small, ledger N9)"` / `(v4, ledger N9)`, `generated: 2026-07-02`, `vacuous_exclusion: true`, `excluded_methods: [error, none, vacuous]`, `faithfulness_source: faithfulness_rescore_v3__{small,base}__vb_agree.json` |
| Ledger | `paper/audit_findings_cc_addenda.md:586-598` | entrada "Cierre N9", Decisión 1 cita exactamente estos artefactos y pares |
| Resumen citable | `RESULTADOS_RESUMEN.md:122-133` | bloque "[Corregido en N9 — framing B…]" con las mismas cifras |

## Punto 2 — Familia BH, regla de pareo y reproducción: **SÍ**

**Familia = 18**: 3 escenarios RAG (léxico, denso, híbrido) × C(4,2)=6 pares de modelos.
`sin_rag` excluido explícitamente de la familia (`compute_faithfulness_metrics.py:452-453`,
"faithfulness 0-by-construction"); `bh_families.between_model_per_scenario = 18` en ambos JSON y
`correction_family_size: 18` en cada par.

**Regla de pareo** (código `aligned_vectors`, `compute_faithfulness_metrics.py:232-251`, y
declarada en el campo `pairing_rule` del propio artefacto): intersección de query_ids con
(a) method ∉ {error, none, **vacuous**} en **cualquiera** de los dos brazos (vacua = fila del
rescore con `genuine==0`, ruta N9), (b) faithfulness no nula en ambos, (c) `class_v2 ≠
pure_decline` en **ambos** brazos (denominador primario `faithfulness_answered`). Wilcoxon
signed-rank two-sided (gate Shapiro para t pareada), d_z = media(b−a)/sd(b−a, ddof=1), BH
(`fdr_bh`) + Holm de statsmodels sobre la familia completa, α=0,05.

**Reproducción independiente** (`scratchpad/recount_v4_between_model.py`: implementación fresh —
no importa el pipeline; el clasificador de declinación se reconstruyó desde la metadata
`decline_classifier_v2` que el propio artefacto firmado declara; parte de
`results.json` + los rescores versionados):

```
small: familia recontada = 18 | firmada = 18 — pares OK: 18/18
   sig_bh: 1/18 -> denso | granite4.1-8b vs mistral-7b   p_bh=0.0136  d_z=+0.4151 (small)  n=75
base:  familia recontada = 18 | firmada = 18 — pares OK: 18/18
   sig_bh: 1/18 -> lexico | gemma4-e4b vs granite4.1-8b  p_bh=0.0383  d_z=-0.5311 (medium) n=42
ROBUSTO (ambos verificadores): 0/18
mismatches vs JSON firmados: small=0 base=0
```

**36/36 pares coinciden exactos** con los JSON firmados en n, p_raw, p_bh, sig_bh, p_holm,
sig_holm, d_z, effect_label, test y medias (tolerancia 1e-9 en p). La tabla de Cowork es
**correcta**: small **1/18**, base **1/18**, robusto **0/18**, con exactamente esos pares y p_BH.

Detalle no reportado por Cowork (no cambia el veredicto): ambos pares aislados también son
`sig_holm=True` dentro de su verificador, y el par de base cumple el umbral de tesis |d_z|≥0,5
con `power_note` (n=42<60). La defensa del claim descansa en el cruce de verificadores, no en que
los pares aislados sean marginales — la redacción del A.3 ("no se replica en el otro") usa
exactamente ese argumento, correcto.

## Punto 3 — Framing B como estándar acordado: **SÍ**

- `paper/audit_findings_cc_addenda.md:588-595` (Cierre N9, Decisión 1 de Enzo): "**0/18 pares
  entre-modelos significativos bajo los dos verificadores** — el mismo estándar de doble
  verificador con el que se defiende el hallazgo central. […] **SUPERSEDE la línea '1/18 bajo el
  verificador primario'**". Refs TRUE (arXiv:2204.04991) y Verifying the Verifiers
  (arXiv:2506.13342).
- Mismo estándar que el hallazgo central: `addenda:426-427` — retrieval n.s. "Confirmado ahora
  bajo **los dos verificadores** (base 0/12 y small 0/12)"; el A.3 V8 §6.4 usa la misma
  construcción para RAG-vs-RAG ("esto se mantiene bajo dos verificadores de inferencia y cuatro
  definiciones de denominador").

## Verificación de los documentos reales (más allá de lo citado en el prompt)

- **A.3 V8 §6.4 (cuerpo):** la frase citada en el prompt existe **verbatim** en el párrafo de
  resultados de fidelidad. Cada componente se sostiene sobre los JSON firmados:
  "ninguna se sostiene de forma robusta bajo los dos" = 0/18 ✓; "cada verificador señala a lo
  sumo un par aislado" = 1/18 y 1/18, distintos ✓; "no se replica en el otro" (al nivel del
  estándar sig-BH: el par de small da p_bh=0,084 bajo base; el de base da p_bh=0,582 bajo
  small) ✓; "los niveles absolutos dependen del verificador" = Δ medias por celda 0,002–0,114,
  small>base en 11/12 celdas ✓.
- **LACCI tex:** la tabla de faithfulness = `v4_small.primary_answered` **exacta en 16/16 medias
  y 16/16 n** (incl. no-RAG 0.000 con n=189/191/193/17). La tabla de declinación (12/12 celdas)
  = `honest_decline_rate_v1` exacta. Abstract/discusión/conclusión no afirman significancia
  entre-modelos de fidelidad; afirman "the dominant difference between models is how often they
  decline" (ver soporte abajo). Dos verificadores (línea 43, 79), exclusión de filas vacuas en
  prosa (línea 79: "Responses whose extracted claims are entirely formatting artifacts […]
  excluded from the denominator") y la precisión obligatoria del cierre N9 aplicada (línea 72:
  "deterministic at temperature zero **in our measurement environment**"). ✓

**Soporte del claim de declinación de LACCI** ("dominant difference is decline"): McNemar v2
sobre los 18 pares → **13/18 con p<0,05**, y los 13 **sobreviven BH sobre la familia de 18**
(mayor p significativo 3,55e-3 < umbral BH rank-13 = 0,0361); los 9 pares que involucran a
mistral tienen p ≤ 3e-5. El ordenamiento (mistral declina mucho menos) se sostiene bajo los DOS
clasificadores de declinación (flag v1: 21–25 % vs 44–66 %; censo v2: 31–39 % vs 55–68 %),
mientras la fidelidad da 0/18 robusto. "Dominante" está soportado.

---

## VEREDICTO ÚNICO

**La frase del A.3 §6.4 (cuerpo) y el claim de LACCI, tal como están redactados, son FIELES a la
evidencia v4.** Los números de Cowork se reproducen exactos.

**PERO** hay 2 defectos puntuales que Cowork pasó por alto, ambos en el A.3 V8 (no en LACCI), y
1 matiz de defensa. **No apliqué ninguna corrección** — decisión frase-por-frase de Enzo.

### Hallazgo 1 (corrección necesaria) — nota de la Tabla 6 del A.3 V8

> "Nota. Fidelidad relativa al instrumento; las comparaciones entre modelos no son
> significativas tras la corrección."

Sin el calificador dual-verificador esto es **falso bajo el verificador de la propia tabla**
(v4_small): 1/18 par SÍ es significativo tras BH (y Holm). La nota comprime el framing B hasta
contradecir el JSON firmado si se lee sola. **Corrección mínima propuesta:**
"…; ninguna comparación entre modelos se sostiene bajo los dos verificadores de inferencia (cada
verificador aislado marca a lo sumo un par, que no se replica en el otro)."

### Hallazgo 2 (inconsistencia interna) — A.3 V8 §6.5, texto vs su propia Tabla 7

El texto dice "Granite y Qwen declinan **entre el 62 y el 66 %**", pero la Tabla 7 (que usa el
flag v1, verificado celda a celda) da **Qwen 43,8 / 47,9 / 46,4 %** — el rango 62–66 de Qwen
corresponde al **censo v2** (63,4–67,7 %), es decir, la frase mezcla clasificadores (el rango de
Granite sí es v1: 61,9–65,5). LACCI no tiene este problema (cita 61,9 % vs 24,2 %, consistente
con su tabla v1). **Corrección mínima propuesta (opción a):** "Granite declina entre el 62 y el
66 % y Qwen entre el 44 y el 48 %" (fiel a Tabla 7/v1); **(opción b):** referir la frase al
desglose v2 de la Figura 5 y ajustar ambos rangos (Granite 55–61 %, Qwen 63–68 %). Elegir una.

### Matiz 3 (preparación de defensa, no error)

El par aislado de small (denso granite-vs-mistral, d_z=+0,415) es **direccionalmente consistente
bajo base** (d_z=+0,257, p_raw=0,0205) aunque no sobreviva BH (p_bh=0,084). Un jurado podría
objetar "sí se replica en crudo". Respuesta defendible: el estándar pre-acordado (Decisión 1,
cierre N9) es significancia **corregida** bajo **ambos** verificadores — el mismo con el que se
defiende el hallazgo central — y ese par no lo pasa; el par de base (léxico gemma-vs-granite) sí
es genuinamente no-replicante (p_raw=0,22 bajo small). Existe además léxico gemma-vs-mistral con
p_raw<0,05 en ambos verificadores (0,024 / 0,007) sin sobrevivir BH en ninguno. Nada de esto
cambia el 0/18; conviene tenerlo listo para preguntas.

---

**Scripts de evidencia (scratchpad de la sesión, reproducibles):**
`dump_v4_between_model.py` (lectura directa 18+18 pares), `recount_v4_between_model.py`
(reproducción independiente, exit 0 = 36/36 OK), `proactive_checks_v4.py` (declinación, niveles
por verificador, consistencia direccional), `dump_n.py` (n de la tabla LACCI).
