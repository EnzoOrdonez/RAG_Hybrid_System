# exp16 — decodificación anclada: NO mejora la fidelidad (resultado negativo triangulado)

**Fecha:** 2026-07-24 · **Fase:** verano, Fase 2 (mejoras) · **Estado:** para revisión de Enzo
**Regla:** report-before-prose. NO cambia cifras firmadas (exp16 nuevo). Toca la matriz de factibilidad
(línea 1a del A.3) → material de discusión, no de prosa sin OK.
**Evidencia:** `experiments/results/exp16_anchored_decoding/{results.json, faithfulness_rows__{small,base}__vb_agree.json, faithfulness_rows__hhem.json, arm_stats__{small,base,hhem}.{json,md}, guards.{json,md}}`

## Diseño
Tier A mostró que las perillas de recuperación no mueven la fidelidad → la palanca, si existe, es de
generación (A.3 línea 1a, decodificación anclada). exp16 la prueba: 3 brazos sobre el MISMO pool híbrido
(ids firmados exp11, identidad de contexto), granite temp0 seed42; **solo cambia el system+sufijo del
prompt**. Contraste pareado within-session vs `baseline_repro` (Wilcoxon+d_z+bootstrap seed42, familia BH
de 2). `baseline_repro` regenerado **fresco --no-cache** (co-temporal con los brazos; el caché habría
servido el baseline de Tier A, ~7h antes → deriva H5 cross-sesión; verificado: fresh q016 2247 vs cached
2704). Métrica vb_agree τ0.7 (NLI) / max_chunk τ0.5 (HHEM); `--strip-inline-cites` quita los `[N]` antes de
extraer claims, uniforme en los 3 brazos (el path firmado `_extract_claims` no se toca).

| Brazo | Intervención de prompt |
|---|---|
| `baseline_repro` | prompt canónico (ancla) |
| `anchored_cite` | citar el nº de chunk `[N]` tras cada oración factual; prohíbe afirmar lo no listado |
| `strict_abstain` | afirmar solo lo explícito; omitir lo incierto; preferir respuesta corta anclada |

Smoke: anchored_cite emite ~14 citas `[N]` (0 `[Source:]`) tras corregir el prompt (1er intento granite
las ignoraba); strict_abstain acorta 21–56%. → las intervenciones SÍ cambian el comportamiento.

## Resultado — 0/2 bajo NLI-small, NLI-base Y HHEM (ninguna mejora, robusto al instrumento)

| Brazo | small Δ (p_BH) | base Δ (p_BH) | HHEM Δ (p_BH) | veredicto |
|---|---|---|---|---|
| anchored_cite | −0,034 (0,59) | −0,040 (0,53) | −0,056 (0,50) | 0/3; **tiende ABAJO en los 3** |
| strict_abstain | −0,002 (0,59) | +0,031 (0,53) | +0,038 (0,54) | 0/3; plano |

Nivel baseline: 0,296 (small) / 0,226 (base) / 0,498 (HHEM). HHEM baseline 0,498 ≈ granite HHEM 0,40–0,50
→ carga verificada. Todos los d_z despreciables (|d_z| ≤ 0,19).

## Guardas anti-gaming — el mecanismo del fallo
| Brazo | declinación | palabras | claims genuinos | solape verbatim 5-gram |
|---|---|---|---|---|
| baseline_repro | 51,7 % (31/60) | 352 | 11,95 | 0,122 |
| anchored_cite | 58,3 % (35/60) | 223 | 7,55 | 0,061 |
| strict_abstain | 60,0 % (36/60) | 191 | 5,27 | 0,233 |

- **Ambas intervenciones suben la declinación** (51,7 %→58,3 %/60 %) y **recortan el contenido** (palabras
  352→223/191; claims 11,95→7,55/5,27). Hacen que granite **diga menos y se abstenga más**, sin ganar
  fidelidad.
- **anchored_cite:** menos claims, MENOS solape (0,061 → no copia), y aun así fidelidad ABAJO. Forzar la
  cita hizo que el modelo afirme hechos "citados pero no soportados" → **cita ≠ grounding** (teatro de
  citación, no anclaje real). El solape bajo descarta que el efecto sea copia.
- **strict_abstain:** el mayor solape (0,233 → cita más verbatim cuando responde) y aún así fidelidad plana
  (HHEM +0,038, n.s.). Copiar + abstenerse no compró fidelidad.

## Veredicto
**La decodificación anclada por prompt NO mejora la fidelidad de granite** (n=60). Ambos brazos suben la
abstención y recortan contenido; anchored_cite incluso **baja** el grounding (cita sin entailment).
Triangulado en 3 instrumentos. Junto con Tier A (nulo del lado de recuperación), **ni el arreglo del
contexto ni el anclaje por prompt mueven la fidelidad** → el techo es más profundo (capacidad del modelo
y/o el efecto real es solo de selección de contenido, per Tier 3).

**Caveat de potencia (real):** declinación baseline **51,7 %** → n efectivo ≈ 29; underpowered, un efecto
positivo pequeño podría escaparse. Pero la DIRECCIÓN (anchored abajo, strict plano) y las guardas (más
abstención, menos contenido) argumentan en contra de un positivo oculto. Confirmatorio a 194q solo con OK.

## Implicación para A.3/LACCI (report-before-prose)
- **Matriz de factibilidad, línea 1a (decodificación anclada):** de "IMPLEMENTAR" pasa a **IMPLEMENTADA Y
  PROBADA — sin ganancia local**. Resultado negativo honesto: descarta una línea de Trabajo Futuro como
  victoria local. Contribución real (no todo lo que se prueba funciona).
- Refuerza el hallazgo central: la baja fidelidad no se arregla ni moviendo el contexto ni el prompt →
  apunta a capacidad del modelo (línea 1b, fuera de 6 GB) o al instrumento (Tier 3, gold pendiente).
- NO tocar prosa del A.3 sin OK frase por frase.

## Pendiente
- Gold humano arbitra el nivel absoluto y valida si el "0/2" es techo real o de instrumento.
- (Opcional) confirmatorio 194q si Enzo lo pide; el diseño y la infra están listos.
