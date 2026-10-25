# Guía de anotación de la referencia humana piloto v4 — paso a paso para Enzo

> **PROTOCOLO HISTÓRICO EJECUTADO (2026-08-29).** Las tandas A, B y el test–retest ya
> se completaron. Este archivo conserva exactamente las reglas que recibió el anotador;
> no es una convocatoria a reabrir ni modificar la referencia adjudicada.

> **Sección de [Kimi Work] — 2026-08-06 23:59 (hora local)**
> Guía práctica para anotar `claim_audit_sample_v4.csv` (Etapa A, 150 claims) y
> `claim_audit_sample_v4_stageB.csv` (Etapa B, 50 claims). Léela completa ANTES de la
> tanda A1 y tenla a mano en cada sesión.

---

## 0. Lo primero: no necesitas saber las respuestas

Esta es la duda más común y la más importante despejar:

**No estás respondiendo las preguntas. Estás juzgando si un claim queda respaldado por la
evidencia que tienes delante.** La pregunta es solo contexto; tu universo es el chunk.

- Si el chunk **dice** lo que el claim afirma → `correcto`, **aunque tú creas que en la
  realidad es falso**. Juzgas claim↔chunk, no claim↔mundo.
- Si el chunk **no dice** lo que el claim afirma → `incorrecto`, **aunque tú sepas que es
  verdad** (ej.: sabes de memoria que Lambda soporta Python, pero si el chunk no lo
  menciona, ese claim NO está respaldado *por esa evidencia*).
- Tu conocimiento de AWS/Azure/GCP sirve solo para ENTENDER el texto técnico, nunca como
  fuente de verdad. De hecho el diseño necesita exactamente esto: medir si el *verificador
  automático* coincide con un humano que juzga la misma evidencia.

Regla de bolsillo: **"¿Un lector cuidadoso, usando SOLO este texto, podría escribir este
claim?"** Sí → correcto. No → incorrecto o dudoso.

## 1. Las tres etiquetas, con la frontera exacta

| Etiqueta | Definición operativa |
|---|---|
| `correcto` | **Todo** lo material del claim está dicho o se sigue directamente del chunk. Vale paráfrasis. Números, entidades, servicios y alcance deben coincidir. |
| `incorrecto` | Puedes AFIRMAR que el chunk no respalda el claim: (a) lo contradice (dice lo contrario, otros números, otro proveedor/servicio), o (b) trata de otra cosa y el respaldo es imposible desde ese texto, o (c) el claim añade hechos que el chunk no menciona en absoluto. |
| `dudoso` | El chunk es **del tema correcto** pero queda genuinamente a medias: está truncado justo donde estaría la respuesta, es ambiguo, o respalda una parte del claim y no puedes verificar el resto. |

**La frontera incorrecto/dudoso** (la más difícil):
- Chunk **off-topic** (claim sobre auto-scaling de Azure, chunk sobre IAM de AWS) →
  `incorrecto`. No hay ambigüedad: ese texto no puede respaldar ese claim.
- Chunk **on-topic pero insuficiente** (claim con 3 afirmaciones; el chunk cubre 1 y se
  corta) → `dudoso`.
- Chunk on-topic que **cubre el tema y silencia el punto del claim** (el claim da un dato
  concreto — un número, un límite, un default — y el chunk describe el mismo tema sin
  mencionarlo) → aquí el diseño del estudio manda: `incorrecto` (no respaldado). Reserva
  `dudoso` para cuando NI SIQUIERA puedes decidir si el silencio es por truncamiento o
  ambigüedad real.

**Claims multi-parte:** todas las partes materiales deben estar respaldadas. Una parte
contradicha o inventada → `incorrecto`. Una parte imposible de verificar por truncamiento
→ `dudoso`.

**Claims degenerados:** el extractor a veces produce claims que son basura de formato
(títulos, restos de markdown, frases vacías de contenido factual, ej. *"Getting Started
with X ()."*). Si no hay proposición factual que juzgar → `incorrecto` con comentario
`claim degenerado / sin proposición factual`. Es un juicio legítimo y útil: alimenta la
taxonomía.

## 2. Procedimiento por claim (checklist de 6 pasos, ~1,5-2 min)

1. **Lee la pregunta.** Solo para saber de qué se esperaba hablar. 10 segundos.
2. **Lee el claim y descompónlo mentalmente:** ¿qué entidades nombra? (servicio,
   proveedor), ¿qué predica de ellas? ¿hay números, límites, defaults, negaciones?
3. **Lee el chunk buscando SOLO esos predicados.** Ignora el resto del texto. El chunk
   puede ser 90 % irrelevante y aun así respaldar el claim en una línea.
4. **Aplica las trampas del §3** antes de decidir (provider swap, números, negación).
5. **Decide y escribe** `correcto` / `incorrecto` / `dudoso` en minúsculas, exactas.
6. **Comentario solo cuando aporta:** obligatorio en `dudoso` (una frase: qué faltó),
   recomendado en casos raros (claim degenerado, chunk corrupto). Vacío en los claros.

## 3. Trampas frecuentes (los estratos ocultos están diseñados para esto)

Ejemplos **sintéticos** (no son del paquete; no busques coincidencias):

1. **Provider swap.** Claim: *"Azure VM Scale Sets escalan por reglas de CPU."* Chunk:
   documentación de **AWS** Auto Scaling con reglas de CPU. → `incorrecto`. Contenido
   equivalente NO es respaldo: el proveedor es parte del claim.
2. **Número cambiado.** Claim: *"se retienen las últimas 5 versiones"*. Chunk: *"the last
   3 versions are retained"* → `incorrecto`.
3. **Negación.** Claim: *"el servicio soporta claves gestionadas por el cliente"*. Chunk:
   *"does not support customer-managed keys"* → `incorrecto`. Lee los `not`, `only`,
   `except`.
4. **Generalización.** Chunk: *"Lambda soporta Python y Node.js"*. Claim: *"Lambda
   soporta todos los lenguajes principales"* → `incorrecto` (el chunk no respalda la
   generalización).
5. **Paráfrasis legítima.** Chunk: *"there is no additional charge for IAM"*. Claim:
   *"IAM es gratuito"* → `correcto`. No exijas las mismas palabras.
6. **Claim multi-parte.** *"EKS requiere actualizar primero el plano de control y los
   nodos se actualizan solos"*: si el chunk respalda la primera mitad y dice lo contrario
   de la segunda → `incorrecto`; si la segunda no aparece y el chunk está cortado →
   `dudoso`.
7. **Markdown corrupto / tabla rota:** el corpus tiene restos de conversión
   (`[AWS > ECS > ...]`, `\.`). Léelos con paciencia; si el daño hace ilegible justo la
   parte decisiva → `dudoso` con comentario.
8. **Confianza por familiaridad:** si lees un claim y piensas "esto es obviamente cierto"
   → SEÑAL DE ALARMA: vuelve al chunk y busca el respaldo literal. La mitad del experimento
   mide exactamente ese sesgo.

## 4. Protocolo de calidad (para que el gold sea defendible)

1. **Calibración (tanda 0).** Anota los idx 1-10. Al día siguiente, sin mirar lo que
   pusiste, re-anota esos mismos 10 en una hoja aparte y compara. Si cambias 3 o más,
   relee el §1-§3, ajusta tu criterio escrito (ver punto 2) y repite con otros 10 antes de
   seguir. Estos re-juicios NO van al CSV: son tu calibración personal.
2. **Diario de reglas.** Crea un archivo fuera del repo (o en tu libreta) donde escribes
   tus decisiones de criterio cuando aparezca un caso tipo nuevo ("los claims que citan
   precios sin fecha los juzgo así…"). Es tu fuente de consistencia entre tandas y, si un
   día hay segundo anotador, es la base para alinear criterios.
3. **Sesiones.** Bloques de 30-40 min con descanso real; máximo 90 min por sesión. La
   tanda típica: 30 claims de etapa A ≈ 45-60 min. Si notas que apruebas todo en
   automático, para: ese es el momento en que la calidad cae.
4. **Re-anotación de control (obligatoria).** Al terminar los 150, elige 20 idx al azar
   (puedes pedirme una lista seed 42 sin criterio tuyo), re-anótalos a ciegas en hoja
   aparte y calcula tu auto-acuerdo. Meta: **≥ 85 %**. Si quedas bajo, revisa los
   discordantes, decide cuál juicio defendes y considera rehacer la tanda más afectada.
   Reporta el % final en el comentario de entrega.
5. **Orden estricto A → B.** Termina y ENTREGA los 150 de la etapa A antes de abrir el
   archivo de la etapa B. Al anotar B, NO consultes lo que pusiste en A: el diseño mide
   cuántos juicios cambian al ver los 5 chunks, y eso solo funciona si B es independiente.
6. **Qué no abrir durante toda la anotación:**
   - `claim_audit_sample_v4_meta.json` — contiene estratos y etiquetas de los
     verificadores → te ancla.
   - Cualquier `arm_stats__*.md`, `hhem_vs_nli.md`, `disagreement_summary.md` — idem.
   - No me preguntes a mí ni a otra IA "¿este claim es correcto?" mientras anotas: eso
     convierte la referencia humana en una referencia asistida. Si te atoras, marca `dudoso` con
     comentario y sigue.
7. **Adjudicación post-hoc (excepción controlada).** Si al terminar te quedan `dudoso`
   que quieres resolver con consulta externa, hazlo DESPUÉS de entregar todo, y márcalos
   en el comentario (`adjudicado con consulta`). Así el análisis puede excluirlos en una
   prueba de sensibilidad. Úsalo como excepción, no como método.

## 5. Cómo llenar los archivos sin romperlos

- El CSV usa **punto y coma** como delimitador y codificación UTF-8 con BOM. **No lo
  abras ni guardes con Excel** (Excel cambia delimitador/codificación y puede destruir la
  columna de texto del chunk). Opciones seguras:
  - **Recomendada:** anota en un archivo de tandas nuevo, p. ej.
    `output/audit/gold_v4_tandas_enzo.md`, con una línea por claim:
    `A-001 | correcto` / `A-023 | dudoso | falta la mitad del claim, chunk cortado`.
    Al terminar todo, un script de 10 líneas hace el merge a los CSV y valida 150+50
    (ese script se lo encargas a un agente de código o me lo pides a mí; NO edites el
    CSV a mano si no hace falta).
  - Alternativa: editar el CSV directamente en VS Code como texto plano, escribiendo
    solo en la columna `juicio_humano` (y `comentario` si aplica), guardando con la
    misma codificación.
- Valores válidos, exactos, en minúsculas: `correcto`, `incorrecto`, `dudoso`. Cualquier
  otra variante (`Correcto`, `dudosa`, `ok`) rompe el análisis.
- Haz commits o copias de respaldo de tu archivo de tandas al cerrar cada tanda. Son
  horas de trabajo humano: lo más caro del proyecto ahora mismo.

## 6. Etapa B: qué cambia

- 50 claims (submuestreo de los 150), cada uno con **5 chunks** (E1..E5, ~1500 chars
  cada uno, en el orden en que el modelo los vio). Tanda de 25 ≈ 1,5-2 h.
- El criterio cambia en un punto: `correcto` = respaldado por **ALGUNO** de los 5 chunks.
  `incorrecto` = contradicho, o no respaldado por **ninguno**.
- Estrategia de lectura: lee E1→E5 buscando respaldo; si encuentras respaldo completo en
  E2, puedes parar y marcar `correcto`. Para `incorrecto` o `dudoso` sí debes leer los 5.
- No compares con tu juicio de la etapa A. Si recuerdas el claim, ignora el recuerdo.

## 7. Qué pasa cuando terminas

1. Me avisas (o al agente de turno) y se corre el merge + validación (150/150 y 50/50,
   solo valores válidos, idx completos).
2. Se ejecuta `scripts/analyze_gold_v4.py` real (ya probado end-to-end en modo
   simulación): reporta κ ponderada por verificador candidato (small, base, hhem,
   E5_base+hhem, E1_mean), κ sin pesos sobre el estrato ancla, curva de calibración/ECE,
   y la **corrección etapa B** (cuántos juicios cambian con el contexto completo).
3. Con eso se responden las dos preguntas que bloquean el paper: el nivel real de
   fidelidad (¿más cerca de 0,30 o de 0,55?) y el verificador definitivo.
4. Tu auto-acuerdo del §4.4 se reporta junto al análisis como la confiabilidad del gold.

## 8. Plan de tandas (referencia rápida)

| Tanda | Contenido | Estimado | Hecha |
|---|---|---|---|
| 0 | Calibración idx 1-10 (no cuenta para el CSV) | 30 min + revisión al día siguiente | ☐ |
| A1 | Etapa A idx 1-30 | ~60 min | ☐ |
| A2 | Etapa A idx 31-60 | ~60 min | ☐ |
| A3 | Etapa A idx 61-90 | ~60 min | ☐ |
| A4 | Etapa A idx 91-120 | ~60 min | ☐ |
| A5 | Etapa A idx 121-150 | ~60 min | ☐ |
| C | Re-anotación de control (20 aleatorios) | 45 min | ☐ |
| B1 | Etapa B idx 1-25 | ~100 min | ☐ |
| B2 | Etapa B idx 26-50 | ~100 min | ☐ |

*Fin de la sección de [Kimi Work]. Si algo de esta guía se contradice con lo que
encuentres en los archivos, avísame: la guía se corrige, no el criterio a mitad de
anotación.*
