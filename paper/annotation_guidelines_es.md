# Guía de Anotación — Relevancia Query ↔ Chunk

**Proyecto:** Hybrid RAG multi-cloud (tesis Enzo Ordoñez, LACCI 2026)
**Duración estimada:** 8-10 horas distribuidas en 2 semanas
**Formato:** Google Sheets (link aparte)

---

## Qué tenés que hacer

Para cada fila del Sheet, leés:

- Una **pregunta** técnica (en inglés) sobre servicios cloud.
- Un **fragmento** (chunk) de documentación oficial (en inglés).

Tu trabajo: decidir si ese fragmento **ayuda a responder** la pregunta.

Marcás en la columna `relevance` un número:

- **0 = NO SIRVE.** El fragmento no ayuda a responder la pregunta.
  - Ejemplo: pregunta sobre *"EC2 auto-scaling"*, fragmento sobre *"Lambda pricing"* → **0**.

- **1 = SIRVE PARCIALMENTE.** Menciona el tema pero no lo resuelve, o contiene información tangencial útil.
  - Ejemplo: pregunta sobre *"EC2 auto-scaling"*, fragmento describe qué es auto-scaling sin explicar cómo configurarlo → **1**.

- **2 = SIRVE DIRECTAMENTE.** Responde la pregunta o contiene información esencial para responderla.
  - Ejemplo: pregunta sobre *"EC2 auto-scaling"*, fragmento muestra configuración paso a paso con ejemplos → **2**.

---

## Reglas estrictas

1. **Juzgá cada fragmento por separado.** No mires otros fragmentos de la misma pregunta al decidir.

2. **Ignorá la calidad de la redacción.** Juzgá solo la relevancia.

3. **Respetá el proveedor de la pregunta.** Si la pregunta es sobre AWS pero el fragmento es de Azure con información equivalente: eso NO es relevante (marcá 0).

4. **En dudas, sé conservador:**
   - Entre 1 y 2 → elegí 1.
   - Entre 0 y 1 → elegí 0.

5. **Usá la columna `notes` solo para casos ambiguos** o preguntas mal formuladas. No comentes fragmentos que son claramente 0 o 2.

6. **No mires qué marcan los otros anotadores.** No discutan casos durante la anotación. Las discusiones son al final, en la reunión de adjudicación.

7. **Trabajá en sesiones cortas:** 1 a 2 horas máximo. Descansá entre sesiones. Después de 30-40 minutos seguidos la calidad cae.

---

## Cuándo preguntarle a Enzo

- La pregunta no se entiende o está mal formulada.
- El fragmento está cortado de forma que no se puede evaluar.
- Dudas metodológicas (NO dudas de *"¿esto es 0 o 1?"* — eso lo decidís vos aplicando la guía).

Enzo **no** te va a decir *"esto es 1 o 2"*. Te va a responder *"aplicá la guía"*. Eso preserva tu independencia de juicio.

---

## Entrega

Cuando termines, avisale a Enzo. Él cierra el Sheet y procesa los datos. Después habrá una reunión corta (30 min) para discutir los casos donde los 3 anotadores tuvieron desacuerdo fuerte.

---

*Versión 1.0 — 2026-04-24*
