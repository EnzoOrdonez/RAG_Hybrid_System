# Paquete Interno de Anotación — Uso de Enzo

**Fecha:** 2026-04-24
**Proyecto:** LACCI 2026 — Hybrid RAG multi-cloud revalidation

Este documento es solo para Enzo. No se comparte con anotadores.

---

## 1. Mensaje de reclutamiento (copia/pega)

Ajustá el saludo según destinatario (asesor / compañero).

```
Hola [Asesor/Nombre],

Te escribo para pedirte una ayuda concreta con mi tesis de RAG
híbrido multi-cloud que voy a enviar a LACCI 2026 (deadline
26 de mayo).

Para que las métricas de evaluación sean científicamente válidas,
necesito que 2 personas aparte de mí califiquen la relevancia
de resultados de búsqueda. Es una tarea de lectura y puntuación
de 0 a 2 sobre fragmentos técnicos de AWS/Azure/GCP/Kubernetes/
CNCF. Son 514 filas (25 preguntas × ~20 fragmentos cada una).

Tiempo estimado: entre 6 y 10 horas totales, distribuidas en
sesiones de 1-2 horas durante 2 semanas (entre el 28 de abril
y el 4 de mayo).

Herramienta: Google Sheets. No tenés que programar nada.

Reconocimiento formal: mención explícita en la sección de
agradecimientos del paper LACCI. Si tu contribución termina
siendo mayor (discusión metodológica, revisión), subimos a
co-autoría.

¿Podés ayudarme?

Si aceptás, te mando el link del Sheet + una guía de 1 página
y hacemos una llamada de 30 minutos para que hagamos 2-3
ejemplos juntos antes de arrancar.

Gracias,
Enzo
```

---

## 2. Setup Google Sheets

### Subida inicial de CSVs

CSVs ubicados en: `data/evaluation/annotation_{enzo,advisor,classmate}.csv`

Para cada CSV:

1. drive.google.com → New → Google Sheets → Blank.
2. Nombrá: `LACCI2026_Annotation_Enzo` (y `_Advisor`, `_Classmate`).
3. File → Import → Upload → seleccioná el CSV correspondiente.
4. Import location: **Replace current sheet**.
5. Separator type: **Comma**.
6. Import data.

Resultado: 3 Sheets separados, uno por anotador.

### Configuración de cada Sheet (idéntica para los 3)

- **Fila 1 (headers):** Format → Freeze → 1 row.
- **Columna `chunk_text_preview`:** Format → Text wrapping → Wrap. Width: ~600px.
- **Columna `relevance`:** Data → Data validation.
  - Criteria: *List of items* → `0,1,2`.
  - Reject input si está fuera de rango.
  - Apply to range: toda la columna excepto header.
- **Columna `notes`:** texto libre, ancho ~300px.
- **Protección:** Data → Protect sheets and ranges.
  - Proteger columnas `query_id`, `query_text`, `chunk_id`, `chunk_text_preview`.
  - Solo Enzo puede editar; anotadores solo pueden editar `relevance` y `notes`.

### Compartir

1. Share → Add people → mail del anotador correspondiente.
2. Permiso: **Editor**.
3. Notify: sí, con mensaje corto:
   > *"Este es el sheet de anotación. Ya te pasé las guidelines por aparte. Cuando terminés, avisame."*

### Sheet de control (privado)

Creá un 4º Sheet: `MASTER_tracking`.

Columnas sugeridas:
- Fecha
- Filas completadas por Enzo
- Filas completadas por Asesor
- Filas completadas por Compañero
- Notas / bloqueos

Incluí checkboxes:
- [ ] Pilot hecho
- [ ] 25% anotado
- [ ] 50% anotado
- [ ] 75% anotado
- [ ] 100% anotado

---

## 3. Cronograma de ejecución

### Viernes 25 / Sábado 26 abril — Reclutamiento
- Mandar los 2 mensajes de reclutamiento (asesor + compañero).
- Esperar respuestas.

### Domingo 27 abril — Setup
- Crear los 3 Google Sheets y configurar según sección 2.
- Mandar links con guidelines (`annotation_guidelines_es.md`).
- Agendar reunión de 30 min con ambos para lunes o martes.

### Lunes 28 / Martes 29 abril — Pilot
- Reunión 30 min (grupal o 1-a-1).
- En la llamada: cada uno abre su sheet. Juntos califican las primeras 3-5 filas como ejemplos. Discuten dudas. Alinean criterio.
- Después de la reunión: cada uno arranca en paralelo, aislado.

### Martes 29 abril – Lunes 4 mayo — Anotación
- Asesor y compañero: ~1.5 horas por sesión × 5-6 sesiones = 8-10 horas totales.
- Enzo: ~2 horas por sesión × 7-8 sesiones = 14-16 horas totales (carga mayor porque tiene las 50 completas).
- Checkpoint diario: Enzo mira progreso en `MASTER_tracking`. Si alguien va atrasado, renegociar deadline con él.

### Martes 5 / Miércoles 6 mayo — Adjudicación
- Bajar los 3 Sheets a CSV (File → Download → CSV).
- Correr cálculo de Cohen's κ y Fleiss' κ (script pendiente, se arma cuando llegue el momento).
- Reunión de 1 hora con los 3 para discutir desacuerdos fuertes (diferencia ≥ 2 en escala 0/1/2).
- Generar `data/evaluation/ground_truth.json` final con judgments poblados.

### Jueves 7 mayo en adelante — Evaluación
- Correr evaluación real con oracle BGE + ground truth gold.
- Análisis estadístico (Cohen's d_z, Wilcoxon, BH-FDR).
- Escritura del paper.

### Deadline LACCI: 26 de mayo 2026

---

## 4. Principios operativos durante la anotación

- Enzo **no** interviene en juicios individuales. Si anotador pregunta *"¿esto es 1 o 2?"*, la respuesta es *"aplicá la guía"*.
- Enzo **sí** responde dudas metodológicas, preguntas mal formuladas, o fragmentos cortados.
- Los 3 anotadores trabajan aislados. No comparten opiniones durante el proceso.
- Las discusiones de desacuerdos son **post-anotación**, en la reunión de adjudicación.

---

## 5. Métricas de calidad esperadas

- **Fleiss' κ (3-way):** objetivo ≥ 0.6 (substantial agreement). Si cae entre 0.4 y 0.6 (moderate), reportar en limitaciones. Por debajo de 0.4, revisar guidelines y re-entrenar.
- **Cohen's κ (pairwise):** Enzo-Asesor, Enzo-Compañero, Asesor-Compañero. Consistencia esperada.
- **Casos de disagreement fuerte (|max - min| = 2):** esperado ~10-15 casos en 25 queries compartidas. Estos van a adjudicación manual.

---

*Versión 1.0 — 2026-04-24*
