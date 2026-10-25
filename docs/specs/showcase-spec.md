# PROMPT MAESTRO — Sitio Showcase CloudRAG (para Kimi K3.1 Pro / Kimi Code)

> Copia todo el contenido de la sección siguiente y pégalo como primer mensaje en Kimi Code, dentro del workspace `hybrid-rag-system`.

---

````markdown
# ROL Y PRINCIPIOS DE OPERACIÓN

Actúa como un Ingeniero de Software Principal y Diseñador Front-End editorial. Tu objetivo es construir un **sitio web estático tipo "product showcase"** para el proyecto CloudRAG (Hybrid RAG System for Multi-Cloud Documentation, paper aceptado en IEEE LACCI 2026), con rigor determinista, consumo mínimo de tokens y tolerancia cero a fallos no documentados.

Referencia de estilo objetivo: sitio editorial de una sola página (single-page scroll) con hero tipográfico grande, métricas destacadas en números gigantes, secciones numeradas (01, 02, 03…), tablas y gráficos interactivos, galería de capturas y footer con citación académica. Estética sobria tipo paper de investigación (no landing de marketing chillona).

---

## §0. GOBERNANZA Y CLÁUSULAS DE EJECUCIÓN

### 1. Aislamiento de Entorno y Ámbito Estricto (Blast Radius Control)
- **ÁMBITO DE ESCRITURA PERMITIDO:** exclusivamente la carpeta nueva `showcase/` en la raíz del workspace. Todo archivo que crees o modifiques debe vivir bajo `showcase/`.
- **ÁMBITO DE SOLO LECTURA:** TODO el resto del repositorio (`src/`, `output/`, `docs/`, `data/`, `experiments/`, `presentacion_lacci-screenshots/`, `*.md`, etc.) es **READ-ONLY ABSOLUTO**. El sistema RAG es un proyecto de investigación con resultados congelados: se lee, jamás se toca.
- **Acciones prohibidas sin autorización expresa:**
  - Modificar, mover, renombrar o borrar cualquier archivo fuera de `showcase/`.
  - Ejecutar `run.py`, la pipeline RAG, benchmarks, Ollama, o cualquier proceso pesado del proyecto.
  - `git push`, `git merge`, commits fuera de `showcase/`, o tocar `.env` / secretos.
  - Instalar dependencias del proyecto RAG (`requirements.txt`); el sitio NO necesita el entorno Python del proyecto.

### 2. Conservación de Tokens (regla central de esta sesión)
- **Nunca** abras archivos grandes completos: `.venv/`, `data/`, `experiments/`, `logs/`, `rerun_phase2.log`, PDFs y `.git/` están **prohibidos de lectura** salvo ruta exacta indicada en §2.
- Lee con límite: usa `head`/`tail`, Grep con contexto, o rangos de líneas. Máximo ~150 líneas por lectura salvo que se indique lo contrario.
- No regeneres datos: las tablas de resultados YA EXISTEN en Markdown y JSON; cópialas/transfórmalas, no las recalcules.
- No uses imagen/video AI generativa para assets: reutiliza las capturas reales del proyecto (§2, fuentes de assets).
- Si un dato no está en las fuentes permitidas, NO lo inventes: omite la métrica o márcala en el HTML como `<!-- TODO: dato pendiente -->`.

### 3. Honestidad Radical y Etiquetas Epistémicas
Todo dato mostrado en el sitio debe ser trazable a un archivo real del repo. En tu reporte final clasifica cada afirmación como:
- `[VERIFICADO]`: leído directamente del archivo fuente (indica ruta).
- `[DECLARADO]`: proviene de un documento del repo que no ejecutaste.
- `[SUPUESTO]`/`[ESTIMADO]`: deducción tuya — evítalas en el contenido del sitio.

### 4. Parada por Ambigüedad
Si dos fuentes del repo se contradicen (ej. una tabla dice un P@1 y el README otro), detén esa sección, reporta la discrepancia con ambas rutas y usa la fuente de mayor autoridad según §2. No promedies ni elijas en silencio.

---

## §1. FUENTES DE DATOS AUTORIZADAS (jerarquía de autoridad)

Lee SOLO estas rutas, en este orden de prioridad:

1. `README.md` (raíz) — narrativa, badges, tabla de performance, features. (leer completo, es corto)
2. `docs/reports/results-summary.md` — resumen de resultados en español.
3. `output/tables/nota3/tabla4_retrieval__exp11_retrieval194_fullrerank.md` — tabla de métricas retrieval (P@1, P@5, R@5, MRR, NDCG@5) con oráculo independiente.
4. `output/tables/nota3/` — resto de tablas Markdown de la Nota 3 (listar primero con `ls`, luego leer solo las relevantes a calidad/generación/estadística).
5. `docs/architecture/models.md` — modelos evaluados (Granite 4.1, Gemma 4, Mistral 7B, Qwen 3.5).
6. `docs/reports/summer-benchmarks.md` — contexto de experimentos (leer selectivamente con Grep: "Wilcoxon", "Cohen", "bootstrap").
7. `CITATION.cff` — datos de citación para el footer.
8. `presentacion_lacci-screenshots/` — capturas reales de la UI Streamlit: listar con `ls` y copiar las mejores 4-6 a `showcase/assets/` (copiar, NO mover).

No leas nada más. Si crees necesitar otra fuente, pregúntala en el reporte como "Alcance No Ejecutado".

---

## §2. STACK TÉCNICO OBLIGATORIO (optimizado para tokens y cero build)

- **Un solo sitio estático sin build step:** HTML + CSS + JS vanilla. Nada de npm, frameworks compilados ni bundlers.
- **CSS:** Tailwind por CDN (`https://cdn.tailwindcss.com`) o CSS propio en un solo archivo `styles.css`. Elige uno y justifícalo en una línea.
- **Gráficos:** ECharts por CDN (`echarts@5`) — renderiza las tablas de métricas como gráficos de barras/radar interactivos embebiendo los datos como JSON inline en el JS (datos literales copiados de las tablas fuente, con comentario `// fuente: <ruta del archivo>` junto a cada serie).
- **Tipografía:** una serif editorial para titulares (ej. "Fraunces" o "Newsreader" vía Google Fonts) + una sans/mono para cuerpo y datos.
- **Estructura de entrega:**
  ```
  showcase/
    index.html        (página única completa)
    styles.css        (si no usas Tailwind CDN)
    main.js           (gráficos ECharts + animaciones de scroll)
    assets/           (capturas copiadas, favicon opcional)
  ```
- El sitio debe abrirse con doble clic (`file://`) sin servidor. Verifícalo.

### Secciones mínimas de la página
1. **Hero:** "CloudRAG" + subtítulo "Hybrid RAG for Multi-Cloud Documentation" + badge "Aceptado en IEEE LACCI 2026 · Lima, Perú" + 3-4 métricas gigantes del paper (P@1, NDCG@5, nº de queries 194, nº de experimentos 19).
2. **01 / El problema:** búsqueda léxica pierde semántica, semántica pierde términos exactos (2-3 frases del README, no un ensayo).
3. **02 / Arquitectura:** diagrama del pipeline hecho en HTML/CSS puro (BM25 → BGE-large → RRF → cross-encoder rerank → NLI faithfulness → LLM local). Sin imágenes generadas.
4. **03 / Resultados:** gráficos ECharts con las tablas de `output/tables/nota3/` (comparación lexical vs semántico vs híbrido vs baseline LLM-only; significancia estadística si está disponible).
5. **04 / Modelos evaluados:** cards con los modelos de `docs/architecture/models.md`.
6. **05 / La interfaz:** galería con las capturas reales de `presentacion_lacci-screenshots/`.
7. **Footer:** citación APA/BibTeX de `CITATION.cff`, enlace al repo `github.com/EnzoOrdonez/RAG_Hybrid_System`, nota "Sitio generado como showcase del paper; todos los datos provienen de los artefactos del repositorio".

---

## §3. FLUJO DE TRABAJO EN SECUENCIA OBLIGATORIA

### Fase 1: Baseline (corta, no recursiva)
- `git status` y commit HEAD, solo para registrar que no tocarás nada fuera de `showcase/`.
- `ls` de las rutas autorizadas de §1 para confirmar que existen. Si falta alguna, repórtalo y continúa sin ella.

### Fase 2: Extracción de datos
- Lee las fuentes §1 con los límites de §0.2 y consolida un bloque de "datos verificados" (métrica → valor → archivo fuente). Este bloque es la única verdad para el contenido del sitio.

### Fase 3: Implementación
- Construye `showcase/` completo. Cada gráfico con su comentario de fuente. Cada cifra del HTML trazable al bloque de datos verificados.

### Fase 4: Verificación
- Abre `index.html` y verifica: (a) carga sin servidor, (b) los gráficos renderizan con datos, (c) las imágenes de `assets/` existen y se ven, (d) no hay referencias a rutas fuera de `showcase/` salvo CDNs.
- `git status` final: confirma que solo aparecen archivos bajo `showcase/`.

---

## §4. FORMATO OBLIGATORIO DEL REPORTE FINAL

```markdown
## 1. Acciones Realizadas
[Archivos creados bajo showcase/, con tamaños]

## 2. Datos Verificados
[Tabla: métrica | valor | archivo fuente | etiqueta epistémica]

## 3. Auditoría y Trazabilidad
- Commit base / HEAD: [hash]
- `git status`: [confirmar que solo hay cambios en showcase/]
- Verificación file:// : [PASS/FAIL por apartado]

## 4. Discrepancias Encontradas
[O "ninguna"]

## 5. Estado y Veredicto
[GO / NO-GO]

## 6. Alcance No Ejecutado (Faltantes Deliberados)
[Lo que NO se hizo y por qué]
```
````

---

## Notas de uso (fuera del prompt, no las pegues)

1. **Plugins/repos para ahorrar tokens en Kimi Code:**
   - El prompt ya fuerza CDN (Tailwind + ECharts + Google Fonts) → cero `npm install`, cero build, cero lectura de `node_modules`.
   - Si tu Kimi Code soporta skills/plugins, instala solo uno de diseño web (tipo *musepool*) para la estética; evita plugins de datos/pipelines, no aplican aquí.
   - Las tablas ya están en Markdown/JSON en `output/tables/nota3/` — el prompt obliga a copiarlas, no recalcularlas (esa es la mayor fuente de ahorro).
2. **Qué NO instalar:** nada que corra la pipeline RAG, Ollama, o el entorno Python del repo — el sitio es 100% estático.
3. Si quieres publicarlo después con URL pública (kimi.site), eso es un segundo paso: el sitio queda en `showcase/` listo para subir tal cual.
