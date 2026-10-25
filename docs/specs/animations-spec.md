# PROMPT MAESTRO — Animaciones de vanguardia para CloudRAG Showcase (Kimi K3.1 Pro / Kimi Code)

> Copia todo el bloque siguiente y pégalo como primer mensaje en Kimi Code, con el workspace abierto en `hybrid-rag-system`. El sitio ya existe en `showcase/` con un hero 3D funcional; tu trabajo es llevar CADA sección a nivel "wow" sin romper nada.

---

````markdown
# ROL

Actúa como un Creative Technologist senior especializado en WebGL y motion design editorial (nivel Awwwards/FWA). El proyecto es el showcase del paper IEEE LACCI 2026 "CloudRAG — Hybrid RAG for Multi-Cloud Documentation". El sitio ya existe y funciona en `showcase/`: hero 3D con partículas (Three.js + GLSL), scroll-pin con GSAP, gráficos ECharts, pipeline interactivo. Tu misión: elevar CADA sección con animaciones 3D/complejas de vanguardia, manteniendo rigor de datos y rendimiento.

---

## §0. GOBERNANZA (innegociable)

### 1. Blast radius
- **Escritura permitida:** solo archivos dentro de `showcase/` (puedes crear `showcase/js/*.js`, `showcase/css/*.css`, etc.).
- **Todo lo demás del repo es READ-ONLY.** No ejecutes la pipeline RAG, Ollama, benchmarks, ni instales `requirements.txt`.
- Al final, `git status` debe mostrar cambios únicamente bajo `showcase/`.

### 2. Datos congelados
- Las cifras, tablas, textos de fuentes (`// fuente: output/tables/nota3/...`) y el contenido factual del HTML NO se modifican. Puedes reestructurar el markup alrededor de ellos, pero cada número debe quedar idéntico y trazable.
- Prohibido inventar métricas, capturas o claims nuevos.

### 3. Stack y rendimiento
- Solo CDN (Three.js, GSAP + ScrollTrigger, opcionalmente Lenis para smooth scroll, ECharts ya presente). Sin npm, sin build, sin frameworks compilados. Debe seguir abriendo por `file://` y servido estático.
- **Presupuesto de rendimiento:** 60 fps en laptop con GPU integrada. Reglas:
  - `devicePixelRatio` capado a 2.
  - Toda animación WebGL/Canvas fuera de viewport se PAUSA (IntersectionObserver controla el rAF de cada escena).
  - Pausa global cuando `document.hidden`.
  - Partículas totales en pantalla ≤ 120k en desktop; en móvil (`max-width:700px`) reduce a ≤ 30% y desactiva efectos secundarios.
- **Accesibilidad:** `prefers-reduced-motion` desactiva TODO motion no esencial (ya existe un modo `?flat` y fallbacks — consérvalos y extiéndelos a lo nuevo). Todo contenido debe ser legible y operable con las animaciones desactivadas.
- **Fallbacks obligatorios:** sin WebGL → las secciones se ven completas en CSS/DOM plano. Sin GSAP → el sitio funciona con scroll nativo. Nunca dejes una sección vacía por un CDN caído (guarda con `if (window.X)`).

### 4. Honestidad y verificación
- Etiquetas epistémicas en el reporte: `[VERIFICADO]` (lo comprobaste ejecutando), `[SUPUESTO]`, `[DECLARADO]`.
- Verificación mínima antes de entregar: `node --check` en cada JS; abrir con Chrome headless y capturar screenshot de cada sección; reportar consola sin errores (`console.error`/`WebGL warnings`).
- Si algo no puedes verificarlo visualmente, dilo explícitamente en el reporte en lugar de afirmarlo.

---

## §1. ESTADO ACTUAL (léelo primero, son archivos cortos)

1. `showcase/index.html` — estructura y contenido (fuente de verdad de datos).
2. `showcase/styles.css` — sistema editorial: paper `#f2efe7`, ink `#1d201c`, accent `#9b4131`, terracota `#bc773f`, dark `#14160f`/`#20241f`. Tipografías: Newsreader (serif display), IBM Plex Sans, IBM Plex Mono. RESPETA este sistema; no cambies la paleta.
3. `showcase/hero3d.js` — hero 3D existente: 3 cúmulos (AWS/Azure/GCP) + núcleo RRF terracota, shader con ondas compuestas + force-field de cursor, pin con convergencia al scroll, chips de leyenda proyectados desde 3D.
4. `showcase/main.js` — contadores, pipeline tabs, ECharts lazy-init, lightbox, modo `?flat`.

---

## §2. ANIMACIONES A IMPLEMENTAR (una pieza signature por sección)

Sé ambicioso en UNA pieza memorable por sección antes que muchos efectos mediocres. Todo con easing físico (cubic-bezier(0.25,1,0.5,1) o springs de GSAP), nada de fades genéricos.

### Hero (mejorar lo existente, no rehacer)
- Reveal de carga: título "CloudRAG." letra a letra (clip-path o y-translate escalonado), luego subtitle, badge y contadores.
- Los chips de leyenda entran flotando con stagger una vez que la escena 3D tiene su primer frame.
- Opcional: leve distorsión shader en las partículas al hacer hover sobre el wordmark.

### 01 · El problema — "dos señales que se buscan"
- Pieza signature: canvas 2D o mini-escena WebGL sticky a un lado del texto. Al hacer scroll por la sección, un campo de puntos representa documentos: primero se filtran por coincidencia exacta de término (se iluminan pocos, modo "léxico"), luego por vecindad semántica (se iluminan otros, modo "denso"), y al final ambos conjuntos se superponen y fusionan (modo "híbrido") — scrubbed por scroll, con leyenda de estado.
- Las dos problem-cards reaccionan al cursor con tilt 3D sutil (perspective + rotateX/Y, máx 3°).

### 02 · Arquitectura — "partícula viajera"
- Convierte el pipeline en una experiencia scroll-driven: una partícula/photon recorre un path SVG que conecta las 6 etapas; al pasar por cada etapa, esa etapa se activa (reutiliza la lógica de tabs existente) y el panel de detalle cambia. El usuario puede seguir clickeando manualmente (el scroll y el click comparten estado).
- Línea de progreso del path animada con `stroke-dashoffset` scrubbed.

### 03 · Resultados — "morph de evidencia"
- Transición animada pre→post rerank: un toggle (o estado scrubbed) que transforma el gráfico de retrieval entre "antes del rerank" y "después del rerank" con morphing de barras (ECharts `setOption` animado o interpolación propia). Deja claro qué cambió y por qué importa.
- El stat-callout (`dz = +0.45`, `p_BH < 0.001`) hace count-up al entrar en viewport.
- Chart cards con tilt 3D muy sutil al hover (máx 2°) — si reduce claridad, descártalo y documéntalo.

### 04 · Modelos — "instrument cluster"
- Sección oscura: cada model-card gana un mini-gauge animado de VRAM (arco SVG o canvas, 0→6 GB) y una barra de tok/s que crece al entrar en viewport, con stagger entre las 4 cards.
- Fondo: grid técnico con leve glow que respira (opacity animada muy lenta) — solo CSS.

### 05 · Interfaz — galería con física
- Reveal de cada lámina con clip-path diagonal + saturate(0.65→1) scrubbed al entrar.
- El lightbox (dialog) abre/cierra con transición FLIP (escala desde la thumbnail hasta el modal).
- Hover en thumbnails: desplazamiento de imagen opuesto al cursor (parallax interno, `object-position` o transform).

### Citación y footer
- El bloque BibTeX se revela línea a línea (como código tecleado, muy rápido) al entrar en viewport; el botón copiar tiene micro-interacción magnética (se desplaza unos px hacia el cursor dentro de su radio).

### Global
- Smooth scroll con Lenis (CDN) integrado con ScrollTrigger (`lenis.on('scroll', ScrollTrigger.update)` + ticker). Si Lenis falla, scroll nativo.
- Barra de progreso de lectura de 2px en terracota bajo el header, scrubbed.
- Transiciones entre secciones: nada brusco; los `h2` hacen un reveal por línea con clip-path al entrar (una sola vez, `once: true`).

---

## §3. PROHIBIDO

- Cambiar paleta, tipografías o datos.
- Animaciones que oculten contenido si JS falla.
- Librerías con build (React, Svelte…), iframes, assets externos no CDN.
- Scroll-jacking que rompa el scroll nativo en móvil (en `max-width:700px` desactiva Lenis y los pins complejos si degradan UX).
- Emojis, glassmorphism, gradientes arcoíris, marquees sin propósito.

---

## §4. REPORTE FINAL (formato obligatorio)

```markdown
## 1. Animaciones implementadas
[Sección → técnica → archivos tocados]

## 2. Verificación
- node --check: [PASS/FAIL por archivo]
- Chrome headless: [screenshots por sección, consola limpia sí/no]
- file:// sin servidor: [PASS/FAIL]
- prefers-reduced-motion / ?flat / sin WebGL: [PASS/FAIL]
- git status (solo showcase/): [PASS/FAIL]
- FPS estimado y cómo lo mediste: [...]

## 3. Decisiones y descartes
[Qué efecto propuesto descartaste y por qué — con etiqueta epistémica]

## 4. Estado
[GO / NO-GO]

## 5. Alcance no ejecutado
[Lo que NO hiciste]
```
````

---

## Notas de uso (no las pegues)

1. **Orden sugerido:** primero deja que K3.1 Pro lea `showcase/` (son 4 archivos cortos — pocos tokens), luego que implemente sección por sección. Si el contexto se degrada, abre una sesión nueva y pide solo las secciones que falten, pegando de nuevo este prompt.
2. **El sitio ya publicado** (`https://75ad36seochvg.kimi.page`) corresponde a la versión actual; cuando K3.1 termine las animaciones, yo puedo republicarlo con `kimix website publish` — pídemelo y lo actualizo.
3. Si K3.1 propone algo que degrade el rendimiento o la legibilidad, su propio prompt le exige descartarlo y documentarlo — revisa esa sección del reporte.
