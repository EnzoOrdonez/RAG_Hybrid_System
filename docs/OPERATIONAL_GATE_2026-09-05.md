# Compuerta local: registro de ejecución

## Crítica previa a la corrección

El navegador real mostró rutas de operador antes del login en modo participante.
La evidencia inicial permanece en
`C:/CloudRAG/operational-20260905T1428Z/finding-participant-navigation.json`.
`st.stop()` no impide el descubrimiento de `pages/`. Tampoco basta invocar
`st.navigation` dentro de la entrada mientras existe esa carpeta: una primera
visita directa puede ejecutar la página autodetectada antes de la entrada.

La corrección acordada mueve las pantallas a `src/ui/views/` y registra una sola
página de participante. Mantiene las herramientas de desarrollo privadas y el
protocolo existente, incluidos descansos y preguntas abiertas. No se cambian
modelos, recuperación, cuestionarios ni el umbral de latencia.

AppTest puede ejecutar el caso de una página (comprobado en memoria con Streamlit
1.54.0), pero no sustituye la prueba de rutas desde navegador. Las regresiones
comprobarán el registro antes/después del login y la ausencia de módulos
autodetectables; el navegador probará también la primera URL tras reiniciar.

El traslado incorpora siete avisos Ruff preexistentes en pantallas de desarrollo.
Se eliminan únicamente imports y asignaciones sin uso, conservando los controles;
no se aprovecha esta corrección para cambiar la lógica de esas pantallas.

Baseline provisionado: 383 pruebas aprobadas, 5 excluidas, ninguna omitida.
Las dos regresiones de rutas fallaron antes de la corrección; después pasaron junto
con la suite completa: 385 aprobadas, 5 excluidas (12,07 s).

## Criterio y alcance

GO local requiere cuatro pasos satisfactorios: bundle, sesión completa, latencia y
resiliencia. Medición acordada: 20 consultas frías y 20 calientes por sistema,
p95 ≤60 s en cada combinación, sin excluir fallos ni observaciones lentas.
Las sesiones son sintéticas y se almacenan fuera del checkout. La preparación de
nube solo comienza después de GO local; no autoriza despliegue ni entrevistas remotas.
