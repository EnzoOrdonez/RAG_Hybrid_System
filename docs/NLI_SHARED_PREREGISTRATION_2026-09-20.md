# Pre-registro candidato: agrupación NLI compartida — 2026-09-20

**APROBADO por el usuario el 2026-09-21, sexta tanda. NO-GO VIGENTE.**
Se autoriza implementar el candidato, sus tests y el runner de 120 intentos;
la cohorte real sólo la lanza el humano. La ganancia del 65 % sigue siendo un
supuesto. El estado original de este documento el 20-sep era pendiente de
aprobación y no implementado. El texto siguiente conserva ese pre-registro;
su parada para aprobación queda satisfecha por la decisión explícita de 21-sep.
Avance técnico en [la nota de implementación](NLI_BATCH_IMPLEMENTATION_2026-09-21.md).

Este documento fija una propuesta revisable por el usuario antes de cualquier
cambio de producción. "Implement the plan" autoriza esta entrega documental;
no elimina la parada obligatoria previa a la optimización. El usuario eligió
NLI compartido y 120 intentos emparejados durante la planificación.

Fuente y límites: [diagnóstico completo](LEXICAL_CONFIRMATION_2026-09-20.md),
build `a06bb98c1d400458b3bd070a8e2eb748de3f1a9f`, paquete
`C:/CloudRAG/diag-run-20260920T223927918Z/`, manifiesto SHA-256
`329b492b8c68b2afaa4c5d42c36764c58b034d1f44a6fd85bc66b1cc1f5a32bf`.
Las mediciones de ese paquete generan esta propuesta; no validan la mejora.

## (a) Hipótesis falsable y crítica

VERIFICADO: en consultas de cola, el verificador realiza numerosas llamadas de
cinco pares; `batch_size=32` no agrupa entre claims. Generación y NLI dominan las
colas de ambos sistemas, con p95 70,61 s léxico y 63,18 s híbrido.

SUPUESTO A CONTRASTAR: agrupar los mismos pares entre claims permitirá una
ejecución NLI más eficiente, con un objetivo experimental de aproximadamente
65 % menos tiempo NLI y sin cambios semánticos. No está demostrada una causa
"exclusivamente léxica": la intervención propuesta afecta al verificador común.

Alternativas consideradas: recortar salida/claims arriesga calidad y cambia la
experiencia; optimizar retrieval no recupera los segundos necesarios; cambiar
modelos/hardware está fuera del alcance. La agrupación conserva el trabajo
semántico, pero puede no acelerar CPU o incluso empeorar por padding, memoria
y longitudes heterogéneas. Reducir llamadas no equivale a reducir FLOPs.
Si falla, no se añade otra optimización dentro de la misma medición.

## (b) Una sola variable: cómo se agrupan los pares

Control: algoritmo actual de `_nli_matching`, una llamada por claim elegible.
Candidato: reunir los pares elegibles en orden claim-major/chunk-major, llamar
a `predict` sobre la lista y reconstruir por claim los resultados originales.
Conservar `batch_size=32`, `apply_softmax=True`, modelo/device/dtype, límites de
secuencia, reglas de artefactos, orden de claims/chunks, extracción, umbrales,
variante `vb_agree`, margen y agregación TRUE. No deduplicar ni omitir pares.

Si la llamada agrupada falla, ejecutar el camino original por claim; si éste
falla, conservar su fallback y marcas de error. El fallback no convierte un
fallo técnico de compuerta en éxito válido. Registrar el fallo agrupado aunque
la recuperación produzca una respuesta; cualquier fallo de inferencia del
candidato incumple el requisito de cero fallos de esta validación.

El experimento futuro utilizará el mismo build instrumentado para control y
candidato, con un selector de estrategia exclusivamente en el ejecutor técnico.
No se añade un control visible al participante. Las recetas restantes deben
tener hashes iguales entre brazos; se registra la única diferencia de estrategia.
Esta entrega no introduce APIs, selectores ni cambios en el pipeline.

## (c) Magnitud propuesta y justificación cuantitativa

Objetivo experimental SUPUESTO: alrededor de **65 % de ahorro NLI**. Es una
hipótesis de rendimiento, no una predicción comprobada ni un nuevo umbral GO.
La oportunidad estructural es pasar, por ejemplo, de 14 llamadas de cinco pares
a una lista de 70 pares, procesada internamente en lotes de hasta 32. No se
asume que ese cambio de número de llamadas determine la aceleración.

Escenarios ESTIMADOS: para cada respuesta histórica, sustituir solamente
`total` por `total - fracción * tiempo_NLI`, reordenar y recalcular el p95.
Se suponen las demás duraciones idénticas; no son datos prospectivos.

| Ahorro NLI supuesto | p95 léxico | p95 híbrido |
|---|---:|---:|
| 0 % | 70,61 s | 63,18 s |
| 50 % | 61,74 s | 59,69 s |
| 60 % | 59,96 s | 58,99 s |
| 65 % | 59,08 s | 58,64 s |
| 100 %, cota ideal no realizable | 52,87 s | 53,50 s |

En esta muestra, alcanzar 60 s requiere al menos 59,79 % de ahorro NLI uniforme
en léxico y 45,56 % en híbrido. La proyección al 65 % deja poco margen; no permite
prometer GO. No hay proyección numérica semántica basada en esta cohorte porque
no contiene consultas semánticas. Se espera ahorro directo cero en retrieval,
re-ranking y generación; cualquier variación observada se reportará por separado.

## (d) Validación prospectiva y decisiones de aceptación

**120 intentos nuevos:** las 20 consultas, en el orden del manifiesto de la
cohorte fuente, por tres sistemas (híbrido, léxico, semántico) y dos estrategias
(control/candidato). "Nuevos" significa nuevas ejecuciones, no reutilizar sus
respuestas o duraciones. Unidad pareada: posición + sistema; n=20 por brazo y
sistema. El calendario se escribe y firma antes de medir.

Calendario determinista: para posición i=0..19, rotar `[hybrid, lexical, semantic]`
i módulo 3 lugares; para cada sistema con índice fijo s=0,1,2, ejecutar control
primero si `(i+s)%2==0` y candidato primero en caso contrario. Resultan diez
pares de cada orden por sistema. Mantener ambos brazos de cada par en la misma
ventana; no iniciar un par sin margen suficiente para ambos y la restauración.

Condiciones: mismo equipo/driver/Ollama/digest/semillas/timeout/offline, modelos
auxiliares en el mismo dispositivo, preparación obligatoria de los tres sistemas,
caché desactivada, telemetría durable y admisión idénticas. Verificar preparación
antes de cada intento sin cargar otro modelo ni cambiar sus parámetros.
Las fronteras del cronómetro y sus regresiones se conservan. El observador y
selector deben superar el contraste sintético antes de las inferencias.

Ejecutor humano desatendido para toda medición >15 minutos. Ventanas con límite
120 minutos y supervisor probado; cada nueva intervención NVIDIA necesita su
autorización específica. Las ventanas se planifican por bloques completos de
posiciones (seis intentos por bloque), con parada anticipada por margen. Un par
interrumpido conserva su fallo/aborto y no se repite ni reemplaza silenciosamente.
Suspensión/defecto de frontera conservan la política de aborto vigente.

**Aceptación del candidato:** 20 posiciones calientes válidas por sistema,
cero fallos/abortos, p95 total ≤60 s en **los tres sistemas**, y equivalencia de
calidad aprobada. Una invalidación deja la validación insuficiente; no se rellena
la posición fuera de protocolo. Fallos/abortos se cuentan en tasa de fallos,
nunca en percentiles. El control se reporta aunque incumpla 60 s: no es el
candidato al GO. Si faltan pares de control, no se declara confirmación causal
del ahorro. Un resultado favorable de latencia no cierra P900/resiliencia por
sí solo: se verifica que sus condiciones y contratos sigan vigentes.

Análisis fijado: cuantiles lineales p50/p90/p95, mínimo/máximo/media, fallos e
inválidos por brazo/sistema; etapas, tokens, claims, pares y diferencias dentro
de cada par; estratificación descriptiva por orden y ventana. Intervalos
exploratorios bootstrap pareado por posición, 10.000 remuestras, seed 42,
percentiles 2,5/97,5 %, recalculando p95 en cada remuestra. No usar esos intervalos
para mover el umbral GO, excluir casos o prometer generalización con n=20.
Reportar explícitamente la discrepancia frente al objetivo supuesto del 65 %,
en ambas direcciones; no usar el dato histórico para "confirmar" ese porcentaje.

## (e) Rollback

Implementación futura en un commit de producción aislado, acompañado de tests;
el soporte del ejecutor no modifica otra variable de rendimiento. Si falla
calidad, integridad, cero fallos o latencia, conservar evidencia y NO-GO, volver
al camino original revirtiendo sólo el commit de optimización y verificar la
suite. No borrar paquetes, reescribir resultados ni encadenar otro candidato.

## (f) Calidad y pruebas obligatorias antes de declarar éxito

1. Tests de correspondencia exacta entre pares, slices y claims; claims vacíos,
   cero chunks, artefactos excluidos, varias longitudes, empates y umbrales.
2. Fallo agrupado, reintento por claim y fallback original: errores visibles,
   conteos correctos, ningún fallo técnico convertido en valoración o éxito.
3. Reproducción técnica de verificación sobre textos/chunks congelados de las
   40 respuestas fuente y sobre las respuestas nuevas, sin regenerarlas ni
   modificar sus archivos: mismos claims, etiquetas, evidencia seleccionada,
   puntuaciones publicadas (redondeo vigente de cuatro decimales), método y
   agregados de faithfulness. Cualquier diferencia semántica bloquea aceptación.
4. Guardar scores sin redondear para explicar diferencias numéricas; no introducir
   una tolerancia nueva después de ver resultados. Registrar mismos prompts,
   fuentes y parámetros de generación. Las variaciones entre generaciones nuevas
   no se atribuyen automáticamente al cambio de NLI.
5. Demostrar que selección/orden de retrieval y métricas P@5/R@5/MRR/NDCG no
   cambian para entradas iguales. El cambio no debe afectar decline rate ni
   presentación de fuentes sobre el mismo texto. Estas comprobaciones prueban
   equivalencia en los casos ensayados, no una garantía universal de calidad.
6. Suite completa, Ruff, diff-check y secretos con baseline antes/después.
   Los replays reales requieren autorización futura de modelos; si exceden
   15 minutos, también se empaquetan para lanzamiento humano.

## (g) Nota UX y parada

No se cambia ningún texto visible, respuesta, cita, reloj ni aviso existente a
60/120 s. No hay un aviso nuevo cuyo texto registrar. El participante debería
percibir solamente una posible reducción de espera; es SUPUESTO hasta validar.
La condición experimental y versión se registran para no mezclar sesiones de
estudio con configuraciones diferentes. Cualquier modificación visible adicional
necesita una nota UX nueva y aprobación previa.

**PARADA OBLIGATORIA:** presentar este documento al usuario y esperar su visto
bueno explícito. No ejecutar la optimización, los 120 intentos, P2s ni nube en
esta entrega documental. Las decisiones de alcance/calendario ya tomadas no
constituyen autorización para iniciar una nueva ventana de procesos.
