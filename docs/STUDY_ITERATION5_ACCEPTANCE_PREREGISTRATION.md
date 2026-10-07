# Preregistro de aceptación de la iteración 5

Este documento se ancla externamente antes de medir. El anexo de entorno posterior
identifica la VM, región, imagen e inventario generados; ningún hash escrito en
este documento sustituye sus recibos. Hasta completar esos controles, todo ensayo
sintético sigue sin conceder aceptación ni GO. No hay participantes.

## Procedimiento congelado y mecanismo

La imagen final es la del segundo build de esta iteración, identificada por su
recibo de descarga y restauración CPU. Incluye UEQ-S y respaldo al cerrar sesión.
El único mecanismo de servicio prospectivamente evaluado es `fresh_runner`,
reinicio del runner de Ollama antes de cada generación, igual para toda consulta
y ambas condiciones. Las opciones siguen siendo temperatura 0, `num_predict`
1024, semilla 42 y contexto 4096. No cambia el prompt, modelo, recuperación,
fusión, reordenamiento, balanceo, NLI ni corpus. No existen respuestas grabadas.

La evidencia heredada de variaciones genera la hipótesis de dependencia del
estado del runner; el diagnóstico prospectivo de la iteración 4 respalda el
mecanismo en cuatro objetivos, sin demostrar todavía la aceptación de los doce.
Hipótesis falsable: con un runner nuevo por consulta, las doce combinaciones
producen un solo texto, clase v2 y conjunto de citas, cualquiera de los historiales
predefinidos. Se espera eliminar las variantes, no un cambio de calidad ni una
reducción prometida de latencia. Preparación, espera de reinicio y generación
entran en el cronómetro de la respuesta. Rollback: detener el candidato y conservar
la evidencia; no volver al servicio anterior para presentar un GO sobre otra imagen.

## Censo del estímulo

Se ejecuta el calendario versionado `scripts/study_operator/stimulus_calendar.py`:
12 arranques fríos, 144 llamadas, de ellas 120 objetivos y 24 antecedentes libres
sintéticos. Las tareas selladas son T1 q001/q068/q180 y T2 q016/q070/q172; q180
conserva su reserva. No se escogen según respuestas del sistema.

Cada combinación tarea × condición tiene exactamente diez objetivos: tres del
calendario de la compuerta, cuatro de las cuatro celdas del cuadrado latino en dos
recorridos, uno como primera consulta de contenido después de un arranque frío y
dos después de las consultas libres literales del calendario. La preparación
anterior a la primera consulta fría es únicamente técnica: no se calienta el modelo
mediante una consulta de contenido. Cada combinación aparece en al menos dos
arranques. Los doce IDs de arranque son distintos; el software y artefactos son
los mismos, comprobados desde los inventarios vivos.

Antes de comparar variantes se exige el calendario completo, incluidos los
antecedentes, sin duplicados, omisiones, sustituciones, fallos ni invalidez. Se
conservan los bytes de texto crudo y mostrado, clase v2 y citas normalizadas, los
contextos recuperados, opciones enviadas y observaciones del servicio. Los
`rerank_score` se registran solo como evidencia. Límite: 600 s por llamada,
900 s de preparación y 120 min por ensayo de arranque, con STOP nativo independiente.

Aceptación: 12/12 combinaciones con exactamente una variante en cada campo y al
menos dos arranques. Los textos pueden diferir de exp12. Si alguna combinación
varía, la rama queda BLOQUEADO-HUMANO; no se declara por cuenta propia tolerable.
Un ensayo terminal, parcial o abortado no se sustituye ni se reanuda como si nada
hubiera ocurrido. Solo se continúa entre ensayos fríos completos.

## Entorno, privacidad y disponibilidad

Los doce contextos vivos deben coincidir con la compuerta de la iteración 3.
Los ensayos restantes exigen identidad nueva, invitaciones sintéticas, XSRF activo,
SA dedicada y metadatos inaccesibles desde la app. Solo 443 es público; IAP temporal.
La privacidad y borrado se deciden por la cláusula 73, con canarios de cliente en
memoria, evidencia heredada separada y los cuatro controles de borrado verificable.
Toda sesión sintética exige respaldo por generación y SHA-256, además de prueba de
restauración y retiro. Se comprueban tres arranques consecutivos ≤15 min con la misma
IP y certificado persistente, y conmutación y vuelta con identidad y smoke propios.

## Piloto y compuerta

Después del estímulo y del smoke completo con UEQ-S: piloto nuevo caliente de
20 posiciones, sin GO; después dos ventanas nuevas de 60 intentos. Las dos ventanas
usan exactamente la misma identidad, VM e imagen y un único arranque. Se conservan
fronteras y reglas de la compuerta vigente de la iteración 3: 600 s por llamada,
900 s de preparación, 120 min por ventana. Cualquier defecto de frontera aborta
la ventana completa; no se corrige a mitad ni se salva un subconjunto.

Solo el agregado íntegro de 120 intentos concede GO técnico: cero fallos, cero
inválidos y p95 con interpolación lineal ≤60 s en cada condición. No se excluyen
intentos lentos ni se decide por una ventana aislada. NO-GO conserva diagnóstico
por etapa y queda BLOQUEADO-HUMANO: no hay remedios automáticos sobre el RAG congelado.

El GO técnico vale para la misma imagen, tipo de máquina e identidad de software.
Conmutar de zona o región durante el periodo de sesiones exige identidad verificada
y smoke propios; no otra compuerta, porque el cronómetro corre dentro de la VM.
Una IP regional requiere IP, subred y certificado nuevos en otra región; las
sesiones permanecen en el bucket de us-central1 y se registra la transferencia.
Jamás se combinan ventanas de entornos distintos ni se interpreta el GO como
dictamen de aptitud para participantes o aprobación ética.
