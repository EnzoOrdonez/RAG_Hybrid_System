# Corrección del helper de diagnóstico, antes de i4-diag02

i4-diag01 terminó sin generar respuestas: el helper omitió el manifiesto
obligatorio de artefactos. Se conserva como FALLO DE HERRAMIENTA; no demuestra
un defecto del sistema ni aporta casos válidos al diagnóstico. Se verificaron
los 18 objetos terminales por generación y SHA-256 en el paquete externo.

La hipótesis, tareas, antecedentes, brazos, repeticiones y criterios del
preregistro permanecen iguales. El nuevo helper monta la copia verificada
del manifiesto heredado y declara commit, digest, modo offline y PYTHONHASHSEED.
Conserva una instancia de pipeline por condición, como el trabajador del gate;
el índice se carga mediante el mismo factory de la app, sin cambios de lógica.
El registro captura el cuerpo efectivo de las solicitudes sintéticas completo,
además de su hash. No se vuelven a ejecutar trabajos terminales.

Los helpers viven fuera del checkout de la imagen; sus hashes son calculados
automáticamente abajo. Esta ancla se publica antes del nuevo arranque.

{
  "iteration4_diagnostic_guest.py": "119e13504b247cd940689bc517a2df50d5fcc0f09803a61fdde7350dd3ac49da",
  "iteration4_diagnostic_measure.py": "9791c9a74d681c0e4e93aac8a8386374bd49cc6e39858fc09ebab04ac56b0280"
}
