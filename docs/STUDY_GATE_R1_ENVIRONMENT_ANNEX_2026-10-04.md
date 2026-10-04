# Anexo de entorno R1: imagen instrumentada y ejecución CPU/GPU

Generado por `iteration3_r1_build_annex34.py` antes de medir el candidato.
Procedencia: paquete externo de la iteración 3, directorio
`cloud-i3-r1-build01`; cada archivo fue descargado por su generación y verificado
contra el manifiesto SHA-256 nativo. No hay mediciones R1 admitidas todavía.

- Proyecto/VM/zona: el recurso original retenido, `pure-loop-474323-a8`,
  `cloudrag-study-l4-20261002`, `us-central1-a`, `us-central1`, estándar
  `g2-standard-4` con una L4, protección y disco sin autoDelete. No reubicación.
- Commit de aplicación construido (leído del recibo de imagen): `41e9a457fc03a388838a0b732225849f6e371e0c`.
- Imagen host-inspeccionada (leída del mismo recibo): `sha256:49899f394b95d80c6deb4db447134a7b71fe813e25d667d4ff7b6cc0686a66e1`,
  etiqueta `cloudrag-study:i3-r1-build01`. El commit posterior de este anexo solo añade
  este archivo; el starter verifica esa única diferencia antes de cualquier pago.
- Suite completa Linux: 856 aprobadas,
  20 omitidas explícitas; filtrada:
  851 aprobadas,
  20 omitidas. Conteos adicionales y duración
  exacta se leen de los logs, no se reconstruyen aquí.
- `pip check` limpio; siete pruebas POSIX sobre el disco persistente, incluida
  la muerte determinista antes del reemplazo. Fuente `persistent-filesystem.log`.
- Construcción de remedio Fase 7, una de máximo dos por causa corregida; se
  conservan los tres builds iniciales y su techo ya consumido.

La imagen añade solo el inventario de visibilidad CUDA y su regresión. No
cambia recetas, prompts, pesos, índices, pins, vendor ni la función de política
existente. `NPY_DISABLE_CPU_FEATURES=X86_V4` sigue fijo. Diferencias frente a
exp12: Linux/Bookworm/Python 3.14.3, L4/CPU GCP, driver y versiones efectivas
Linux (incluido PyTorch CUDA) según sus inventarios, en lugar del equipo Windows;
se conservan las diferencias de plataforma ya documentadas, sin afirmar
igualdad general de salidas. El driver exacto se revalida en el entorno vivo.

El inventario nuevo se genera en la VM, fuera del checkout, en
`/srv/cloudrag/iteration3/deployment/<job-R1>/<control|candidate>/environment_identity.json`.
Cada preflight/runner/reporte lo consume y rechaza deriva. Los dos brazos
comparten aplicación, imagen, driver, modelos y recetas; el selector de
visibilidad CUDA vacío/`0` deriva el permiso de dispositivo `0`/`1`.
Los inventarios de control y candidato son diferentes y sus posiciones no
se agregan como compuerta. Solo las dos ventanas nuevas del candidato,
si llega a ser elegible y su piloto íntegro termina, compartirán su inventario.

Rige la enmienda [R1](STUDY_GATE_R1_GPU_AMENDMENT_2026-10-04.md): doce pares
nuevos de texto exacto antes del piloto nuevo de veinte y la compuerta nueva
de 120 intentos. Ningún sintético/piloto concede GO. R2 está inelegible por
el fallo terminal de calidad NLI. Temperatura cero, umbral 60 s, tareas fijas,
q180 con reserva y ninguna sustitución/reanudación terminal.
No cambia el texto visible al participante; las esperas podrían reducirse.
Rollback CUDA vacía con identidad nueva. Sin participantes ni autorización
ética inferida de un eventual GO. No se editan documentos de Enzo.
