# Checklist humano — estado al 1-oct-2026

**NO-GO operativo** hasta completar todos los controles. Responsables: Enzo
(operación/decisiones) y mantenedor (correcciones y evidencia de software).
No se configura exposición de red, cuentas ni servicios desde esta tanda.

## Prerrequisitos antes del orden de lanzamiento

1. Confirmar `git status --short` vacío y registrar `git log --oneline -8`.
2. Verificar el sello con `scripts/manage_study.py verify-draw`, usando la venv
   congelada. SHA-256 esperado de draw_seal.json:
   `1b89a741bfab80e8a7c1a4ee5d97640f70ea439fb7e6a04d0ecf42382284a04f`.
3. Verificar B.4 en la cuenta estándar: atajos, file:, abrir/guardar, descargas,
   ejecución de programas y carpetas del investigador. Registrar resultados sin
   datos personales. Pantalla completa/ventana compartida no prueban aislamiento.
4. Elegir respaldo en otro disco físico, comprobar identidad y escritura. No
   reutilizar D: por su letra. No admitir nuevas sesiones con respaldo pendiente.
5. Ejecutar [P999 con recorrido humano](STUDY_SMOKE_P999.md); conservar el paquete,
   verificar exportación/respaldo y revisar privacidad de práctica. Si falla,
   diagnosticar antes de repetir. No es compuerta ni evidencia con personas.
6. **Antes de la compuerta, corregir y verificar los hallazgos del runner** listados
   en [el cierre](STUDY_CLOSURE_2026-10-01.md#hallazgos-heredados-que-impiden-lanzar-la-compuerta).
   No usar el CLI actual sin `--dry-run` como si fuera un lanzamiento válido.

## Orden obligatorio (sin saltos)

1. **Cohorte NLI**: desde `.worktrees/nli-521f525`, comprobar su HEAD y seguir
   `docs/NLI_BATCH_RUNNER_2026-09-21.md` de esa copia. Conservar manifiestos,
   analizar el resultado y registrar veredicto. No cambiar automáticamente a
   cross_claim aunque el candidato resulte GO; app actual sigue per_claim.
2. **Compuerta ventana 1**: sólo después de reparar el runner y completar el
   preflight aprobado. Declarar reunión Zoom y ventana compartida; comprobar
   procesos permitidos y 60 s de CPU/GPU media <10 %. Repeticiones 1–5, 60 intentos.
   Si hay fallo, inválida, expiración o interrupción, cohorte terminal: no reponer.
3. **Compuerta ventana 2**: sólo tras integridad de la primera y revalidación de
   identidad, hashes, configuración y entorno. Repeticiones 6–10, 60 intentos.
   Analizar ambas juntas: 60 válidas por condición, cero fallos/ inválidas,
   p95 caliente ≤60 s en cada condición. Una ventana sola no concede GO.
4. **Ensayo P998 por Zoom**, Enzo como participante desde otro dispositivo:
   usar un directorio nuevo y `--purpose rehearsal`, invitación P998 con celda/perfil
   explícitos. Verificar exclusión analítica, recorrido y respaldo. No es piloto.
5. **Pilotos P900/P901**: sólo tras superar los pasos anteriores y aprobación
   humana. Propósito pilot, directorio separado, consentimiento/elegibilidad fuera
   de la app. Revisar incidencias antes de autorizar el estudio de participantes.

El [pre-registro aprobado](STUDY_GATE_PREREGISTRATION.md) prevalece sobre borradores.
Los pasos 2–3 están bloqueados por software pendiente; por honestidad no se da un
comando real ficticiamente listo. No ejecutar NLI/compuerta desde el worktree de
app mientras corren tests o cambios. Las probabilidades no medidas son desconocidas.

## Guion verificable del ensayo P998

1. Operador local prepara servidor 127.0.0.1, cuenta y respaldo. Verifica reunión
   y comparte sólo ventana app; comprueba restricciones de control remoto.
2. Desde el segundo dispositivo, probar B.4 antes de introducir datos: no acceder
   a otras carpetas/programas. Si un escape funciona, NO-GO, revocar control y
   registrar el caso sin capturas de datos personales.
3. Moderación neutral: «Usarás dos sistemas. Sigue las indicaciones de cada bloque.
   Si ocurre un problema técnico, avísame.» No identificar condiciones ni sugerir
   respuestas o interpretación de escalas.
4. Completar ambos bloques, práctica efímera, tres tareas y libre, SUS/Likert,
   comparativas y cegamiento. Una declinación completa la tarea; no reemplazarla.
5. Comprobar lectura y controles con la latencia real del canal. Ante pérdida de
   Zoom, retirar el control y reprogramar; no usar Meet ni otro canal como sustituto.
   No provocar una desconexión durante una cohorte de compuerta válida.
6. Al cerrar, verificar ocho respuestas/clases, dos instrumentos y exclusión
   rehearsal. Copiar a respaldo, comparar hashes y revisar práctica sin conservar
   su contenido. Anotar incidencias y decisión humana antes del piloto.
