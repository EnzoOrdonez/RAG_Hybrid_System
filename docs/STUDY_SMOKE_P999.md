# P999: recorrido humano, modelos reales, verificación automática

Modalidad aprobada por Enzo el 1-oct-2026. **SMOKE_NOT_GATE**. No es un smoke
desatendido ni un resultado GO. No se ejecutó P999 real durante esta tanda.
No se instalan dependencias ni se cambian modelos/configuración. El lanzador
usa la aplicación real y deja la interacción al operador; los formularios son
sintéticos. Las pruebas de su verificador usan dobles, no miden rendimiento.

## Criterios fijados antes del lanzamiento

Una sesión sin errores: seis tareas y dos libres, ocho respuestas reconocidas
con clase/versión v2, SUS ×2, Likert ×2, C1–C4 y cegamiento; exportación sellada,
back-up en otro disco físico y hashes iguales. Cero contenido, latencias o errores
persistidos de familiarización. Latencias descriptivas, nunca criterio GO.
Una declinación es respuesta: no repreguntar, sustituir tareas ni pulsar reintento.

Ante error técnico, cerrar el smoke y diagnosticar **antes** de repetirlo. No
borrar su directorio. Un fallo de respaldo permite repetir sólo `verify`, después
de corregir el destino, sin repetir formularios. Timeout/fallo del recorrido no
puede reclasificarse como aprobado mediante `verify`. Nunca reusar un directorio
de lanzamiento ni sobreescribir otro paquete.

## Antes de ejecutar (operador)

1. Verificar checkout limpio, commit y Python congelado 3.14.3; conservar digest
   y manifiesto de despliegue previamente aprobados, no generar otros nuevos.
2. Verificar cuenta estándar, permisos y escapes B.4 en Windows Home, navegador
   sin datos personales/sincronización. Si se usa Zoom, comprobar realmente
   reunión, pantalla compartida y restricciones. El lanzador no observa Zoom ni
   certifica B.4; `--operator-checks-confirmed` es una declaración humana.
3. Elegir explícitamente destino de respaldo. No asumir que D: es otro disco:
   el lanzador resuelve particiones y puntos de montaje, compara DiskNumber y
   rechaza topologías virtuales/pools/red no demostrables. Comprobar etiquetas
   físicas del equipo y disponibilidad. No hay destino implícito.
4. Confirmar que el puerto local 8501 no está ocupado por otra app. Mantener
   Ollama preparado con el modelo/digest aprobado. No lanzar cohortes ni tests
   simultáneamente. No ejecutar el script de aislamiento en mitad del recorrido.
5. Elegir duración máxima humana (ejemplo: 45 minutos; **no es estimación de
   latencia**). El padre usa reloj monotónico y un job Windows kill-on-close para
   su propio servidor. No termina Ollama, Zoom ni servicios; no crea túneles.

Desde `.worktrees/interview-readiness` en PowerShell, completar valores reales:

```powershell
$smokeRoot = Read-Host 'Directorio NUEVO externo para P999'
$smokeBackup = Read-Host 'Destino obligatorio en OTRO disco físico'
$smokeManifest = Read-Host 'Manifiesto de despliegue ya aprobado (ruta completa)'
$smokeDigest = Read-Host 'Digest Granite ya aprobado (SHA-256)'
$smokeArgs = @('--config-dir', 'C:/CloudRAG/study-config', '--root', $smokeRoot, '--backup', $smokeBackup)
& ./.venv-app/Scripts/python.exe scripts/study_smoke.py plan @smokeArgs
& ./.venv-app/Scripts/python.exe scripts/study_smoke.py launch @smokeArgs --artifact-manifest $smokeManifest --model-digest $smokeDigest --max-minutes 45 --operator-checks-confirmed
```

`plan` no crea sesiones, consulta modelos ni copia archivos: informa NOT_RUN y
NOT_CHECKED. Revisar sus rutas antes de `launch`. No redirigir la consola del
lanzador: muestra una sola vez la invitación, que no debe copiarse a evidencia.
Abrir la URL **127.0.0.1** indicada en el navegador designado; no abrir otro puerto
ni publicar la app en la red. `app.log` no incluye la invitación impresa por el padre.

## Recorrido humano sintético

1. Entrar con la invitación; pulsar «Preparar sesión» y esperar la preparación real.
2. Primer bloque: «Probar consulta» y «Comenzar tareas». No guardar capturas, notas
   ni transcripciones de la familiarización.
3. Consultar las tres tareas fijadas, leer cada respuesta y pulsar «Continuar».
   Para la libre escribir `What are the documented storage options for AWS S3?`.
   Aceptar también las respuestas con declinación; no reintentar por su contenido.
4. Seleccionar 3 en los diez ítems SUS y los diez Likert; guardar el bloque.
5. Repetir pasos 2–4 para el segundo bloque. No cambiar las tareas del sello.
6. En C1/C2 seleccionar «iguales», C3 «ninguno» y en C4 escribir
   `SMOKE: respuestas instrumentales sintéticas, no participante`.
   Para cegamiento elegir «No sabría decir» y motivo `SMOKE sintético`.
7. Pulsar «Finalizar». El lanzador verifica y copia automáticamente. Resultado
   esperado: `smoke_verification.json`, status `VERIFIED_EXPORT_AND_BACKUP`,
   marker `SMOKE_NOT_GATE`. Confirmar los dos archivos y hashes en destino.
8. Revisar que no se añadió contenido de práctica a logs/artefactos ajenos al
   esquema. El verificador comprueba estructura y consulta fija, no puede auditar
   grabaciones de pantalla ni todos los archivos de la máquina. Registrar esta
   comprobación humana sin transcribir familiarización.

Si sólo falló la copia, corregir accesibilidad del mismo destino y ejecutar:

```powershell
& ./.venv-app/Scripts/python.exe scripts/study_smoke.py verify @smokeArgs
```

Esto no llama al modelo ni modifica `full_session.json`. El resultado no elimina
la incidencia inicial. Si falló el recorrido o expiró, conservar todo y devolver
el paquete para diagnóstico; no lanzar otro P999 automáticamente.

## Paquete y límites

`smoke_launch.json` identifica propósito, commit, sello, rutas y estado terminal;
el token no se guarda en él. Las respuestas instrumentales quedan identificadas
como sintéticas en la exportación. `export_manifest.json` y `backup_state.json`
permiten comprobar integridad y repetir sólo la copia. No se certifica el
alcance real de Zoom, aislamiento Home ni rendimiento del estudio con este smoke.
