# BORRADOR — operación y piloto del estudio de usuarios

2026-09-27. **NO-GO para piloto con personas y estudio real.** Software probado
con dobles, no validado todavía en el despliegue del estudio. No se configuró
túnel, acceso remoto ni nube. No se reclutó/contactó a nadie.

## Decisiones y alcance

- D1: mantener la cohorte NLI aprobada sin cambios; ejecutarla desde la copia
  aislada `.worktrees/nli-521f525`, HEAD `521f525`. [Compuerta separada propuesta](STUDY_GATE_PREREGISTRATION_DRAFT.md)
  pendiente de aprobación. Ningún resultado sintético concede GO.
- D2: decidir entre las modalidades siguientes antes de exponer la aplicación.
- D3: revisar T1=q001/q064/q171 y T2=q010/q070/q172; son propuesta verificada
  contra el catálogo de 194 consultas, no una asignación ya aprobada por el investigador.
  Las seis son de dificultad media; cada conjunto tiene factual/procedimental AWS y
  comparación AWS/Azure. Validación informática no demuestra equivalencia cognitiva.
- D4: usuario pega SUS literal validado de Sevilla-Gonzalez et al. (2020), con
  solo «herramienta»→«sistema». Los diez marcadores se dejan vacíos. No corregirlos
  ni sustituirlos por los ítems sintéticos de tests.

## Modalidad D2 — comparación y recomendación

| Modalidad | Seguridad / privacidad | Latencia y cargas | Esfuerzo |
|---|---|---|---|
| Laptop + videollamada con control remoto | Puede permitir interacción con todo el escritorio: cerrar documentos, notificaciones, terminales y otras apps; usar cuenta aislada. Proveedor de videollamada/acceso procesa su tráfico. Riesgo de exposición accidental mayor que una sola aplicación. | Incluye transporte/compresión de pantalla y eventos de control, carga CPU/GPU de video. Incremento **NO MEDIDO**; no asumir que el piloto previo lo cubre. | Menor si existe herramienta institucional aprobada; comprobar alcance de permisos y revocación. |
| Laptop + túnel temporal autenticado hacia una sola app | Restringe la superficie publicada a Streamlit; exige autenticación externa, HTTPS y token de estudio. Proveedor del túnel/proxy pasa a formar parte del tratamiento de datos. Un hostname público por sí solo no equivale a control de acceso. | Navegador del participante y red hacia proxy/laptop. Incremento **NO MEDIDO**; validar ida/vuelta, WebSocket y reconexión en el entorno elegido. | Mayor: política de acceso, HTTPS, origen local, expiración y retirada del túnel. |

**Recomendación (juicio técnico, pendiente de D2): túnel autenticado limitado a la
app**, con cuenta/proveedor permitido por la universidad y aprobación de privacidad.
No usar un túnel público anónimo. Si esa revisión no está disponible, no sustituirla
por exposición improvisada. La opción remota sigue siendo viable con escritorio
aislado y prueba de latencia propia; no es la opción seleccionada automáticamente.

Fuentes primarias para los alcances, no benchmarks: [Microsoft Quick Assist](https://learn.microsoft.com/en-us/windows/client-management/client-tools/quick-assist)
describe compartir pantalla/control; [Cloudflare: publicar una aplicación](https://developers.cloudflare.com/learning-paths/clientless-access/connect-private-applications/create-tunnel/)
y [exigir protección Access](https://developers.cloudflare.com/cloudflare-one/access-controls/access-settings/require-access-protection/)
separan conectividad y control de acceso. No se crean cuentas ni se contratan servicios.

## Preparación de configuración (operador, antes de reclutar)

1. Trabajar en `.worktrees/interview-readiness`, rama `fix/interview-readiness`.
   Verificar `git status --short` y `git rev-parse HEAD`. No ejecutar cohortes durante
   tests, videollamadas o cambios de código. Modelo/digest/driver se mantienen.
2. Copiar `config/study.example.json` y `config/study_assignments.example.csv` a un
   directorio privado **externo al checkout**, por ejemplo `C:/CloudRAG/study-config/`.
   No usar datos reales para probar. Guardar UTF-8.
3. Llenar `labels` con una permutación global de `hybrid`/`no_rag`, idéntica para
   todos. Completar SUS literal. Revisar tareas y consulta fija «What is cloud computing?»:
   propuesta de una sola consulta genérica, repetida en ambos bloques, fuera de T1/T2
   y de las ocho premisas inválidas. La app no admite una de esas ocho como tarea.
4. Completar CSV P01–P20, `role=primary`, `cell=1..4`, `profile=without_experience`
   o `with_experience`. Cuotas por celda sin/con experiencia: 3/2, 2/3, 3/2, 2/3.
   Reservas opcionales P21–P24 con `role=reserve`, celda/perfil fijados antes de reclutar.
   No poner nombres, correo, empresa, consentimiento ni notas en ese CSV.
5. Validar y congelar; el comando rechaza SUS vacío, tareas inválidas, columnas
   identificables o cuotas erróneas. Desde el worktree, en PowerShell:

```powershell
$studyConfig = 'C:/CloudRAG/study-config/study.json'
$studyCsv = 'C:/CloudRAG/study-config/assignments.csv'
$studyRoot = 'C:/CloudRAG/study-pilot-NEW'
$studyArgs = @('--config', $studyConfig, '--assignments', $studyCsv, '--root', $studyRoot, '--purpose', 'pilot')
& ./.venv-app/Scripts/python.exe scripts/manage_study.py @studyArgs freeze
& ./.venv-app/Scripts/python.exe scripts/manage_study.py @studyArgs preflight
```

El preflight CLI solo verifica configuración e integridad: informa explícitamente
`deployment=NOT_CHECKED`, `warmup=NOT_RUN`. No confundirlo con preparación de modelos.
El sello impide cambiar mapeo, textos, CSV y tareas durante la cohorte. Un cambio
posterior requiere otro directorio/protocolo autorizado, nunca editar el sello.

## Pre-flight operativo antes de cada sesión

1. Validar configuración sellada con el comando anterior. Confirmar consentimiento
   y elegibilidad **fuera de la app** según el procedimiento del curso/comité.
2. Verificar el bundle confiable con `scripts/check_deployment_artifacts.py verify`
   y el manifiesto ya provisionado. No crear un nuevo snapshot para ocultar diferencias.
   Verificar `/api/tags` contra el digest Granite esperado y registrar commit/versiones.
3. Verificar AC, plan de energía y espacio libre para sesiones/exports. Cerrar cargas
   ajenas; el navegador que usa la app y el canal D2 elegido son parte del despliegue,
   no se exige cerrarlos. No intervenir servicios NVIDIA/AnyDesk sin autorización nueva.
4. Directorio de sesiones privado, disco local persistente, una inferencia/sesión
   activa. Respaldar exportaciones cerradas y comprobar sus manifiestos. No sincronizar
   archivos de checkpoints mientras se escriben; parar nuevas sesiones para una copia
   coherente del directorio completo.
5. Variables obligatorias (rutas locales ilustrativas; adaptar sin secretos en repo):

```powershell
$env:CLOUDRAG_MODE = 'participant'
$env:CLOUDRAG_STUDY_CONFIG = $studyConfig
$env:CLOUDRAG_STUDY_ASSIGNMENTS = $studyCsv
$env:CLOUDRAG_STUDY_SESSION_DIR = $studyRoot
$env:CLOUDRAG_STUDY_PURPOSE = 'pilot'
$env:CLOUDRAG_BUILD_ID = (git rev-parse HEAD).Trim()
$env:CLOUDRAG_ARTIFACT_MANIFEST = 'C:/CloudRAG/operational-20260905T1428Z/deployment-manifest.json'
$env:CLOUDRAG_MODEL_DIGEST = '444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852'
$env:OLLAMA_HOST = 'http://localhost:11434'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONHASHSEED = '42'
$env:PYTHONUTF8 = '1'
$env:CUDA_VISIBLE_DEVICES = ''
```

6. Tras aprobación para piloto, emitir P900+ en directorio piloto separado:

```powershell
& ./.venv-app/Scripts/python.exe scripts/manage_study.py @studyArgs invite P900 --cell 1 --profile without_experience
& ./.venv-app/Scripts/python.exe -m streamlit run src/ui/app.py --server.address 127.0.0.1 --server.enableCORS true --server.enableXsrfProtection true
```

El token se muestra una vez; compartir por canal privado y no pegarlo en logs/repo.
En disco se conserva su hash. La configuración Streamlit histórica desactiva CORS/XSRF;
por eso estas opciones explícitas son necesarias. **Este comando solo escucha localmente**;
no configura la modalidad D2 ni autoriza exposición de red.

7. Entrar con la invitación y, antes de entregar el control, pulsar «Preparar sesión».
   El proceso calienta los dos pipelines y NLI de la condición que lo usa. Registra
   evidencia operacional fuera del checkout. Comprueba identidad y residencia antes
   de cada consulta; una preparación de otro participante no es válida. Si caduca,
   volver a preparar antes de continuar y documentar incidencia operativa.

## Moderación y secuencia

- Presentación neutral: «Usarás dos sistemas. Sigue las indicaciones de cada bloque.
  Si ocurre un problema técnico, avísame.» No identificar condiciones ni interpretar
  escalas durante las respuestas. No sugerir que la demora indica mayor calidad.
- Familiarización fija → tres tareas del conjunto asignado → consulta libre, con
  advertencia literal visible **antes** del campo → SUS/Likert de ese bloque.
  No hay valoraciones de utilidad/exactitud por consulta.
- Repetir el flujo del segundo bloque; después C1–C4 y chequeo de cegamiento.
  Entrevista breve posterior fuera de la app, siguiendo el protocolo ya existente.
- Aclaración del prompt, no enmienda del artículo: familiarización no persiste
  contenido, fuentes, latencia ni errores de reintentos; solo avance y timestamp.
  No tomar capturas/exportar respuestas de práctica. La consulta libre sí se conserva
  completa para análisis cualitativo. Calentamiento no depende de la familiarización.
- Si se pierde conexión: misma invitación; respuesta guardada no se regenera.
  «Comprobar consulta pendiente» solo recupera después de liberar el lock. Si bloquea
  la sesión, el operador registra incidente sin contenido de práctica:
  `... manage_study.py @studyArgs incident UUID`.
- Abandono: `... manage_study.py @studyArgs abandon UUID`; revoca invitación y conserva
  errores/datos existentes. Nunca completar instrumentos por la persona ni imputar.

## Reemplazo, exportación y respaldo

Para estudio real usar otro directorio, `purpose=study`, CSV congelado e invitaciones
P01–P20. Una reserva solo se activa para la misma celda/perfil, después de abandonar
la sesión primaria: `... manage_study.py @studyArgs replace P01 P21`, luego `invite P21`.
La reserva no cambia cuotas ni CSV; queda trazabilidad de la plaza reemplazada.

Exportación consolidada en **ruta nueva**:

```powershell
& ./.venv-app/Scripts/python.exe scripts/manage_study.py @studyArgs export C:/CloudRAG/study-export-NEW
```

Incluye sesiones cerradas, análisis pareado, diccionario y SHA-256. `full_session.json`
por UUID conserva respuesta original y fuentes mostradas. Los abandonos se conservan
pero no se imputan al contraste. Exportación repetida por participante es idempotente;
la consolidada exige ruta nueva y verifica hashes antes de analizar.

Respaldar en almacenamiento institucional cifrado con acceso limitado y comprobar
hashes tras copiar. No publicar el JSON con textos libres: revisar posibles datos
confidenciales y producir una copia redactada trazable si se comparte. Consentimiento
firmado, contactos y correspondencia código/persona permanecen separados. Plazo de
conservación/borrado y proveedores D2 deben decidirlos investigador/comité; no se
inventan aquí. Tras entrevistas, detener la app y retirar el acceso D2 conforme al
procedimiento que se apruebe. No hay script destructivo ni borrado automático.

## Checklist de piloto (1–2 personas fuera de la muestra, pendiente)

1. Aprobar D1/D2, llenar/revisar D3/D4 y superar compuerta técnica real antes de
   autorizar piloto. Confirmar permisos/consentimiento del curso para esas personas.
2. Usar P900/P901, `purpose=pilot` y directorio nuevo; comprobar que no entra a análisis.
3. Verificar ambos bloques en el orden/celda asignados, textos y ausencia de pistas
   propias de UI; las fuentes siguen siendo diferencia funcional declarada.
4. Probar reconexión y error técnico sin duplicación ni instrumentos fabricados.
5. Confirmar exportación con dos SUS, diez Likert por bloque, seis tareas, dos libres,
   C1–C4, chequeo final y **ningún contenido de familiarización**.
6. Registrar problemas, decidir corrección y repetir validación técnica si corresponde.
   El piloto no concede automáticamente GO al estudio de veinte participantes.

## NLI aislado

La copia autorizada `.worktrees/nli-521f525` tiene bundle físico verificado y la
venv enlazada al entorno existente; no actualizar dependencias. Usar allí el checklist
`docs/NLI_BATCH_RUNNER_2026-09-21.md` de **esa copia**. No ejecutar desde el worktree
de app modificado. No hay ejecución NLI en esta tanda ni autorización de procesos
renovada por este documento. No solapar la medición con tests o preparación de app.
