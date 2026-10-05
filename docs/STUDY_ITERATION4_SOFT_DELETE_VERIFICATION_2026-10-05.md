# Verificación de copias soft-deleted con retención cero

Antes de guardar o borrar datos en el bucket nuevo, `objects.list(softDeleted=true)`
devuelve HTTP400: la API requiere una política soft-delete activa. El rechazo
se conserva; nunca se presenta como listado vacío exitoso. La [referencia
oficial](https://cloud.google.com/storage/docs/json_api/v1/objects/list) documenta
la restricción. No se activa retención para conseguir un listado.

El criterio material sigue siendo cero copias retenidas. Una prueba alternativa
admite el inventario de creación con retención0 y metageneración1 sin cambios;
o exclusivamente las cuatro operaciones iniciales verificadas de este bucket:
creación con `--soft-delete-duration=0`, actualización `--no-versioning` y dos
bindings IAM. Cada una se vincula con su evento de Cloud Audit Activity, hora de
servidor, insertId, intervalo del comando exitoso y SHA-256 de su recibo. El
censo de auditoría contiene esas cuatro operaciones; la lectura viva contiene
metageneración4. Los datos de IP, User-Agent y principal del registro de auditoría
se eliminaron en memoria antes de guardar el extracto técnico.

El ancla externa `sessions-bucket-creation-history-anchor.json` se genera a
partir de esas evidencias. El operador exige la misma identidad, fecha,
metageneración y retención0 en cada verificación. Una modificación posterior,
operación desconocida, evidencia incompleta o error distinto bloquea el borrado
antes de eliminar cualquier copia. Un bucket con retención anterior no satisface
la alternativa aunque después se desactive soft delete.

Los listados normales y de versiones son consultas reales. El recibo distingue
`ANCHORED_ZERO_RETENTION_SETUP_HISTORY` de `API_LIST`, y conserva
`HTTP400_POLICY_REQUIRED`. Se conserva también el rechazo literal del comando
`--soft-deleted`. No se afirma que ese comando imprima un listado vacío.

Esta precisión técnica se ancla antes de los ensayos de borrado; no cambia el
umbral material ni autoriza borrar el bucket heredado o su evidencia.