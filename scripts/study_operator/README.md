# Operador CloudRAG, iteración 4

Instalación exclusiva: `C:/CloudRAG/operator-iteration4`. El operador de la
iteración 3 y los paquetes anteriores se conservan. El runbook en español
`C:/CloudRAG/operator-iteration4/RUNBOOK.md`
define la preparación, el día de sesión, la contingencia, la recuperación y el
borrado. El ensayo completo en vivo y la aceptación de la imagen final siguen
pendientes; esta documentación no autoriza participantes.

Abre PowerShell **sin administrador**. Se usan el SDK de Google Cloud ya
autenticado y el cliente existente `C:/Windows/System32/OpenSSH/ssh.exe`.
Las llamadas privadas verifican las claves públicas devueltas por la API y el
ID de la VM; no aceptan hosts desconocidos ni usan la configuración SSH del
usuario. La clave que ya mantiene el SDK permanece en su ubicación original.
El operador no la copia, imprime ni crea otra. Si falta el cliente, la clave o
la autenticación, conserva el recibo y corrige la instalación antes de iniciar.

```powershell
& C:/CloudRAG/operator-iteration4/operator.ps1 --help
& C:/CloudRAG/operator-iteration4/operator.ps1 status
```

`study` requiere la aprobación ética y el registro creado **a mano por Enzo**,
con el PDF y B.4 aprobado, según el runbook. Los ensayos usan códigos sintéticos
y propósito `technical`; no crean ese registro ni invitaciones para personas.
Un GO técnico no habilita reclutamiento.

Las invitaciones exigen consola interactiva. El token se muestra una vez;
no uses transcripciones, redirecciones ni capturas. Solo su SHA-256 viaja al
servidor. Consulta el runbook para TTL, revocación, respaldo y margen mínimo de
70 minutos. Un preflight rechazado mantiene cerrada la admisión.

Las descargas, exportaciones por código y textos para revisión se custodian en
`private/`. `export-anonymized` produce un archivo privado seudonimizado y una
propuesta de conteos agregados: su archivo completo no debe publicarse.
`withdraw` y `purge-study` simulan por defecto; la ejecución descarga y verifica
antes de borrar, y `study` exige confirmación interactiva. Nunca retires datos
históricos ni de terceros durante una prueba.

La instalación conserva `installation.json`, estado reanudable `active.json`,
recibos técnicos en `runs/` y versiones anteriores en `versions/`. No edites los
hashes para sortear un rechazo. Ante un error, revisa el mensaje y el recibo,
ejecuta `status` y resuelve la causa; no repitas una creación a ciegas.

No se cierran aplicaciones del equipo ni se realizan mediciones locales. La
compuerta y el diagnóstico de modelos se ejecutan en la VM con supervisores,
STOP nativo, identidad y preregistros verificados. La IP regional se prepara
al menos tres días antes; al terminar sin una sesión agendada se libera.
