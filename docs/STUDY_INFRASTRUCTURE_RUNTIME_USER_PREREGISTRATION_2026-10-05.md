# Prerregistro: usuario técnico del contenedor numérico

La observación prospectiva context21, bloqueada antes de generación, localizó
el error al importar el encoder: transformers → torch._dynamo.package →
torch._inductor.runtime.cache_dir_utils.default_cache_dir → getpass.getuser.
La evidencia conserva módulos y líneas, sin mensajes ni texto de consulta.
No demuestra todavía el resultado de la intervención.

Hipótesis falsable: el UID 10001 carece de entrada passwd y de variables de
usuario; getpass falla y evita cargar el encoder. Proporcionar únicamente
USER=cloudrag a todos los contenedores de la app permite resolver un nombre
técnico y crear la caché temporal bajo /tmp, manteniendo UID/GID 10001,
aislamiento, imagen, pesos, índices, plantillas y opciones de generación.

Alternativas: usar root debilitaría el aislamiento; reconstruir con passwd
consumiría un cuarto build prohibido; TORCHINDUCTOR_CACHE_DIR evitaría solo una
llamada específica, dejando el mismo defecto de usuario en otras bibliotecas.
Se elige una variable de infraestructura, idéntica para ambas condiciones y
para cada consulta, independiente del ID de tarea. No se compila ni modifica
la lógica RAG. La imagen final conserva su digest.

Ensayo nuevo: dos contenedores con la imagen exacta, UID 10001, red none,
rootfs de solo lectura, /tmp en memoria y logs descartados. El primero conserva
el entorno sin USER; el segundo añade solo USER=cloudrag. Importan getpass y
torch._dynamo, sin modelos, recuperación ni generación. Criterio: el control
reproduce OSError en getpass; el candidato resuelve cloudrag e importa Dynamo
sin error. Los contenedores se conservan con --rm=false y tienen límite propio.
Los comandos y parsers se validan antes del encendido pagado.

Después del ensayo se instala exclusivamente deployment.py del Host, con
archivo previo y recibo por SHA-256. Un nuevo arranque y observador con generación
bloqueada deben recuperar los 12 contextos iguales a R1. No se usa un contexto
fallido como aprobado ni se sustituye una cohorte terminal. Si el ensayo o los
contextos fallan, se conserva la evidencia y se diagnostica antes de intervenir.

Efecto esperado: eliminar el fallo de inicialización, no una reducción de
latencia cuantificada. La aceptación del estímulo y el piloto/compuerta nuevos
siguen siendo obligatorios sobre la misma imagen y la infraestructura final.
No se atribuye ninguna mejora de rendimiento con estas pruebas.

Rollback: conservar y restaurar el deployment.py previo y el archivo de
instalación previo, con la VM detenida y admisión cerrada. No borrar modelos,
índices, datos, registros ni evidencia; no crear una imagen adicional.
