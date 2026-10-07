# Iteración 5: restauración antes de reducir la retención

Este protocolo se publica antes de crear el clon de prueba. No mide respuestas,
latencia del RAG ni una compuerta. Nunca califica el despliegue para participantes.

## Alcance y fronteras

`scripts.study_operator.cpu_restoration` comprueba el sello externo heredado,
las identidades de la API, el presupuesto y los apagados independientes antes
de crear un disco desde una instantánea. El candidato inicial es
`cloudrag-i4-user-recovery-20261006`; su ID se obtiene del inventario de la API,
sin introducir hashes a mano. Se conserva toda la evidencia heredada.

El clon tiene e2-standard-2, sin GPU, IP pública, cuenta de servicio ni scopes.
Usa IAP con una regla propia temporal. Su disco se retiene; tiene protección de
borrado y STOP nativo a las dos horas. El arranque reemplaza el trabajo heredado
en el clon, detiene los contenedores heredados y programa otro apagado a los
110 minutos. El controlador pide y verifica STOP aunque falle la prueba.

La duración técnica registrada abarca la comprobación de admisión, creación,
preparación SSH, lectura y hash de archivos, prueba de importación y STOP. No se
interpreta como latencia del participante. El supervisor local tiene un límite
de 65 minutos; el STOP nativo no depende de la conversación ni de ese supervisor.

## Criterios anteriores al resultado

La restauración solo se acepta si la VM y el disco observados corresponden a sus
IDs registrados y a la instantánea seleccionada; coinciden todos los archivos
de lógica congelada y los 79 artefactos esperados; el archivo de configuración
de la imagen Docker coincide con su digest; y coinciden el manifiesto Ollama y
todos sus blobs. Estos valores se consumen desde los inventarios existentes.
La prueba no lee ni exporta contenido de sesiones.

Por separado, se prueba el candidato de usuario de runtime en dos contenedores
retenidos de la misma imagen, sin red y con UID 10001. Se limpian USER, LOGNAME,
LNAME y USERNAME en ambos; solo el segundo establece USER=cloudrag. La hipótesis
se sostiene únicamente si el primero falla por getpwuid/UID 10001 y el segundo
importa TorchDynamo y devuelve el usuario esperado. Un resultado distinto se
conserva como no respaldado; no se transforma en éxito por la mera restauración
de archivos. No se cargan modelos ni se generan respuestas en esta comparación.

Solo después de un recibo real de restauración y STOP se admite un plan de
borrado con inventario y comprobación de IDs. Ese plan conserva la VM y el disco
originales, los dos buckets, la instantánea calificada y toda la evidencia en
archivos. Un resultado sintético no habilita ningún borrado. Antes de eliminar
cada recurso se verifica su identidad en vivo; después, su ausencia en la API.

## Costos y reversión

La cotización de CPU se deriva del catálogo oficial de SKUs descargado. El
ledger registra por separado cómputo, disco, retención, reservas y márgenes.
La exposición inicial reserva dos horas de CPU y hasta 72 horas del disco de
100 GiB; es una reserva, no gasto facturado. No hay una IP externa en esta prueba.

Ante fallo se apaga el clon, se conservan los recibos y no se borra redundancia.
El disco, la VM de prueba, los contenedores retenidos y la regla IAP se declaran
desechables antes de crearse y solo se retiran tras registrar sus metadatos.
El candidato se vuelve a comprobar después del build final; el primer clon no
se presenta como instantánea de la imagen final con UEQ-S.

## Enmienda de observabilidad, anterior al segundo ensayo

El primer ensayo conservó un fallo de la ejecución privada por SSH y un STOP
verificado, pero no dejó la etapa del fallo: el stderr privado se representó
solo por su hash. No se atribuye ese fallo a una causa inventada ni se borra nada.
El segundo ensayo incorpora un resultado JSON de error por etapa y clase de
excepción, con la versión de Python, sin mensaje privado ni traceback del
invitado. Reutiliza el mismo clon detenido y reserva por separado otro máximo
de dos horas de CPU. Conserva el recibo STOP anterior y sus registros originales.
No cambian los criterios de coincidencia, la instantánea ni la lógica del RAG.
Si revela un defecto, se registra antes de corregir y no se transforma este
ensayo diagnóstico en evidencia de restauración aprobada.

La lectura del instalador heredado confirmó una incompatibilidad del controlador:
`host_code` solo recibe `scripts/study_operator`, no un checkout completo. La
prueba verifica allí los hashes de infraestructura declarados en la instalación
y verifica los 19 archivos congelados dentro de la misma imagen Docker
restaurada, cuya configuración y directorio de trabajo se comprueban primero.
El contenedor de lectura de fuente también se retiene, sin red, GPU ni registros
de Docker. No se sustituye ningún hash ni se reconstruye la imagen para aprobar.
El primer error remoto sigue sin atribución exacta; el segundo ensayo dará una
etapa explícita si falla. Esta corrección verifica el objeto correcto sin mover
el criterio de identidad del RAG.

El ensayo diagnóstico informó FileNotFoundError en ARTIFACT_FILES y confirmó
Python 3.12.3. La comprobación se ajusta al mismo espacio de archivos efectivo
que `cloud_entrypoint.verify_manifest`: índices y modelos se leen en los binds
del directorio de activos; los artefactos restantes, incluido el catálogo de
consultas versionado, se leen dentro de la imagen. Se conservan los 79 hashes y
se exige la suma completa de ambas ubicaciones; no se omite una entrada, caché
o archivo pendiente del manifiesto. Esta corrección de ubicación se publica y
prueba antes de una nueva ejecución sobre el mismo clon. Ambos ensayos fallidos
y sus respectivos STOP permanecen inmutables.

El tercer ensayo verificó las ubicaciones de los activos, pero falló en
IMAGE_CONFIG: la ruta privada de almacenamiento de Docker asumida por el
controlador no existe. No se infiere el driver ni otra ruta privada. Se cierra
ese método de lectura y se usa la interfaz documentada `docker image inspect`
y `docker image save --output`: el ID observado debe coincidir y se verifican
los bytes de la configuración exportada. En el almacén clásico su hash es el
ID; en containerd el ID identifica el manifiesto OCI, cuyo hash se verifica
primero y cuya referencia a la configuración debe coincidir con su hash y tamaño.
El archivo de exportación propio se conserva y declara antes de crearse.
Las pruebas incluyen configuración alterada e ID distinto. No se reconstruye
la imagen, no se cambia el criterio y no se reemplaza ninguno de los tres fallos.
Fuentes: [inspect](https://docs.docker.com/reference/cli/docker/image/inspect/)
y [save](https://docs.docker.com/reference/cli/docker/image/save/).

La lectura de `i4-build03/image-inspect.stdout` del paquete sellado confirma
que la imagen heredada usa un descriptor de manifiesto OCI y GraphDriver nulo.
Este es un defecto de dominio de hash en el controlador, detectado antes del
siguiente ensayo; no una alteración del entorno. Se añaden casos de manifiesto
y configuración modificados. La identidad del ID se deriva de la instalación,
sin introducir a mano hashes. Fuente de la implementación de Docker:
[inspección containerd](https://github.com/moby/moby/blob/master/daemon/containerd/image_inspect.go).
