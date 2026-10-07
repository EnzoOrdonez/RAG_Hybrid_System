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
