# Anexo de entorno · iteración 3 · previo a mediciones de nube

La autorización del 3 de octubre de 2026 admite la escalera de capacidad L4 y
la emisión de certificados de Let's Encrypt para este despliegue. El primer
intento obtuvo capacidad en la VM heredada de `us-central1-a`, región
`us-central1`, con tipo `g2-standard-4`, una L4 y disco persistente retenido.
No se necesitó una instantánea ni reubicación. Los identificadores de recurso,
la respuesta de arranque y su procedencia permanecen en el paquete externo
`C:/CloudRAG/autonomous-run-20261003T205530Z/`.

Este anexo conserva las tareas revisadas (incluida q180 con su reserva), la
frontera de latencia, las dos ventanas de 60 intentos, cero fallos/inválidas y
el p95 lineal agregado de 60 segundos. La emisión HTTPS y la identidad nueva
no constituyen resultados de equivalencia frente a exp12 ni autorización ética.
No se combinan ventanas que difieran en identidad efectiva.

## Generación y consumo de la identidad efectiva

Después de commitear la fuente final limpia y construir su imagen definitiva,
el host inspeccionará la imagen y el contenedor realmente usados. Montará ese
recibo de solo lectura, junto con los manifiestos verificados de pesos e
índices, fuera del checkout. El entrypoint `freeze` generará
`environment_identity.json` en el directorio externo de esa ejecución.

La ubicación prevista en la VM es
`/srv/cloudrag/iteration3/deployment/<ejecucion>/environment_identity.json`.
El paquete de auditoría recibirá una copia cotejada por generación y SHA-256,
junto con los settings efectivos que la referencian. Entrada del contenedor,
preflight y runner verificarán el mismo inventario contra el entorno vivo;
el reporte consumirá ese archivo. Los campos y la frontera de confianza del
recibo del host están descritos en
[el anexo de inventario](STUDY_GATE_ENVIRONMENT_INVENTORY_2026-10-03.md).

La identidad registra commits completos, sello, recetas, dependencias
efectivas, driver, GPU, imagen, Ollama, digest y manifiestos HF/índices.
No se escriben aquí hashes manuales como pruebas ni se incorpora la identidad
efectiva al checkout: así se evita que incluir un hash cambie el commit que
pretende identificar. Sin suites Linux, durabilidad sobre el disco persistente,
inventario vivo verificado y este anexo commiteado, no se inicia el smoke remoto
ni una ventana de compuerta.

## Acceso y diferencias de plataforma

El proxy en la VM usará un nombre derivado de su IP efímera y Let's Encrypt,
con desafío TLS-ALPN por 443, redirección HTTP desactivada y backend loopback.
La URL y el certificado se verificarán tras cada encendido; las claves y la
cuenta ACME permanecerán privadas en el disco retenido. No hay componente
HTTPS con cargo en reposo ni IP reservada prevista. SSH administrativo será
solo por IAP con regla temporal; Streamlit y Ollama no tendrán ingreso público.

Python, paquetes Linux y entorno L4 se registran como plataforma distinta de
Windows/RTX3060 y de exp12; no se presume igualdad de salidas. Si posteriormente
cambian zona, región, VM, fuente o imagen, se documentará antes de medir y se
generará otra identidad; una ventana anterior no se reutiliza para agregarlas.
