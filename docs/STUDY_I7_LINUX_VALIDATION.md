# Validación local Linux I7

La suite se ejecuta en Docker Desktop local, sin dispositivos GPU y con red
deshabilitada durante las pruebas. La base es la misma de la imagen:
python:3.14.3-slim-bookworm con su digest fijado en Dockerfile. Los logs completos,
Dockerfile de pruebas, manifiesto del vendor y scripts de repetición están en
el paquete privado iteration7-faseA2-20261016.

Se mantiene el lock de la app. La dependencia privada se exporta del commit
fijado y se verifica con locked_vendor_requirements.py, cambiando solo su
transporte. Playwright es una dependencia adicional del entorno de pruebas
del dueño; no se incorpora al Dockerfile congelado de la app. Se fijan por
plataforma el árbol del paquete y el ejecutable de Chromium. Un recibo de
navegador de otra plataforma no acredita el ejecutable local.

El checkout de pruebas queda en /opt/cloudrag/repository, igual que la imagen.
Los modelos e índices locales se montan de solo lectura y se copian al espacio
propio del contenedor. Los originales no se modifican. Git conserva el historial
necesario para las comprobaciones de fuente; no se necesita red para las pruebas.

Los ensayos iniciales se conservaron: la ruta demasiado superficial hizo que
las guardas de exclusión abarcaran la raíz, y otro ensayo no tenía los índices
ignorados por Git. Se corrigió la preparación del contenedor, sin relajar guardas,
sin cambiar RAG y sin añadir marcas de omisión. La primera ejecución completa
con preparación correcta pasó 1770 pruebas, con 39 omisiones ya definidas para
contratos de Windows y sus herramientas. La ejecución final posterior a R8
queda en linux-suite-final.txt del paquete.

Las cuatro pruebas omitidas en Windows por POSIX se ejecutaron: descriptores de
directorio, interrupción antes de rename y ciclo de socket Unix. El constructor
y el saneamiento offline se incluyeron. El subconjunto previo pasó 64 pruebas.
Esto valida código local y contratos con dobles; no califica una VM ni reemplaza
el build CPU y las mediciones desplegadas de noviembre.
