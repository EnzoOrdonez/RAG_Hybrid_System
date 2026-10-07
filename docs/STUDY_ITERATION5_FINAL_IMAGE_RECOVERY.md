# Conservación y restauración de la imagen final

Esta secuencia técnica no acredita aceptación del estímulo, HTTPS ni compuerta.
Se ejecuta desde el worktree autorizado mediante el broker Limited, con recibos
de comandos, límite nativo y supervisión independiente. No usa una GPU local.

`scripts.study_operator.final_snapshot` exige la descarga por generación y
SHA-256 de un build Linux aprobado, identidad consistente de imagen y commit,
los inventarios de código exportado y el estado terminal del build. Comprueba
en vivo que la VM CPU propia está detenida y que su disco retenido coincide con
el registrado. Una operación atrasada se rechaza si otro build está usando esa
VM. Verifica la tarifa mensual oficial de instantáneas del catálogo descargado,
registra la reserva hasta el plazo global e intención antes de crear el recurso.
La conversión de 730 horas por mes es una estimación, no una factura.

La interfaz es:

```powershell
& ./.venv-app/Scripts/python.exe -B -m scripts.study_operator.final_snapshot --package C:/CloudRAG/iteration5-run-20261007T020139Z --sdk 'C:/Users/enziz/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd' --label final02 --installation C:/CloudRAG/operator-iteration4/installation.json
```

El archivo de instalación anterior solo se lee. El controlador deriva una
entrada de restauración CPU propia desde la imagen y el inventario de archivos
exportados; conserva las ubicaciones y referencias de pesos e índices para
volver a comprobarlos. No instala ni modifica los operadores anteriores.

La instantánea queda `FINAL_IMAGE_SNAPSHOT_READY_CPU_RESTORATION_PENDING`.
No se sustituye por ese estado la prueba de restauración. El controlador
`scripts.study_operator.cpu_restoration` recibe un censo nuevo de recursos,
el nombre exacto de esa instantánea, la entrada CPU derivada y una etiqueta
nueva. Crea un disco y una VM e2-standard-2 sin GPU ni IP pública, con acceso
IAP limitado y apagado nativo. Verifica los archivos congelados dentro de la
imagen, los 79 artefactos, la configuración exportada de Docker y todos los
blobs de Ollama. Registra el par de importaciones con y sin `USER=cloudrag` y
comprueba STOP al terminar, incluso ante fallo.

Solo una prueba real `CPU_RESTORATION_VERIFIED`, ligada al ID de esta
instantánea, permite considerarla el recurso mínimo que reconstruye la nueva
imagen. Después se inventaría el recurso heredado redundante antes de cualquier
borrado autorizado. La VM y el disco originales, los buckets y la evidencia
en archivos se conservan. Las suites Linux sobre el disco CPU no sustituyen
las comprobaciones posteriores de identidad y durabilidad en la VM con L4.
