# Operación del despliegue de estudio

Este procedimiento se usa con el anexo de entorno del 2026-10-03 y el inventario
externo generado. No concede permiso para reclutar participantes: Enzo debe
obtener primero la aprobación del Comité de Ética e Integridad de la Facultad
de Ingeniería y completar sus documentos pendientes. Las invitaciones de la
iteración autónoma son únicamente para scripts de prueba, con propósito `smoke`.

## Antes de encender

Los scripts de operación y sus recibos se entregan fuera del checkout, en
`C:/CloudRAG/operator-iteration3/`. El paquete de auditoría final conserva sus
fuentes y hashes; las ejecuciones posteriores escriben en directorios nuevos,
sin modificar ese paquete. Consultar allí el resultado real de validación y el
archivo de configuración que referencia la imagen verificada. No reutilizar
un arranque terminal de build, smoke o compuerta.

En PowerShell, ejecutar los scripts con el Python registrado por la instalación:

```powershell
python C:/CloudRAG/operator-iteration3/operator.py status
python C:/CloudRAG/operator-iteration3/operator.py start
python C:/CloudRAG/operator-iteration3/operator.py preflight
```

`start` coteja el ID de la VM, su protección, la retención del disco y la ausencia
de otra GPU encendida; reserva costo antes de encender y exige STOP nativo. El
backend solo escucha en loopback. La regla administrativa IAP es temporal y no
abre SSH público. HTTPS usa exclusivamente 443, la IP efímera actual y Let's
Encrypt bajo la autorización de Enzo; no se reserva una IP ni un balanceador.

Cada encendido obtiene la IP actual, verifica el certificado público y genera
un inventario externo nuevo para el driver/GPU, imagen, fuente, dependencias,
recetas, pesos e índices realmente presentes. Los settings que lo referencian
también son nuevos; no se sobrescribe una admisión anterior. `preflight` falla
ante deriva o respaldo pendiente. La URL válida está en el recibo de arranque;
no se presume que la URL de una ejecución anterior siga vigente.

## Sesiones y apagado

La app requiere una invitación individual. No publicar tokens en el repositorio,
el paquete, capturas o comandos registrados. Un certificado válido no sustituye
la admisión de identidad ni la autenticación. El modo técnico entregado conserva
el propósito de prueba hasta que Enzo habilite una sesión bajo aprobación ética.

No apagar hasta comprobar que todos los respaldos están completos. El recibo
de respaldo debe identificar la generación del objeto y su SHA-256, y la copia
descargada al equipo de Enzo debe coincidir. Ante respaldo pendiente se bloquea
la admisión y se conserva la evidencia; no se elimina la sesión.

```powershell
python C:/CloudRAG/operator-iteration3/operator.py stop
python C:/CloudRAG/operator-iteration3/operator.py status
```

`stop` detiene únicamente la VM verificada, desarma sus trabajos de arranque y
elimina únicamente la regla IAP temporal creada por ese script. Mantiene disco,
bucket, certificados privados y evidencia. Verificar `TERMINATED`, protección
activa y disco retenido en el recibo. El límite nativo detiene la VM aunque el
equipo local pierda conexión; no dejarla encendida sin actividad necesaria.

Disco y bucket siguen cobrando con la VM detenida. Sus tarifas y costos diarios
figuran en el ledger de la entrega; HTTPS no agrega costo en reposo. No iniciar
una sesión cuya reserva completa supere el corte conservador de USD45.
