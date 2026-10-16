# Declaraciones I6: borrador para revisión de Enzo

Estado: PROPUESTA, sin aprobación editorial ni modificación de documentos de
Enzo. No se decide admisibilidad metodológica por él. No hay participantes ni
registro ético. Las citas siguientes remiten al paquete privado
iteration6-cowork-20261015, cuyo índice de auditoría es AUDITORIA_CODEX/INFORME.md.

## Versiones históricas no reproducibles con los bytes actuales

Fuente: AUDITORIA_CODEX/evidence/tool-hashes.json, seis referencias con match=false.
Los hashes son los registrados para esas ejecuciones; no se atribuyen a las
versiones actuales de los scripts. Dos referencias apuntan a la misma versión
histórica. El archivo vigente con el mismo nombre no sustituye al original.

| Script y hora UTC | SHA-256 histórico registrado | Evidencia pendiente |
|---|---|---|
| g_prestim.sh, 15/10 14:41:43 | 149b151c2003da09c305a30af823b4604db17fce55ae5e0db0ab831d177ecf7b | Bytes de esa versión. |
| g_prestim.sh, 15/10 14:45:19 y 14:55:09 | 9d025c963c6f3cba7945a03de9d79bd3183f851e371b46a47f407463f8a63ff1 | Bytes de esa versión, dos ejecuciones referidas. |
| g_privacy_run.sh, 15/10 23:33:26 | 8f30d2bc919e02116c6a6f2b5e6605b9751aa3262e3aadb539a24fdbd0350977 | Copia anterior a su redacción. |
| g_privacy_dates.sh, 15/10 23:34:53 | d1b10209964f3ede03c20d7871ff74194b818e45ccf6a671bc29148d6dd0944e | Copia anterior a su redacción. |
| g_privacy_run2.sh, 16/10 00:16:37 | b247ab70970957e6bdac977f53eea2189c5b725503ae393a87806ca2bd06b1bf | Copia anterior a su redacción. |

La auditoría señala también que no pudo comprobar la versión inicial del
instrumento de la Adenda 2. El anexo, punto 8, declara una copia
stimulus_host.py.orig-i6 en el disco; no se extrajo en esta tarea. Esto no prueba
que sea irrecuperable: queda pendiente verificarla contra su anclaje original.

Propuesta: «Se conservaron comandos y hashes históricos, pero no contamos con
los bytes de cinco versiones de scripts referidas en seis ejecuciones ni con
una copia descargada verificable del instrumento previo a la Adenda 2. No
presentamos las versiones posteriores como reproducción de esas ejecuciones.»

## Desviaciones, cambios y límites a declarar

Cada fila es una propuesta para que Enzo la contraste y apruebe. La aceptación
de la serie final no elimina los terminales ni acredita una compuerta.

| Hecho y fuente privada | Propuesta de redacción |
|---|---|
| Operación propia de Cowork y creación en dos pasos. PLAN_I6_COWORK.md; ANEXO, puntos 1–4. | «La operación I6 se ejecutó con herramientas propias que reutilizaron código de I5. La creación de VM y el envío del startup se separaron por errores de Compute; no equivalen a un ensayo del ciclo completo del operador I5.» |
| Retirada del reenvío de seis insumos desde el segundo arranque. ANEXO, Adenda 1, puntos 5–6. | «Los insumos se materializaron en el primer arranque y se verificaron desde disco; después no se reenviaron en metadatos por errores 503. El primer intento sin descarga confirmada se conservó como antecedente, no como prueba completa.» |
| Plazo del host de 3 a 30 s y reinicio tras dos terminales sin filas. ANEXO, Adenda 2, puntos 7–9; RUN_LOG.jsonl. | «Se corrigió el plazo de comandos del instrumento en el host. Los dos intentos previos terminaron sin muestras ni llamadas medidas según los recibos y se mantuvieron en la cronología. La serie final comenzó de nuevo con ese instrumento.» |
| Persistencia NVIDIA complementaria. ANEXO, Adenda 3. | «Se activó persistencia NVIDIA para impedir reinicializaciones del controlador durante el muestreo. El entorno efectivo incluye ese cambio del host; no se presenta como el host inicial intacto.» |
| Caché de páginas previa a cada estímulo. ANEXO, Adenda 4. | «Se precargaron los bytes del modelo en la caché de páginas. El historial de generación siguió vacío al iniciar cada prueba, pero la lectura física de los pesos desde disco quedó fuera de la primera llamada.» |
| Caché JIT conservada entre arranques. ANEXO, Adenda 5, puntos 20–22. | «Se restauró una caché CUDA previamente compilada para el driver y GPU fijados. El arranque frío se refiere al historial del servicio; no describe una caché JIT del sistema operativo vacía. Enzo debe decidir cómo declarar esta condición frente al preregistro.» |
| Sondeos de diagnóstico fuera de la serie. ANEXO, Adendas 4–5; RUN_LOG.jsonl. | «Hubo cargas de diagnóstico fuera de la serie. Se identifican por separado, se incluyen en costos y cronología y no se mezclan con la aceptación ni la compuerta.» |
| Tiempo del muestreador fijado al terminar la lectura. ANEXO, Adenda 6, puntos 23–26, autorización expresa allí registrada. | «Se corrigió una carrera del instrumento: una transición escrita durante la muestra aparecía después de su marca inicial. La observación pasó a cerrarse tras la última lectura. Se conservaron los intentos previos y la autorización del cambio.» |
| Terminal parcial y nueva serie. ANEXO, Adenda 6, puntos 23 y 26; FINDINGS F08. | «Un intento de 37 posiciones conservó 33 filas antes de terminar durante la llamada siguiente. Se excluyó de la serie final y se reinició desde el índice 1. Se declara esa sustitución de serie y su estado terminal; no se afirma que los doce arranques aceptados sean todos los intentos efectuados.» |
| Verificación histórica de concesiones de residencia. ANEXO, Adenda 7; auditoría de cierre I6, pruebas del verificador. | «La verificación posterior usó el reloj de cierre de la prueba para evitar que una concesión válida al observarse caducara por el tiempo transcurrido hasta el análisis. La regla de frescura en vivo y el umbral de 180 s se conservaron.» |
| Preparación obligatoria también para sesiones. ANEXO, Adenda 8 y PRUEBAS_FINALES. | «Una sesión sintética inicial falló sin la preparación de cachés. Se declaró obligatorio el paso previo para todo arranque con sesiones. Final03 no contenía una integración bloqueante de esa receta.» |
| Reserva no borrada pese al mensaje de la guardia. ANEXO, punto 27; capacity/log.txt; auditoría F06/F07. | «La guardia informó borrado sin confirmarlo. La reserva siguió asignada unas 2,2 h con VM detenida. El costo estimado incluye esa espera y requiere conciliación con la factura; el mensaje de la herramienta no acredita ausencia del recurso.» |
| Cobertura de privacidad y nueve registros heredados. PRUEBAS_FINALES; evidencia de auditoría F04 y LEGACY_LOG_INVENTORY. | «El escaneo I6 tenía exclusiones y cobertura incompleta de serial y objetos. La imagen conservaba nueve registros históricos con un identificador del investigador. No se declara ausencia universal de identificadores; su retirada del disco de servicio quedó pendiente.» |
| Respaldos según propósito. PRUEBAS_FINALES; evidence/final-tests.json; auditoría F08. | «La sesión technical tenía respaldo local; el recorrido smoke contó con respaldo remoto. No se atribuye a todos los propósitos la cobertura remota observada en smoke.» |
| TLS y número de arranques. Auditoría F09; technical-diagnostics.json. | «La huella final coincide con la inicial observada. Los múltiples recibos READY no demuestran por sí solos tres arranques distintos consecutivos con la misma huella.» |
| Restauración fallida y compuerta/conmutación pendientes. PRUEBAS_FINALES; auditoría F01/F02/F10. | «La restauración desplegada falló y se corrigió después en código local. La nueva imagen y su restauración requieren validación. No se midieron compuerta I7 ni conmutación; la aceptación del estímulo I6 no concede GO de servicio.» |

## Decisión editorial pendiente

Enzo debe aprobar, corregir o rechazar estas formulaciones y resolver su efecto
sobre el artículo y los documentos del estudio. Esta tarea no modifica B.4,
4.8 ni Declaraciones oficiales. No se ha accedido a personas ni creado ética.
Los terminales no descargados de la instantánea siguen siendo evidencia pendiente,
sin afirmar que se haya comprobado su inventario completo en A2.

## Procedencia de los doce contextos: archivo I4 ausente

Comprobación local del 16/10/2026: no existe
`C:/CloudRAG/autonomous-run-20261004T230147Z/verified-final-context-provenance.json`,
el archivo de procedencia I4 citado por la enmienda de servicio. Su directorio
padre sí existe. No se atribuye a ese archivo un hash ni contenido reconstruido.

Se emplea la referencia sellada de la I5:
`C:/CloudRAG/iteration5-run-20261007T020139Z/rag_freeze_baseline.json`,
SHA-256 del archivo completo:
`3616b0cc78b9de4c5fe918c5adc621197c967278872a40e4dcea479ed4ceb8cb`.
Las doce listas se leen de `rag.contexts`. La admisión liga la ruta y ese hash,
y compara las listas con los contextos vivos del entorno I7 antes de generar.
Esto no recupera el archivo original I4 ni acredita por sí mismo procedencia
viva I7; esa medición queda para B.

Propuesta editorial para Enzo: «El archivo de procedencia I4 citado en la
enmienda no estuvo disponible en el paquete consultado. La comprobación empleó
la referencia de los doce contextos conservada en el manifiesto sellado de I5,
identificado por SHA-256. Se declara esta fuente alternativa, sin presentar
como recuperado el archivo I4 ausente.»
