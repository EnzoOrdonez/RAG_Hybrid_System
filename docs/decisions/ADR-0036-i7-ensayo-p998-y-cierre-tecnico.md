# ADR-0036: Ensayo P998 de la fase humana y decisiones de cierre técnico

[DECLARADO] Registrado el 2026-11-17 sobre la imagen de b5e54f1 y la raíz del GO del intento 15
(ADR-0035). Tarea F-020, paso F. Enzo hizo el recorrido en el navegador y pegó él mismo la
invitación; Claude operó la capacidad, el cierre y el registro. Las decisiones de este ADR siguen
la delegación de Enzo del 17/11 ("Puedes hacer todas tu excepto la 7 con las opciones mas
recomendadas?"). Este ADR no cambia código.

## Resultado del ensayo P998

[VERIFICADO en recibos privados del operador y en la bitácora del intento 15]

| Momento | Resultado |
|---|---|
| Capacidad, 16/11 | 42 rondas de `reserve --hours 4` en `us-central1-a` (ADR-0026): 41 rechazos liberados y 1 reserva retenida a las 11:16Z. READY a las 11:22Z, pero la ventana venció sin recorrido: la VM se apagó sola al llegar al plazo del huésped. La reserva expiró por su TTL nativo y se reconcilió (`RESERVATION_ABSENT_RECONCILED`). |
| Capacidad, 17/11 | Reserva retenida en la ronda 2 (03:02Z) y READY a las 03:08Z. |
| Invitación | `INVITATION_HASH_REGISTERED` a las 03:14Z, con `token_not_persisted` y TTL de 24 h. Enzo pegó el token; el token no pasó por el chat ni por disco. Un `invite` lanzado antes del READY falló con "La VM no está RUNNING" sin escribir recibo, como corresponde. |
| Recorrido | Respuestas sintéticas hasta la pantalla de cierre «La sesión ha terminado. Gracias por participar.». |
| Cierre | `stop` dio `TERMINATED_VERIFIED` (04:06Z) y `guard` dio `RESERVATION_DELETED_VERIFIED` (04:07Z). La lectura de GCP muestra la VM detenida y 0 reservas. |
| Gasto | USD 13,06 según el ledger del operador: Fase B 9,46, primera ventana P998 2,83 y segunda 0,77. Exposición abierta 0. No es la factura. |

## Hallazgo: un segundo clic en «Preparar sesión» muestra un error genérico

[VERIFICADO en código y por reproducción local; INFERIDO para el ensayo]
- El primer clic del ensayo mostró «No se pudo continuar. Contacta al coordinador antes de
  reintentar.». Tras recargar la página y volver a entrar con la misma invitación, el recorrido
  siguió sin incidencias.
- `study_page._flow` llama a `runtime.prepare(session)` sin aviso de espera. La preparación real
  tarda unos 52 s: tres preparaciones sintéticas archivadas del 15/11 dieron 51,3 a 51,6 s, sin
  fallos.
- Streamlit 1.54.0 tiene `runner.fastReruns` activo por defecto, y la app no lo cambia. Una segunda
  interacción durante esos segundos detiene la ejecución en curso y crea otra, aunque la primera
  sigue bloqueada en `prepare` (`AppSession.request_rerun`).
- La segunda ejecución pide `FileLock(_inference.lock, timeout=0)` y recibe `filelock.Timeout`.
  `study_page.render` lo captura y muestra el mensaje genérico. La primera ejecución termina la
  preparación.
- Una reproducción local con dos hilos y una preparación simulada dio `ok` para el primer clic y
  `filelock._error.Timeout` para el segundo.
- Explica por qué el smoke y el gate pasaron: el navegador automatizado hace un solo clic y espera.
- Sin verificar: el evento `_preparation` del ensayo, que sigue en el disco de la VM. La hipótesis
  predice `preparation_started` y `preparation_ready` sin `preparation_failed`. Se lee al purgar
  las sesiones sintéticas.

## Decisión

[DECLARADO]
1. El ensayo P998 cumple la parte técnica del paso F. F-020 sigue `active`: faltan la aprobación
   ética escrita y las sesiones con personas (paso G).
2. No se cambia código. El paso 7 de [STUDY_DEPLOYMENT_RUNBOOK.md](../STUDY_DEPLOYMENT_RUNBOOK.md)
   pide un solo clic en «Preparar sesión», esperar alrededor de un minuto y, si aparece el error,
   recargar y volver a entrar con la misma invitación. El arreglo en código (aviso de espera, botón
   desactivado o manejo del bloqueo) exige imagen nueva y otra Fase B, y queda para después del
   estudio.
3. La purga de las sesiones sintéticas de los intentos 14 y 15 y del ensayo P998 se aplaza al
   cierre del estudio. Exige otro boot GPU, y la capacidad L4 es escasa. Los códigos P900 y
   superiores no son personas y quedan fuera del análisis. El ensayo funcionó con esas sesiones
   presentes.
4. Se conservan la VM GPU detenida con su disco de 180 GB, el snapshot limpio del intento 15, los
   tres snapshots base y la IP estática. Cuestan unos USD 37 al mes de almacenamiento. La fase
   humana los necesita: liberar la IP cambia el nombre del host y el certificado, y eso exige otra
   Fase B. Se revisan al cerrar el estudio.
5. El identificador del proyecto GCP aparece en `scripts/` y en documentos que ya están en `main`.
   Es un identificador, no una credencial: el acceso depende de IAM. No se reescribe el historial,
   porque exigiría un force push. No se cambia ahora, porque tocar `scripts/` invalida la raíz del
   GO (ADR-0035). Se pasa a configuración cuando cierre el estudio.

## Límites conocidos

[DECLARADO]
- La capacidad L4 en la zona fijada es el cuello de botella de la fase humana: 2 reservas en 44
  rondas entre el 16 y el 17/11. Cada sesión con personas debe programarse con margen y con el
  coordinador presente al READY.
- Tras el READY, la VM se apaga sola unas 2 h 40 min después. Si el recorrido no empieza en esa
  ventana, se pierde el boot.

## Verificación

[VERIFICADO]
- Recibos: `SPECIFIC_RESERVATION_HELD`, `INVITATION_HASH_REGISTERED`, `TERMINATED_VERIFIED` y
  `RESERVATION_DELETED_VERIFIED`.
- Lectura de GCP tras el cierre: VM detenida y 0 reservas.
- `init.sh` con exit 0 en las tres capas sobre el árbol con la nota del runbook.

## Actualización del 2026-11-17: F-023

[DECLARADO] Con OK de Enzo («haz la 1,2,4 …») y su elección de aplazar el identificador del proyecto:
- **Aviso de espera en «Preparar sesión» (decisión 2):** corregido en el código de `main` (tarea
  F-023). La página muestra un aviso mientras prepara; si otra ejecución ya prepara o el bloqueo
  de inferencia está tomado, muestra «La preparación de la sesión está en curso…» con el botón
  «Comprobar preparación», en vez del error genérico. Tests en
  `tests/test_study_prepare_feedback.py`. La imagen desplegada b5e54f1 no lo incluye, así que la
  nota del runbook sigue vigente para la fase humana.
- **Identificador del proyecto (decisión 5):** sigue aplazado al cierre del estudio. Aparece en 25
  sitios del operador, entre ellos el guardián de alcance de `cloud_client` y scripts que corren
  dentro de la VM. Esos caminos solo se prueban de verdad en la nube, y el identificador ya es
  público en el historial.
