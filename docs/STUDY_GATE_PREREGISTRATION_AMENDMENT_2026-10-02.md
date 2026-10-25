# Enmienda prospectiva: compuerta del despliegue GCP

Autorización: instrucciones de Enzo de 2026-10-02, §§0.2, 0.5–0.7 y A1–A7.
Se publica antes de cualquier smoke o medición real de esta corrida. No modifica
el artículo, B.4, el corpus ni los preregistros NLI/WARM.

## Entorno y condiciones

Proyecto único `pure-loop-474323-a8`, región `us-central1`, zona `us-central1-a`,
VM estándar no Spot `g2-standard-4`, una NVIDIA L4. Linux Ubuntu LTS y contenedor
Python 3.14.3. La imagen concreta, driver, Ollama, commit, dependencias efectivas,
digest completo del modelo y hashes de HF/índices se fijarán en el inventario
de despliegue antes de admitir ventanas. Ningún valor abreviado es una prueba
de identidad. Un cambio de esa identidad impide combinar ventanas.

Tratamientos: `SURVEY_DEPLOY` y control sin recuperación de la misma app;
balance por proveedor activo en híbrido. Auxiliares inicialmente CPU, Ollama
en L4; misma política en UI y runner. Caché de respuestas desactivada, seed 42,
contexto 4096, `keep_alive=30m`, timeout de lectura 180 s y sin reintento
automático adicional. Los flags sin consumidores se inventariarán, no se activan.

Zoom se usa solo para videollamada y entrevista y deja de ser requisito de la
compuerta. El participante utilizará su navegador. TLS válido e invitación son
requisitos de despliegue; la latencia Lima–nube se informa descriptivamente.

## Calendario, frontera y decisión

Las seis tareas proceden exclusivamente del nuevo sello aprobado por la auditoría
literal del corpus. No se puede iniciar una compuerta si ese proceso no encuentra
seis tareas emparejadas válidas. No se incorporan respuestas de septiembre.

Dos ventanas: repeticiones 1–5 y 6–10, 60 intentos por ventana. Orden alternado
por repetición y posición, balanceado por condición. Solo el agregado puede dar
GO: 60 observaciones válidas por condición, cero fallos, cero inválidas y p95
caliente <=60 s por condición, percentil lineal NumPy. Se informan además p50 y
los resultados por ventana, sin decisiones separadas.

Cronómetro monotónico desde antes de la solicitud durable hasta el payload
presentable: incluye preparación/residencia, recuperación, reranking, generación,
NLI y citas. Excluye espera de lock, calentamiento, publicación final, transporte,
pintado y lectura. Un defecto de frontera termina la cohorte; nunca se rescatan
sus datos como válidos. El diario previo y cada resultado se publican por intento.

Preflight real: commit limpio, sello, recetas, versiones instaladas, manifiesto
y artefactos verificados; modelo único residente con digest y contexto correctos;
60 s de CPU/GPU media <10%, sin navegador, overlay, cómputo GPU ajeno ni carga CPU
ajena sostenida >=10%. Telemetría cada 5 s, hueco máximo 15 s, comprobación de
residencia durante la ventana. Ventana 2 exige ventana 1 íntegra y revalidación
completa. En Windows también se aplican los controles de energía existentes.

Supervisor en proceso independiente: máximo 600 s por llamada, 900 s para
preparación y 120 min por ventana. El límite de seguridad no cambia el timeout
HTTP ni el umbral de aceptación. Interrupción, fallo, contaminación o expiración
producen estado terminal; no se reintenta una posición ni se completa una cohorte
terminal. El modo sintético nunca concede GO.

## Remedios prospectivos

Como máximo tres rondas en orden R1–R3. Cada candidato requiere diagnóstico por
etapa, hipótesis falsable, mecanismo, efecto esperado justificado, rollback y
enmienda específica commiteada antes de implementarlo/medirlo. Una variable por
candidato; inseparabilidad requiere justificación. Piloto nuevo caliente de 20
posiciones y después compuerta nueva completa; ninguna reutilización de intentos.

- R1: infraestructura/ejecución, con texto generado exactamente idéntico antes y
  después en las seis tareas y ambas condiciones, a temperatura cero. Si difiere,
  el candidato no es R1 elegible; no se redefine equivalencia.
- R2: `cross_claim` por lotes únicamente con GO de equivalencia completo conforme
  al preregistro NLI. Si NLI está bloqueado o NO-GO, R2 queda inelegible.
- R3: modificación de generación simétrica solo después de R1 y de evaluar la
  elegibilidad/resultado de R2. Revisar otra vez v2 en seis tareas y declarar la
  diferencia frente al sistema del artículo, sin editar el artículo.

El umbral nunca cambia. Un GO de latencia no sustituye los smokes, la cobertura
del corpus, la seguridad, el respaldo ni la restauración. Un NO-GO o una rama
bloqueada se preservan como tales.

## UX

La enmienda no añade mensajes al participante. Continúan las etiquetas A/B y
los textos actuales de la app. Los fallos del preflight/supervisor son información
operativa externa; no muestran métricas, configuración ni condición al usuario.
