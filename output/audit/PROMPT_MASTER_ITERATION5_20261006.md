# PROMPT MAESTRO · ITERACIÓN 5 (para Codex o Claude Code) · completar la iteración 4, agregar el UEQ-S y lograr la aceptación del despliegue
# Fecha: 2026-10-06 | Rama: fix/interview-readiness (worktree .worktrees/interview-readiness) | HEAD esperado: el último commit publicado por la iteración 4 (al 06/10 14:03 UTC era afb365c o posterior)
# Agente: Codex o Claude Code, el que tenga cuota, en el equipo Windows de Enzo, en modo autónomo y sin privilegios de administrador. Si la cuota se agota a mitad de camino, el otro agente continúa con el relevo de la cláusula 77.
# Arranque: después del cierre automático de la iteración 4 (ver §0.1). Pegar TODO este contenido tal cual.

---

## §0. Contexto y decisiones de Enzo (2026-10-06), definitivas para esta iteración

La iteración 4 (Codex, paquete `C:/CloudRAG/autonomous-run-20261004T230147Z/`) remedió parte de los hallazgos ALTO de la auditoría de la iteración 3. Quedó sin terminar por dos motivos: no hubo GPU L4 disponible en ninguna zona de `us-central1`, y Codex agotó su cuota.

Antes de quedarse sin cuota, Codex dejó programado un cierre autónomo con cuatro tareas de Windows de nivel Limited; el 06/10 a las 15:13 UTC corrigió el desfase de milisegundos de sus disparadores y verificó 428 evidencias sin discrepancias. Tú continúas en una **iteración 5 con paquete propio**, seas Codex o Claude Code. Tu objetivo tiene dos partes: completar y corregir lo que la iteración 4 dejó pendiente o mal hecho, y agregar lo que Enzo decidió después de la exposición del parcial.

Decisiones cerradas de Enzo; no las reabras:

1. **Primero se respeta el cierre automático de la iteración 4.** Estas tareas corren solas en el equipo de Enzo el 06/10 (hora de Lima):

   | Hora | Tarea | Qué hace |
   |---|---|---|
   | 15:28:30 | `CloudRAG-I4-GlobalClosure-20261004` | Detiene y verifica la nube |
   | 15:35:30 | `CloudRAG-I4-PostClosureIP2-20261006` | Libera la IP estática |
   | 15:45:30 | `CloudRAG-I4-ClosureAudit-20261006` | Auditoría de cierre |
   | 16:05:30 | `CloudRAG-I4-CloseReportHandoff2-20261006` | Reporte final de 20 secciones; luego, el trabajo de sello hacia las 16:35:30 |

   No modifiques, canceles ni adelantes esas tareas. No hagas ninguna acción que cambie la nube antes de comprobar en la Fase 0 que el cierre terminó, o antes de las 17:15 de Lima del 06/10, lo que ocurra después.

   Si el cierre falló o quedó incompleto, lo completas tú siguiendo su propio plan (`global-closure-*`, `closure-audit*`, `close-report-handoff02*`, `final-seal-job01-plan.json`). Registras cada paso y no reescribes la evidencia existente.
2. **Alcance de la iteración 5:**
   - **(a)** todo lo que el reporte de la iteración 4 lista en «Falta» y «Bloqueos»;
   - **(b)** lo que la iteración 4 hizo mal (cláusula 68);
   - **(c)** el UEQ-S en la app y en el análisis, antes del build final (cláusula 71);
   - **(d)** la regla estadística del estudio implementada en código (cláusula 72);
   - **(e)** la compuerta nueva sobre la imagen final.
3. **Capacidad: Enzo autoriza otras regiones de EE. UU.** (cláusula 69). Primero se reintenta `us-central1` con pausas. Si no hay L4, se usa la primera región de EE. UU. con L4 disponible, en orden de latencia medida desde Lima. Va con anexo de entorno antes de medir, identidad nueva, subred, IP estática y certificado propios de esa región. Los datos se quedan en EE. UU.; el bucket de sesiones sigue en `us-central1`.
4. **Retención: Enzo autoriza limpiar lo redundante** (cláusula 70).
   - **Se conservan:** la VM y el disco originales de la iteración 2, los dos buckets, una instantánea de la imagen final con prueba de restauración y toda la evidencia en archivos.
   - **Se borran:** las demás instantáneas y los discos y VM de prueba, después de registrar sus metadatos.
   - **Meta:** bajar el costo en reposo de ≈ USD 1,16 a ≈ USD 0,45 por día o menos.
5. **UEQ-S y regla estadística.** Enzo los decidió el 06/10 por la retroalimentación del parcial y ya figuran en el artículo V2.17 (secciones 4.7 y 4.8).
   - **El UEQ-S:** versión corta del User Experience Questionnaire, con 8 ítems. Es un desenlace secundario validado y se aplica justo después del SUS en cada bloque.
   - **La regla del análisis del estudio:** Shapiro-Wilk sobre las diferencias pareadas del SUS; t pareada si son normales y Wilcoxon si no; la otra prueba como sensibilidad; d_z e intervalos por bootstrap; Benjamini-Hochberg solo sobre las dos escalas del UEQ-S. Los ítems Likert propios se reportan de forma descriptiva.
6. **Privacidad: criterio decidido** (Enzo eligió «registros nuevos», D01 de la iteración 4; cláusula 73). El criterio literal de «cero IP en todo el paquete» queda reemplazado por el de la cláusula 73. No es una relajación: separa la evidencia técnica de infraestructura de los datos de personas.
7. **Borrado: criterio verificable** (cláusula 73). GCS rechaza listar objetos borrados de forma temporal cuando la retención es 0 (HTTP 400). Por eso el listado `--soft-deleted` vacío se reemplaza por una prueba equivalente que sí se puede demostrar.
8. **La lógica del RAG sigue congelada** (cláusula 57): recuperación, fusión, reordenamiento, plantillas, ruteo, balanceo, modelo, `num_predict` 1024, temperatura, NLI, índices, corpus y pesos. El UEQ-S y la regla estadística no tocan esa lógica.
9. **Presupuesto:** sigue el techo de 100 USD acumulados, con corte propio a 90.
   - **Gasto real:** el reporte de uso de la consola dice PEN 31,20 entre el 1 y el 6 de octubre (≈ USD 9,20 con 3,39). Es DECLARADO por Codex, no una factura final.
   - **Estimado conservador de Codex:** USD 11,40 más márgenes separados.
   - Llevas el estimado y los márgenes por separado y vuelves a proyectar el periodo de sesiones después de la limpieza.
10. **Ética:** no hay participantes ni invitaciones reales. El registro ético lo crea solo Enzo. El trámite ético todavía no se envió.
11. **Documentos de Enzo: no los toques.** El artículo, la ficha, el B.4 y el documento C los actualiza Claude (Cowork) aparte. Tú entregas `DATOS_B4_V6.md` con lo que realmente quedó.
12. **Plazo:** techo de 72 horas de reloj desde tu primer comando, con presupuestos por fase según la cláusula 33. El techo no se extiende por pausas de cuota (cláusula 74).
13. **Independencia:** la auditoría de cierre de esta iteración la hará un agente que no haya implementado nada en ella. Tú no te autoevalúas como apto (cláusula 76).

## §0-bis. Cómo trabajar según el agente

**Para ambos agentes:**
- **Sistema y shell:** el equipo es Windows. Usa PowerShell para tareas programadas y el Python de `.venv-app` del worktree para los scripts del proyecto, como hizo la iteración 4.
- **Mediciones largas:** van como tareas programadas Limited o procesos en segundo plano con su propio supervisor y límite duro, nunca como esperas dentro de la conversación (cláusulas 22 y 26).
- **Cuota:** puedes quedarte sin cuota en cualquier momento. Antes y después de cada efecto pagado, y al cerrar cada fase, deja `STATE.json`, `RUN_LOG.md`, `COMMANDS.log` y `HANDOVER.md` consistentes, de modo que el mismo agente u otro pueda retomar leyendo solo esos archivos (cláusulas 74 y 77).
- **Aprobaciones:** si tu herramienta te pide aprobar una acción que este prompt ya autoriza, Enzo puede no estar presente. Agrupa al inicio todo lo previsible (cláusula 13) y avanza en las ramas que no dependan de una aprobación pendiente.
- **Identidad del agente:** registra en `STATE.json` y en cada línea de `RUN_LOG.md` qué agente y qué modelo ejecutó cada paso.

**Si eres Claude Code:**
- El equipo tiene un hook local, GateGuard (plugin ECC), que bloquea la primera escritura de cada archivo hasta que expongas ciertos datos. Si aparece «[Fact-Forcing Gate]», no es un fallo: responde con lo que pide y reintenta (cláusula 75). Enzo puede haber definido `GATEGUARD_EXEMPT_GLOBS` para `C:/CloudRAG/**`.
- Tu conversación puede compactarse. Después de una compactación, vuelve a leer `STATE.json` y `HANDOVER.md` antes de actuar.

**Si eres Codex:**
- No repitas el patrón de la iteración 4 de crear decenas de scripts de un solo uso en la raíz de `C:/CloudRAG` (cláusula 68).
- Si necesitas una decisión de Enzo que este prompt no resuelve, la rama queda `BLOQUEADO-HUMANO` con la pregunta exacta en el checklist humano, y sigues con lo demás.

## §1. Estado heredado

### 1.1 VERIFICADO por Claude (Cowork) el 06/10 a las 14:17 UTC, por lectura del paquete de la iteración 4

- **`STATE.json`:**
  - `status` ACTIVE, fase 4;
  - `acceptance_status` = `NO_REAL_STIMULUS_SMOKE_OR_GATE_ACCEPTANCE_YET`;
  - plazo 2026-10-06T22:28:29Z;
  - cierre reservado desde las 20:28:29Z.
- **Estado por fase:**
  - **Fase 1:** `DIAGNOSIS_SUPPORTED_SERVICE_IMPLEMENTED_FINAL_ACCEPTANCE_PENDING`. El diagnóstico prospectivo respaldó reiniciar el runner antes de cada generación: un solo texto, una clase y un conjunto de citas en los 4 objetivos probados, con seis PID distintos por objetivo. Falta la aceptación de las 12 combinaciones.
  - **Fase 2:** `DEDICATED_SA_AND_PRIVATE_BUCKET_VERIFIED_ISOLATION_PENDING`, con `literal_privacy_package_zero` = FAILED.
  - **Fase 3:** `OPERATOR_SOURCE_VERIFIED_LIVE_INSTALLATION_AND_LIFECYCLE_PENDING`.
  - **Fase 4:** `FINAL_IMAGE_FUNCTIONAL_PASS_L4_CAPACITY_BLOCKED_ALL_AUTHORIZED_ZONES`. Detalle:
    - builds 3 de 3;
    - `final_live_contexts_and_freeze` NOT_VERIFIED;
    - `new_GPU_measurements` NOT_LAUNCHED;
    - commit de anexo `afb365c`;
    - candidato de usuario de runtime `4f43cfc`, porque UID 10001 sin nombre en passwd hacía fallar `getpass` al importar TorchDynamo.
- **Capacidad:** varios `ZONE_RESOURCE_POOL_EXHAUSTED` en `us-central1-a`, `-b` y `-c` entre el 5 y el 6 de octubre.
- **Recursos vivos según `STATE.json`:**
  - dos VM TERMINATED, la original y `cloudrag-i4-alternate-b4-final-20261005` en `us-central1-b`;
  - dos discos persistentes;
  - siete instantáneas de ≈ 53 GB (`bootstrap`, `build02f-recovery`, `final-recovery`, `user-recovery`, `preserve-b3`, `preserve-c2`, `preserve-storage`);
  - IP estática `cloudrag-i4-static-20261005`, que se libera al cerrar;
  - bucket original y bucket `cloudrag-study-i4-103950017681-20261004`;
  - SA `cloudrag-study-i4@…`;
  - regla IAP temporal.
  - En reposo cuesta ≈ USD 1,156 por día sin la IP. `session-period-cost-projection05.json` proyecta USD 92,63 aun sin espera, por encima del corte de 90.
- **`REPORT_WORKING05.md`, sección «Falta»:**
  - aceptación del estímulo (144 llamadas, 120 objetivos);
  - congelamiento final con los 12 contextos vivos e identidad final;
  - 3 arranques calificados;
  - continuidad y vuelta (failback);
  - smoke público completo;
  - disparo automático del respaldo al cerrar sesión;
  - privacidad (cláusula 59) completa;
  - listado `--soft-deleted` vacío;
  - runbook (cláusula 66) de principio a fin;
  - piloto de 20;
  - dos ventanas de 60;
  - conciliación de la factura;
  - cierre real y manifiesto final.
- **Último paso de Codex (06/10, 15:13 UTC):** verificó de forma independiente los hashes de 428 evidencias y abrió el disyuntor del comprobador de inventario completo tras tres fallos distintos (ruta del bundle, tareas sin disparador y codificación de stderr). Ese comprobador no quedó aprobado.
- **Calidad del código:** los controles más recientes conservan un fallo de Ruff previo. En la raíz de `C:/CloudRAG` hay decenas de scripts `iteration4_*.py` de un solo uso fuera del repositorio.

### 1.2 Lo que debes leer en la Fase 0

- El reporte final y el sello que haya dejado el cierre automático.
- `REPORT_WORKING05.md`, `DECISIONS_LOG.md` y `COST_LEDGER.md`.
- El runbook `docs/STUDY_OPERATOR_ITERATION4_RUNBOOK.md`.
- Los documentos `docs/STUDY_ITERATION4_*` y `docs/STUDY_GATE_*_ITERATION4_*`.
- El paquete de auditoría de la iteración 3 (`C:/CloudRAG/audit-iteration3-20261004T204327Z/`), con sus hallazgos.

### 1.3 Promesas de B.4 V5 que la implementación debe cumplir (texto del documento entregado)

- (a) Solo el investigador y su asesor acceden a los datos.
- (b) Los datos se guardan asociados a un código, sin nombre.
- (c) Consultas, respuestas, tiempos y cuestionarios quedan en Google Cloud durante el periodo de sesiones y, al terminar, se descargan y se eliminan del servidor.
- (d) Los resultados se reportan agregados y los datos anonimizados pueden publicarse.
- (e) El participante usa los sistemas desde su navegador, sin instalar nada.
- (f) La consulta libre se pide sin datos personales.
- (g) La sesión tiene una consulta de familiarización, tres tareas y una consulta libre por sistema, cuestionarios después de cada sistema, preguntas comparativas y una entrevista, en 45 a 60 minutos.
- (h) El tratamiento sigue la Ley 29733.
- (i) El participante puede retirar sus datos mientras sigan vinculados a su código.

## §2. Autorizaciones de esta iteración

- **B1. Código, pruebas y documentación** del worktree, con commits atómicos, dentro de la cláusula 57. Los scripts nuevos van versionados en el repositorio (`scripts/study_operator/` o equivalente), no sueltos en `C:/CloudRAG` (cláusula 68).
- **B2. Google Cloud** sobre `pure-loop-474323-a8`, dentro de §0.9:
  - encender y detener VM propias;
  - builds nuevos, hasta 3 en esta iteración;
  - reservar y liberar IP estáticas regionales;
  - crear discos desde instantáneas;
  - crear VM en `us-central1` y, si hace falta, en otra región de EE. UU. con L4 (cláusula 69);
  - crear una subred en esa región dentro de la red del estudio, con las reglas de firewall equivalentes (solo 443 público e IAP temporal);
  - borrar los recursos redundantes de §0.4 según la cláusula 70;
  - presupuestos, alertas y reglas IAP temporales.

  Prohibido:
  - borrar la VM o el disco originales, los buckets o evidencia en archivos;
  - reservar capacidad;
  - crear recursos fuera de EE. UU.;
  - tener dos GPU encendidas a la vez.
- **B3. Mediciones reales en la nube:** congelamiento final, aceptación del estímulo, ensayos, smoke, piloto y compuerta.
- **B4. Let's Encrypt:** sigue vigente la aceptación de Enzo para la cuenta ACME de este despliegue (contacto `20221789@aloe.ulima.edu.pe`), incluidas las emisiones para las IP estáticas de esta iteración.
- **B5. Descarga del material oficial del UEQ** desde `https://www.ueq-online.org/`, solo para tomar de forma literal los ítems en español y las reglas de puntuación del UEQ-S (cláusula 71).
- **B6. Borrado sintético:** purga y retiro solo sobre datos sintéticos propios (A5 de la iteración 4).
- **B7. Push** de `fix/interview-readiness` sin force ni merge, nunca a `main`, con pushes intermedios para anclar preregistros (cláusula 65).
- **B8. Instalación** del operador actualizado en `C:/CloudRAG/operator-iteration5`. Los operadores 3 y 4 y los paquetes anteriores quedan en solo lectura.
- **No se autoriza:**
  - administrador;
  - cerrar aplicaciones de Enzo;
  - cambiar la lógica RAG;
  - crear el registro ético;
  - invitar a personas;
  - editar documentos de Enzo;
  - modificar o adelantar las tareas de cierre de la iteración 4.

## Cómo rigen las cláusulas en esta iteración

Las cláusulas 1 a 44 y 57 a 66 se copian sin cambios de la iteración 4. Aplican con estas precisiones:

- **Autorizaciones:** las referencias a «§0» y «§2» se leen como las de este prompt.
- **Cláusula 24a:** el corte es 90 USD.
- **Cláusula 41 y la contingencia de capacidad de la 64:** se reemplazan por la cláusula 69.
- **Cláusula 59 y el criterio de listado de la 60:** se reemplazan por la cláusula 73.
- **Cláusula 43:** no aplica.
- **Cláusula 63:** su tabla de atributos se amplía con el UEQ-S (cláusula 71) y con el plan de análisis (cláusula 72).

Ante cualquier conflicto prevalecen las cláusulas 67 a 77 y las decisiones de §0.

## §3. Cláusulas de calidad obligatorias (1–22, texto vigente de `output/audit/CLAUSULAS_CALIDAD_CODEX.md`)

1. **No repudio (trazabilidad total).** Cada afirmación del reporte debe poder reconstruirse: commits con hash, archivos de evidencia con ruta y timestamp, `git status` y `git log` en la sección Auditoría. Si no hay evidencia, no hay afirmación.
2. **No creas a ciegas ningún plan — ni el tuyo ni el mío.** Antes de implementar, critica el plan: ¿es la mejor opción?, ¿qué alternativas existen?, ¿qué fallas no evidentes podría tener?, ¿qué supuestos esconde? Documenta la crítica. Si encuentras una opción mejor, proponla y pregúntame antes de cambiar de rumbo.
3. **Verifica antes y después.** Antes de tocar nada: registra baseline (suite de tests, `git status`, estado relevante). Después de cada cambio: re-corre la suite completa, Ruff, `git diff --check` y escaneo de secretos con baseline. Todo cambio de código trae sus tests; un cambio sin test es un cambio incompleto.
4. **Pregunta ante ambigüedad o contradicción.** Prohibido asumir "la verdad absoluta". Si dos fuentes discrepan, si el protocolo no define un caso, o si una premisa del prompt resulta falsa: detente en ese punto, documéntalo y pregúntame.
5. **Honestidad radical de fallos.** Un fallo se reporta como fallo, con su evidencia. Prohibido maquillar, suavizar, omitir o convertir un ❌ en ⚠️. Un NO-GO bien documentado vale más que un GO falso.
6. **Etiquetas de certeza.** Distingue siempre: VERIFICADO (lo comprobaste ejecutando/leyendo), DECLARADO (lo dice un documento, no lo verificaste), SUPUESTO (inferencia tuya), ESTIMADO (proyección numérica con supuestos explícitos).
7. **Reversibilidad e idempotencia.** Prefiere cambios revertibles y scripts re-ejecutables sin efectos duplicados. Nada destructivo sin autorización explícita.
8. **Ámbito acotado.** Trabaja solo donde el prompt indica (rama/worktree). Sin push, sin merges, sin tocar evidencia congelada (`experiments/results/`, gold, corpus, CSVs históricos, `paper/`), sin secretos reales en archivos. Commits atómicos con mensajes descriptivos.
9. **Criterios de aceptación medibles.** Cada entregable debe tener un criterio verificable ("p95 ≤ 60 s", "385 tests pasan", "el test X falla si la navegación reaparece"). Si el criterio no es medible, proponlo antes de implementar.
10. **Reporte final en formato fijo** (el que el prompt concrete, típicamente: `## Hecho / ## Auditoría / ## Veredicto / ## Bloqueos / ## Falta`), incluyendo siempre la lista de lo que NO hiciste aunque el prompt lo sugiriera.
11. **Anti-bucle de admisión.** Máximo 2 ventanas de admisión fallidas por la misma causa. Si el bloqueo persiste, no abortes de nuevo: identifica la causa exacta con evidencia (p. ej. `nvidia-smi` para procesos GPU), documéntala y propón al usuario la acción concreta antes de reintentar.
12. **Acciones humanas explícitas.** Cuando necesites algo del usuario, entrégalo como checklist numerado y verificable (qué cerrar, con qué comando verificar), no como texto narrativo.
13. **Aprobaciones agrupadas.** Al inicio de la tarea, presenta UNA lista consolidada de todas las acciones que requerirán aprobación o elevación durante la ejecución, para que el usuario apruebe todo de una vez. No interrumpas a mitad de camino con aprobaciones que pudiste prever; solo detente ante lo genuinamente imprevisto.
14. **Gestión reversible de procesos del sistema.** Solo puedes detener/desactivar los procesos y servicios explícitamente autorizados en el prompt, siempre de forma reversible, y debes restaurarlos al final (o documentar por qué no fue posible). Si una acción corta el canal de acceso remoto del usuario (p. ej. AnyDesk), avísalo con la duración estimada de la ventana de desconexión antes de ejecutarla. Toda autorización de procesos es **puntual**: declara alcance exacto, supervisor de restauración y límite temporal antes de actuar; la autorización no se renueva por silencio — cada nueva ventana de intervención requiere nueva autorización explícita.
15. **Pre-registro de criterios de aceptación.** Los criterios que deciden un veredicto (umbrales, condiciones, qué cuenta como éxito) se fijan ANTES de medir y quedan escritos. Si cambian a mitad de camino, se documenta quién los cambió, por qué y con qué autorización. Prohibido mover la meta después de ver el resultado.
16. **Cambios visibles al participante con nota UX.** Todo cambio que un participante pueda ver o sentir (tiempos de espera, mensajes, relojes, avisos) debe venir con una nota UX adjunta: qué verá, en qué momento, con qué texto exacto, y por qué eso no compromete la validez del estudio.
17. **Paradojas como señal de defecto.** Si un resultado contradice lo estructuralmente esperado (p. ej. léxico más lento que híbrido, que hace estrictamente más trabajo), se trata como un defecto a explicar, no como ruido. Antes de proponer cualquier arreglo: desglose por etapa (retrieval / generación / NLI) con instrumentación y evidencia por sistema. Prohibido optimizar sin una historia causal verificada; una mejora que "funciona" sin explicar la paradoja no cierra el hallazgo.
18. **Mejora prospectiva con hipótesis falsable.** Toda optimización declara por escrito ANTES de implementar: (a) hipótesis causal, (b) mecanismo concreto del cambio, (c) tamaño de efecto esperado con justificación, (d) plan de medición (qué cohorte, qué métrica, qué umbral), (e) plan de rollback. Se valida en datos NUEVOS (piloto limpio caliente de 20 posiciones), jamás sobre los mismos datos que la sugirieron. Si el efecto observado difiere del esperado en cualquier dirección, se reporta la discrepancia, no solo el éxito/fracaso.
19. **Una variable a la vez (atribución causal).** En trabajo de optimización, cada cambio candidato se implementa y mide de forma aislada. Prohibido combinar dos o más modificaciones en una misma medición: si la latencia mejora (o empeora) no se podría atribuir. Si dos cambios son estructuralmente inseparables, se documenta por qué y se declara la hipótesis como conjunta.
20. **Datos históricos generan hipótesis, no las confirman.** Todo patrón encontrado en evidencia retrospectiva (analizadores sobre cohortes ya ejecutadas) se etiqueta como hipótesis candidata; su confirmación exige una medición prospectiva diseñada para ese fin (posiciones emparejadas, instrumentación en vivo). Prohibido usar el mismo corpus histórico para proponer y "confirmar" un mecanismo, y prohibido presentar una correlación histórica como causa establecida.
21. **Fronteras de medición definidas antes de medir.** Antes de cualquier medición queda escrito qué entra y qué no entra en el cronómetro (inicio, fin, preparación, exclusiones) y cada frontera tiene su regresión. Si durante una ventana se descubre un defecto en la frontera, se ABORTA la ventana y se corrige: prohibido parchear a mitad de medición, salvar datos parciales de una ventana con frontera defectuosa, o re-etiquetarlos como válidos. Los datos de una ventana abortada se conservan etiquetados como abortados, nunca mezclados con la cohorte válida.
22. **Mediciones largas como scripts desatendidos.** Toda medición de más de ~15 minutos de reloj se empaqueta como script idempotente y reanudable que el HUMANO lanza con un comando: pre-flight con verificación de entorno (rechaza iniciar si hay contaminación: navegadores, overlays, procesos GPU ajenos), progreso visible (intento i/N, ETA), marcado de invalidez en vivo según protocolo, restauración automática del sistema con límite duro aunque el script muera, y paquete de evidencia final con manifiesto y hashes. El agente diseña el script, lo prueba en seco (modo sintético rápido que demuestre pre-flight, reanudación, invalidez y restauración) y entrega checklist de lanzamiento; luego analiza el paquete en la sesión siguiente. Prohibido consumir la sesión del agente esperando el reloj.

## §3-bis. Cláusulas del modo autónomo (23–32). En esta corrida prevalecen sobre las cláusulas 4, 11, 12 y 13

23. **Decidir sin preguntar.** Donde la cláusula 4 dice «pregúntame», aquí documentas y decides. Registra cada decisión no trivial en `DECISIONS_LOG.md` con: contexto, opciones consideradas, opción elegida, por qué es la más conservadora para la validez, cómo revertirla y evidencia. Prefiere siempre la opción reversible y la que no mueve criterios.
24. **Paradas duras (lista cerrada).** Solo te detienes en una rama, nunca en la corrida entera, si continuar exige:
    - (a) superar el presupuesto de nube (corte propio a 45 USD acumulados);
    - (b) crear cuentas, ingresar datos de pago o credenciales, o aceptar términos nuevos;
    - (c) exponer datos de participantes, secretos o servicios internos a internet (Ollama, Streamlit sin TLS ni autenticación);
    - (d) borrar datos del usuario o evidencia;
    - (e) modificar evidencia congelada, el artículo o un criterio pre-registrado fuera del procedimiento de enmienda de §0.5 y §0.6;
    - (f) tocar NVDisplay, el driver o `.venv-app`.

    La rama queda `BLOQUEADO-HUMANO`, con la acción exacta que hace falta, y sigues con todo lo que no dependa de ella.
25. **Presupuesto de reintentos y tiempo por fase.** Cada fase declara de antemano su límite de intentos y de reloj. Si lo agota, la fase queda `BLOQUEADA` con su evidencia y pasas a la siguiente fase independiente. Prohibido reintentar lo mismo sin una causa nueva identificada (cláusula 11).
26. **Esperas eficientes.** Toda medición larga corre en segundo plano, con su propio supervisor y su límite duro. Tú la consultas con esperas largas (p. ej., `Start-Sleep` de 5 a 10 minutos entre consultas) y nunca en un bucle activo que consuma tu contexto.
27. **Estado reanudable.** Mantén `STATE.json` con la fase actual, los recursos creados (IDs de GCP), los cambios de sistema aplicados y pendientes de restaurar, y el costo acumulado. Si tu sesión se reinicia, retoma desde ese estado sin repetir efectos (cláusula 7).
28. **Ledger de costos.** Antes de cada recurso pagado, estima su costo por hora con la tarifa **oficial** consultada en la fuente de Google y regístrala. Lleva el costo acumulado en `COST_LEDGER.md`. Detén la VM siempre que no esté midiendo, haciendo un smoke o un build.
29. **Seguridad de la nube.** Desde internet solo es accesible el puerto 443 con TLS válido. Ollama, Streamlit y SSH no quedan expuestos (SSH solo por IAP o por tu IP, con regla temporal que se elimina al terminar). Hay autenticación por invitación de la app. Sin claves de servicio en archivos ni en el repositorio, y bucket privado con acceso uniforme. Los datos de sesión quedan en la región elegida.
30. **Equivalencia con lo medido.** El despliegue usa el mismo commit, los mismos pesos (digest de Ollama `444af1c4…2852` y hashes de los modelos de HF), los mismos índices (con hash) y las mismas versiones de dependencias que `requirements-lock.txt`, salvo diferencias de plataforma Linux/Windows inevitables, que van listadas. Toda diferencia de hardware o software con el entorno de exp12 se documenta como diferencia; no se afirma igualdad de salidas sin demostrarla.
31. **Paquete para el auditor.** Todo queda en `C:/CloudRAG/autonomous-run-<timestamp>/`:
    - `RUN_LOG.md`, cronológico;
    - `DECISIONS_LOG.md`;
    - `COMMANDS.log`, con cada comando, su código de salida y su duración;
    - `CLAIMS_LEDGER.md`, donde cada afirmación del reporte final apunta a su evidencia (ruta, hash y el comando exacto para re-verificarla);
    - `SYSTEM_CHANGES.md`, con cada cambio de sistema y su restauración verificada;
    - `COST_LEDGER.md`;
    - `STATE.json`;
    - un manifiesto SHA-256 de todo el directorio.

    Una afirmación sin entrada en el ledger no va en el reporte.
32. **Honestidad sobre el objetivo.** El objetivo es una compuerta GO, pero un GO vale solo si es **defendible**. Prohibido relajar umbrales, excluir intentos lentos, reetiquetar intentos inválidos, elegir tareas por cómo responde el sistema o declarar GO con datos de una ventana aislada. Un NO-GO bien diagnosticado después de 3 rondas es un resultado aceptable; un GO maquillado, no.

## §3-ter. Cláusulas de la iteración 2 (33–40), vigentes. La 33 se corrige en esta iteración

La cláusula 33 precisa la 25: donde la 25 habla de límite de reloj por fase, aquí rige la medición de la 33; el único límite de reloj es el techo de 48 horas junto con los límites duros de cada medición.

33. **Plazos por fase medibles, con techo de reloj (corregida).** El presupuesto de cada fase se mide con lo que sí puede verificarse: la unión de las duraciones de comandos y mediciones registradas en `COMMANDS.log` y en los logs de cada supervisor. Esa cifra es una cota inferior del trabajo real y así se declara; no se intenta reconstruir el tiempo de razonamiento ni se bloquea una fase por no poder demostrarlo. Los huecos sin comandos (pausas de la conversación) no consumen el presupuesto de ninguna fase y se registran con inicio, fin y duración. Siempre corren el techo de 48 horas de reloj de §0.2 y los límites duros de cada medición. Un proceso de fondo que excede su límite duro es un fallo de esa medición, no una fase vencida.
34. **Declaración previa de lo desechable.** Antes de cualquier operación que cree artefactos temporales gestionados por una herramienta (contenedores intermedios, cargas compuestas, cachés de build, archivos temporales de gcloud o pip), o los declaras desechables por escrito en `DESTRUCTION_LOG.md` antes de lanzarla, o desactivas ese comportamiento (`--rm=false`, carga compuesta desactivada por variable de entorno del comando). Lo que no se declaró no se borra a mano.
35. **Ensayo en seco de herramientas antes de gastar tiempo pagado.** Todo comando nuevo que vaya a correr en la VM (flags de Docker, parseo de salida de gcloud, formatos JSON) se valida antes en seco o contra la versión exacta instalada. Un fallo de herramienta (sintaxis, flag, parser) se clasifica como **fallo de herramienta**, distinto de un fallo del sistema medido, y cuenta para el disyuntor de tres fallos no relacionados de tu D12.
36. **El automatizador del navegador se valida antes del smoke real.** El smoke real solo arranca después de un recorrido sintético completo y aprobado con el **mismo** script que se usará en real, que interactúa como una persona (etiquetas visibles, salida de foco con Tab, espera de botones habilitados) y que verifica cada instrumento antes de avanzar. Si un smoke real falla por un defecto demostrado del automatizador, se clasifica como **defecto del automatizador**, se conserva como fallido, y se permite **un** reintento real después de corregirlo y de volver a pasar el sintético. Nunca se reetiqueta un smoke fallido como aprobado.
37. **Durabilidad POSIX demostrada.** Antes de guardar datos de sesión en Linux: escritura atómica con `fsync` del archivo, `rename` y `fsync` del directorio padre, con un test que lo verifique en el contenedor y una prueba de corte (matar el proceso entre escritura y `rename`) que demuestre que no se pierde ni se corrompe un registro confirmado.
38. **Una sola fuente de identidad.** Commit, sello, digest de Ollama, hashes de HF e índices, imagen y driver se leen de un único inventario generado por script (`environment_identity.json`) que el preflight, el runner y el reporte consumen. Prohibido teclear a mano un hash o abreviarlo como prueba de identidad.
39. **Costo en reposo explícito.** Cada recurso que siga cobrando con la VM detenida (disco, bucket, IP reservada, balanceador, certificado) entra al `COST_LEDGER.md` con su costo por día y la decisión de conservarlo o liberarlo al cierre. Ningún recurso de costo en reposo se deja creado sin esa línea.
40. **Ética del estudio.** Ninguna invitación de esta iteración llega a una persona distinta de Enzo o de tus propios scripts. Ningún dato real de participantes entra en la VM, el bucket o el paquete. Si algún paso requiriera una persona real, esa rama queda `BLOQUEADO-HUMANO`.

## §3-quater. Cláusulas nuevas de la iteración 3 (41–44)

41. **Escalera de capacidad de nube.** Ante un error de capacidad (`ZONE_RESOURCE_POOL_EXHAUSTED` o equivalente) aplicas §0.4 en orden: reintentos cada 30 minutos durante 6 horas como máximo en la zona original; después, instantánea del disco y VM nueva en otra zona de `us-central1`; después, otra región de Estados Unidos con L4, en orden de latencia medida desde Lima. Cada reintento y cada reubicación queda en `RUN_LOG.md` con el error exacto. Un error de capacidad no es un fallo de la compuerta ni habilita remedios. Antes de crear un recurso nuevo estimas su costo con la tarifa oficial (cláusula 28) y su costo en reposo (cláusula 39). Nunca hay dos VM con GPU encendidas a la vez, y toda VM nueva hereda `deletionProtection`, un disco que no se borra con la VM, STOP nativo y arranque sin trabajos terminales repetibles.
42. **Anexo de entorno antes de medir.** Si cambian la zona, la región, la VM o la imagen, commiteas antes de la primera medición un anexo (`docs/STUDY_GATE_ENVIRONMENT_ANNEX_<fecha>.md`) que liste el entorno nuevo y lo enlace con `environment_identity.json` generado en esa VM. Sin ese anexo no se mide. Ninguna ventana medida en un entorno se combina con otra de un entorno distinto.
43. **Cierre de aplicaciones del usuario.** Con la autorización de §0.6 cierras primero de forma ordenada y solo después de forma forzada, siempre con la lista previa de lo abierto en `SYSTEM_CHANGES.md`. AnyDesk se puede detener durante la cohorte y la compuerta locales; el corte de acceso remoto se declara con su duración estimada antes de hacerlo y se restaura al terminar. Todo lo que se reabre al final se verifica.
44. **Avanzar mientras haya camino.** Mientras una rama espera capacidad o una medición larga, avanzas en las demás ramas independientes. Solo cierras antes del techo si todas las ramas restantes quedaron en `BLOQUEADO-HUMANO` por una parada dura de la cláusula 24, y lo justificas rama por rama.

## §3-quinquies. Cláusulas nuevas de la iteración 4 (57–66)

57. **Congelamiento de la lógica RAG, con prueba.** Antes de tocar código generas `rag_freeze_baseline.json`, con:
    - los valores efectivos de `SURVEY_DEPLOY` y `LLM_ONLY_NO_RAG`;
    - el SHA-256 de cada plantilla de prompt;
    - las opciones enviadas a Ollama (`num_predict` 1024, temperatura 0, semilla, contexto);
    - la ruta NLI;
    - los hashes de índices y pesos;
    - los IDs de contextos recuperados para las 6 tareas en ambas condiciones, que deben ser iguales a los de la compuerta de la iteración 3.

    Al final generas `rag_freeze_final.json` con el mismo script. La única diferencia admitida es la capa de servicio declarada en la cláusula 58. Cualquier otra diferencia es un fallo de esta iteración y se reporta como tal.
58. **Estímulo fijo por la capa de servicio.**
    - El hallazgo de la auditoría es retrospectivo: es una hipótesis (cláusula 20), no una causa.
    - Diagnosticas en la VM con una medición prospectiva diseñada para eso. El registro debe probar, por ejemplo, si cambia el estado de caché de prompt o KV de Ollama según la consulta previa, o si influyen el ordenamiento de lotes y `num_parallel`.
    - Preregistras el mecanismo con hipótesis, efecto esperado, medición y rollback (cláusula 18).
    - El mecanismo se aplica igual a toda consulta y en ambas condiciones. No depende del ID de la tarea, no cambia el texto del prompt ni las opciones de generación, y no sirve respuestas grabadas.
    - Si el usuario espera mientras corre, entra en el cronómetro.

    **Aceptación:** al menos 10 repeticiones por cada una de las 12 combinaciones tarea × condición, con historiales previos variados:
    - el orden del calendario de la compuerta;
    - las cuatro celdas del cuadrado latino del estudio (orden de sistemas × conjunto de tareas);
    - primera consulta tras un arranque;
    - tras una consulta libre sintética arbitraria.

    Todo en al menos 2 arranques. Resultado exigido: 12/12 combinaciones con un solo texto, una sola clase v2 y un solo conjunto de citas. Se declara que esos textos pueden diferir de los de exp12. Si no se alcanza, la rama queda `BLOQUEADO-HUMANO` con la evidencia.
59. **Privacidad por diseño.** Con propósito `study` ningún componente guarda datos que vinculen una sesión con una persona fuera del código de participante:
    - Caddy sin `remote_ip`, `client_ip` ni User-Agent, o con la IP truncada;
    - Streamlit, la app, Docker, journald y la consola serial sin IP ni texto libre de la sesión;
    - `evidence/` sin contenido de sesiones: solo metadatos técnicos, hashes y conteos.

    **Prueba:** sesiones sintéticas más peticiones fallidas forzadas durante la ventana de arranque. Después, un `grep` de IPs públicas y User-Agents sobre el disco de la VM, el bucket nuevo y el paquete debe dar cero aciertos.
60. **Borrado verificable.** Todo dato de sesión tiene un inventario de rutas, y existen dos comandos de operador, cada uno con modo de simulación y recibo:
    - `purge-study`: descarga verificada por SHA-256, borrado del bucket de sesiones sin versiones ni copias soft-deleted, limpieza de las copias en el disco de la VM y listado final vacío, incluido `--soft-deleted`;
    - `withdraw <codigo>`: el mismo procedimiento acotado a un código.

    En propósito `study`, ambos exigen escribir el código o la palabra de confirmación. En esta iteración solo se ejecutan sobre datos sintéticos propios (A5). El bucket nuevo de sesiones se crea con soft delete en 0 y sin versionado. El bucket original queda para evidencia técnica y no recibe datos de sesión con propósito `study`.
61. **Mínimo privilegio verificable.**
    - La VM usa una SA dedicada, con roles solo sobre los recursos que necesita: escritura y lectura en el bucket de sesiones y escritura de evidencia técnica.
    - Ninguna SA que use la VM tiene roles de proyecto amplios.
    - Los scopes son mínimos.
    - El contenedor de la app no alcanza el servidor de metadatos, salvo que se demuestre que lo necesita.
    - El preflight comprueba todo esto y falla si no se cumple.
62. **El propósito `study` exige el registro ético.** El operador rechaza `study` si no existe `C:/CloudRAG/operator-iteration4/ethics/ethics_approval.json`. Ese archivo lo crea Enzo a mano, con:
    - comité;
    - código de aprobación;
    - fecha;
    - SHA-256 del PDF de la aprobación;
    - versión de B.4 aprobada.

    Tú solo pruebas con fixtures en los tests. Las invitaciones de `invite <codigo>` funcionan así:
    - el token se genera en local y solo su SHA-256 viaja a la VM;
    - se muestra una única vez en consola;
    - nunca se escribe en logs, archivos ni capturas;
    - caduca en 24 h o al primer uso completo de la sesión;
    - es revocable.
63. **Atributos de calidad con criterio medible.** El reporte trae una tabla con cada atributo, su criterio y su evidencia:

    | Atributo | Criterio |
    |---|---|
    | Disponibilidad | start → READY ≤ 15 min en cada uno de al menos 3 arranques; 3 arranques consecutivos con la IP estática sin emisión nueva de certificado |
    | Continuidad | un simulacro real de conmutación a zona alterna llega a READY con identidad verificada y la misma URL |
    | Confiabilidad | compuerta con cero fallos e inválidos; respaldo verificado por generación y SHA-256 en cada sesión sintética |
    | Recuperabilidad | una sesión sintética respaldada se restaura en una instancia nueva de la app y se exporta igual |
    | Rendimiento | p95 ≤ 60 s por condición en la compuerta nueva |
    | Seguridad | solo el 443 expuesto; XSRF activo con la app funcionando; SA mínima; cero secretos en paquete, operador y diff |
    | Privacidad | cláusulas 59 y 60 aprobadas |
    | Operabilidad | cada error del operador da un mensaje legible con la acción siguiente; runbook en español probado de forma literal (cláusula 66) |
    | Costo | ledger con estimado y márgenes por separado; proyección del periodo de sesiones dentro del techo |
64. **Contingencia de capacidad y ciclo de vida de la IP.** El operador trae estos comandos, todos idempotentes y con recibo:
    - `ip-reserve`;
    - `tls-prepare`, que emite y verifica el certificado con al menos 3 días de anticipación a la primera sesión;
    - `ip-release`;
    - `failover --zone us-central1-b|us-central1-c`, desde la imagen preparada, con identidad verificada, misma IP y misma URL;
    - `failback`.

    El runbook fija: encender 60 minutos antes; si hay `ZONE_RESOURCE_POOL_EXHAUSTED`, conmutar; si tampoco hay capacidad, reprogramar con un texto para el participante. Antes de cada recurso estimas su costo con la tarifa oficial (cláusulas 28 y 39).
65. **Prompts y preregistros anclados.**
    - Antes de cada medición que decide algo, el preregistro o la enmienda están en un commit con push al remoto, o en un objeto de GCS cuya generación y hora de servidor quedan registradas.
    - Copias este prompt en `output/audit/` del worktree y lo commiteas al inicio.
66. **Runbook humano probado de forma literal.** Al final ejecutas el runbook del operador nuevo de principio a fin, copiando cada comando tal como está escrito, como si fueras Enzo, y guardas la transcripción. Cualquier paso que exija información que no está en el runbook es un defecto que corriges antes de cerrar.

## §3-sexies. Cláusulas nuevas de la iteración 5 (67–77)

67. **Continuidad con paquete propio.** Trabajas en `C:/CloudRAG/iteration5-run-<timestamp UTC>/`, con la estructura de la cláusula 31.
    - El paquete de la iteración 4 es de solo lectura después de su sello. Si el sello no existe, solo puedes completar su cierre (§0.1).
    - Toda afirmación heredada entra a tu `CLAIMS_LEDGER.md` como DECLARADA, salvo que la vuelvas a verificar con tu propio comando.
    - No mezclas mediciones de la iteración 4 con las tuyas.
68. **Corregir lo que quedó mal y usar herramientas mínimas.** Antes de escribir código nuevo, haz un inventario de defectos de la iteración 4 a partir de sus registros y del código:
    - controles en rojo (por ejemplo, el fallo de Ruff conservado);
    - fallos de herramienta repetidos;
    - pruebas que no cubren lo que afirman;
    - tareas programadas o recursos sin dueño;
    - scripts de un solo uso fuera del repositorio.

    Cada defecto se corrige con prueba o se declara con su motivo. La lógica que se vaya a volver a usar (operador, verificadores, controladores de medición) vive versionada en el repositorio, con pruebas. Prefieres los comandos del operador y del runbook a los scripts ad hoc. Si necesitas un script puntual, va dentro de tu paquete, con propósito y recibo, y nunca reemplaza a una prueba del repositorio.
69. **Capacidad en varias regiones de EE. UU.**
    - **Orden:**
      1. `us-central1` (a, b, c), con hasta 3 rondas separadas por al menos 45 minutos;
      2. si no hay L4, las regiones de EE. UU. que la API muestre con `nvidia-l4` disponible, en orden de latencia medida desde el equipo de Enzo;
      3. si ninguna tiene capacidad, la rama queda en espera documentada y avanzas en las demás.
    - **Nunca** hay dos GPU encendidas a la vez.
    - **Si cambias de región,** antes de medir: subred, reglas de firewall equivalentes, IP estática regional, certificado, disco desde la instantánea calificada, identidad nueva y anexo de entorno commiteado con push (cláusulas 42 y 65). Las ventanas medidas en un entorno no se combinan con las de otro.
    - **Para las sesiones,** la enmienda de la compuerta fija antes de medir esta regla: el GO vale para la misma imagen, el mismo tipo de máquina y la misma identidad de software. Conmutar de zona o de región el día de una sesión exige identidad verificada y smoke propio, no una compuerta nueva, porque el cronómetro se mide dentro de la VM.
    - El bucket de sesiones sigue en `us-central1`; registras el costo de transferencia entre regiones.
70. **Limpieza de retención con inventario previo.** Antes de borrar cualquier recurso de §0.4:
    1. Registras su nombre, ID, zona o región, tamaño real, origen, fecha, etiquetas y la evidencia que lo menciona en `RETENTION_INVENTORY.json`, con hash del listado.
    2. Identificas el conjunto mínimo que reconstruye el entorno final: la instantánea de la imagen final y el archivo de la imagen en el bucket, si existe.
    3. Pruebas esa restauración: creas un disco desde la instantánea, lo montas en una VM pequeña sin GPU y verificas archivos clave e identidad. Así la prueba no depende de capacidad L4.
    4. Solo entonces borras lo redundante y verificas por la API que ya no existe.

    Lo declaras en `DESTRUCTION_LOG.md` antes de ejecutar (cláusula 34). La VM y el disco originales, los buckets y la evidencia en archivos nunca se borran.
71. **UEQ-S literal.**
    - **Fuente:** los 8 ítems se toman de forma literal de la versión en español del material oficial del UEQ (B5). Registras la fuente y el SHA-256 del archivo descargado. Prohibido traducir o redactar ítems.
    - **Formato:** orden, polaridad, diferencial semántico de 7 puntos y puntuación de −3 a +3, según el manual oficial. La calidad pragmática son los ítems 1 a 4, la hedónica los ítems 5 a 8, y la global el promedio de los 8.
    - **Momento:** se aplica justo después del SUS en cada bloque, antes de ver el otro sistema.
    - **Nota UX (cláusula 16):** texto exacto de la instrucción, momento y tiempo estimado, que debe ser de alrededor de un minuto.
    - **Sin cambios:** al SUS, a los Likert ni a las tareas.
    - **Registro:** las respuestas y puntajes van solo asociados al código del participante y entran en la exportación y en el borrado.
    - **Pruebas:** captura completa, puntuación con casos conocidos, polaridad, exportación y retiro.
72. **Plan de análisis ejecutable.** El análisis del estudio queda implementado y probado con datos sintéticos antes de cualquier sesión, en un módulo versionado con pruebas:
    - SUS de Brooke (de 0 a 100) por participante y sistema;
    - Shapiro-Wilk (α = 0,05) sobre las diferencias pareadas;
    - t pareada si son normales y Wilcoxon si no, con la otra prueba como sensibilidad;
    - d_z;
    - intervalos por bootstrap con semilla fija;
    - Benjamini-Hochberg sobre las dos escalas del UEQ-S;
    - Likert y comparación entre perfiles solo de forma descriptiva.

    Las pruebas incluyen un caso con diferencias normales y otro claramente no normal, y verifican qué prueba se elige en cada uno.
73. **Criterios decididos de privacidad y borrado.**
    - **Privacidad:** en todo registro, exportación o evidencia que generen los componentes del estudio desde esta iteración (Caddy, app, runner, journald, consola serial, respaldos, subidas a `evidence/`) hay cero identificadores de personas: IP de clientes, User-Agent, tokens, nombres, correos y textos de sesión fuera del registro asociado al código. Las direcciones de la propia infraestructura (IP de la VM, URL del servicio, IDs de recursos) se permiten en la evidencia técnica. La evidencia heredada no se reescribe: se inventaría aparte.
    - **Prueba de privacidad:** sesiones sintéticas y peticiones fallidas forzadas desde una IP de cliente conocida, seguidas de una búsqueda de esa IP y de ese User-Agent en todo lo nuevo. Resultado exigido: cero aciertos.
    - **Borrado:** el criterio de listado `--soft-deleted` vacío se reemplaza por cuatro comprobaciones:
      - política de soft delete con retención 0 y sin versionado, verificadas en los metadatos del bucket;
      - historial de Cloud Audit Logs sin cambios de esa política desde su creación;
      - listado normal de objetos vacío para el código o el periodo borrado;
      - recibo de la descarga verificada por SHA-256 antes de borrar.
74. **Reanudación ante cuota o compactación.**
    - Cada fase termina en un estado consistente: `STATE.json` con la fase, los recursos vivos con su ID y costo, las tareas programadas propias y la siguiente acción exacta.
    - Antes de cada efecto pagado registras la intención; después, el resultado.
    - Una sesión nueva retoma leyendo `STATE.json` y verificando en vivo los recursos antes de actuar; nunca repite un efecto sin comprobar que no ocurrió.
    - Si te quedas sin cuota con una GPU encendida, la protege el apagado nativo y el supervisor independiente que registras antes de encender (cláusulas 22 y 28).
75. **Hooks locales (solo Claude Code).** Una denegación de GateGuard («[Fact-Forcing Gate]») se responde con los datos que pide y se reintenta. No cuenta como fallo de herramienta ni de medición, y no se desactiva el hook. Si el hook bloquea un paso automatizado que corre fuera de la conversación, documentas el caso y pides a Enzo la exención `GATEGUARD_EXEMPT_GLOBS` en el checklist humano.
76. **Independencia y no autoevaluación.** Tu reporte dice qué se midió y qué no, con evidencia, pero no declara el despliegue «apto para participantes». Ese dictamen corresponde a la auditoría de cierre, hecha por un agente que no haya implementado nada en esta iteración, y a la aprobación ética. Si ambos agentes participaron por relevo, la auditoría la hace una sesión nueva sin acceso a sus conversaciones, que solo verifica con evidencia. Prohibido llamar GO a un criterio no medido o medido en otro entorno.
77. **Relevo entre agentes.** Solo un agente trabaja a la vez. Al empezar creas `C:/CloudRAG/iteration5.lock` con agente, modelo, hora UTC y PID. Si el archivo existe y no lo creaste tú, verificas que el agente anterior ya no está activo (sin actualizaciones de `STATE.json` en 30 minutos y sin procesos propios vivos) antes de tomarlo, y lo registras. Al detenerte por cuota o por cierre de fase escribes en `HANDOVER.md`: fase actual, recursos vivos con ID y costo, tareas programadas propias, mediciones en curso y la siguiente acción exacta. El agente entrante no repite efectos sin verificarlos en vivo, y conserva las decisiones y la numeración del saliente. Cambiar de agente no reinicia el techo de 72 horas ni el presupuesto.

## §4. Plan por fases (criterios escritos antes de ejecutar; tiempo según la cláusula 33)

### Fase 0 · Cierre heredado y arranque (≤ 2 h)
1. Comprueba la hora. Antes de las 17:15 de Lima del 06/10 o mientras alguna tarea `CloudRAG-I4-*` esté en ejecución o pendiente, solo lees.
2. Verifica el cierre de la iteración 4:
   - estado y último resultado de cada tarea `CloudRAG-I4-*`;
   - reporte final y sello externo;
   - VM TERMINATED, IP liberada y reglas IAP retiradas, con lectura de GCP;
   - costo registrado.
3. Si algo falló, completas el cierre según su propio plan, con registro.
4. Crea tu paquete, `STATE.json` y `COMMANDS.log`.
5. Registra el baseline: `git status`, `git log`, suite Windows, Ruff, `git diff --check`, secretos e inventario completo de GCP.
6. Copia este prompt a `output/audit/` y commitéalo con push. Crea el lock y `HANDOVER.md` (cláusula 77).
7. Presenta la lista consolidada de efectos pagados y destructivos previstos, con su costo (cláusula 13).

### Fase 1 · Limpieza y proyección (≤ 3 h)
1. Aplica la cláusula 70: inventario, prueba de restauración con CPU y borrado de lo redundante.
2. Recalcula el costo en reposo y la proyección del periodo de sesiones con espera de 0, 30, 60 y 90 días, frente al corte de 90.
3. Ajusta las alertas del proyecto.

**Criterio:** costo en reposo ≤ USD 0,45 por día y proyección de 30 días por debajo del corte. Si no se cumple, lo informas con su motivo.

### Fase 2 · Código (≤ 10 h)
1. Inventario y corrección de defectos (cláusula 68), incluido Ruff en verde.
2. UEQ-S (cláusula 71) y plan de análisis (cláusula 72), con pruebas.
3. Disparo automático del respaldo al cerrar sesión, con prueba.
4. Integra el candidato de usuario de runtime `4f43cfc` solo si su prueba pareada pasó.
5. Verifica `rag_freeze` estático contra el baseline de la iteración 4 (cláusula 57).

**Criterio:** suites Windows y Linux en verde, Ruff sin hallazgos nuevos, secretos limpios y congelamiento estático igual.

### Fase 3 · Imagen y entorno (≤ 6 h; hasta 3 builds)
1. Build final que incluye el UEQ-S.
2. `pip check` y las 7 pruebas POSIX.
3. Capacidad según la cláusula 69.
4. Instala el operador iteración 5.
5. Genera la identidad nueva y commitea el anexo de entorno con push antes de medir.

### Fase 4 · Aceptación (≤ 10 h)
1. Congelamiento final con los 12 contextos vivos, que deben ser iguales a los de la compuerta de la iteración 3 (cláusula 57).
2. Aceptación del estímulo (cláusula 58): 12/12 con historiales variados y 2 arranques.
3. Privacidad (cláusula 73).
4. Purga y retiro sintéticos con el criterio de la cláusula 73.
5. Restauración de un respaldo.
6. TLS: 3 arranques sin emisión nueva de certificado.
7. Simulacro de conmutación y vuelta (cláusula 64), con zona o región alterna según la disponibilidad.
8. Smoke real completo con el automatizador validado, que ahora incluye el UEQ-S (cláusula 36).

**Criterio:** la tabla de la cláusula 63 completa, salvo Rendimiento.

### Fase 5 · Compuerta nueva (sobre la imagen final)
1. Enmienda preregistrada con push, que incluye la regla de entornos de la cláusula 69.
2. Piloto nuevo de 20 posiciones.
3. Dos ventanas de 60 con la identidad única de la Fase 3, umbral de p95 ≤ 60 s por condición, cero fallos e inválidos, y decisión solo con el agregado.
4. Si es NO-GO, no hay remedios que toquen la lógica: la rama queda `BLOQUEADO-HUMANO` con el diagnóstico por etapa.

### Fase 6 · Cierre (≤ 3 h)
1. Runbook literal de principio a fin (cláusula 66).
2. Libera las IP estáticas, salvo que Enzo haya agendado una sesión.
3. Comprueba que todas las VM estén TERMINATED, que no queden recursos de prueba sin declarar y que tus tareas programadas estén retiradas.
4. Ledger final con la conciliación de la factura si la consola ya la muestra; si no, DECLARADO.
5. Push.
6. `DATOS_B4_V6.md` con el UEQ-S, la región y el país de almacenamiento, los registros, los plazos y el borrado.
7. Paquete del auditor con manifiesto SHA-256 y sello externo.

## §5. Reporte final (obligatorio; si falta una sección, el reporte está incompleto)

## Resumen ejecutivo         (7 líneas: cierre de la iteración 4, limpieza y costo, código y UEQ-S, capacidad y región, aceptación, compuerta, pendientes)
## Cierre de la iteración 4   (estado de cada tarea programada, sello, recursos y lo que completaste)
## Auditoría                 (commits, suites, Ruff, secretos, diff --check, manifiesto)
## Defectos heredados        (inventario de la cláusula 68: corregido o declarado)
## Retención y costos        (inventario, prueba de restauración, borrados, costo en reposo antes y después, proyección, factura)
## Congelamiento RAG         (estático y con los 12 contextos vivos)
## UEQ-S y plan de análisis  (fuente literal y SHA, nota UX, pruebas, salida con datos sintéticos)
## Estímulo                  (aceptación 12/12)
## Privacidad y borrado      (cláusula 73, matriz de promesas a–i de B.4)
## Capacidad y entorno       (intentos por zona y región, región final, anexo e identidad)
## TLS y continuidad         (tres arranques, conmutación y vuelta)
## Smoke y compuerta         (piloto, ventanas, p50/p95 por condición, fallos, inválidos)
## Atributos de calidad      (tabla de la cláusula 63 ampliada)
## Plazos y tiempo           (cotas inferiores por fase, pausas de cuota, uso del techo de 72 h)
## Relevos                   (qué agente hizo cada fase; cada relevo con su HANDOVER)
## Decisiones autónomas
## Matriz de riesgos residuales
## Checklist humano          (registro ético, B.4 V6, fecha de sesiones, IP y tls-prepare, exención de GateGuard si aplica)
## Cambios documentales pendientes (para Claude Cowork: B.4 V6, 4.8 y Declaraciones)
## Veredicto                 (por fase y por atributo; sin dictamen de aptitud, cláusula 76)
## Bloqueos
## Falta                     (incluye lo que NO hiciste aunque el prompt lo sugiriera)