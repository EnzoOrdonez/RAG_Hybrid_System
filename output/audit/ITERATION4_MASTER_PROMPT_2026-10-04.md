# PROMPT MAESTRO PARA CODEX · ITERACIÓN 4 · remediar el despliegue para que quede apto para sesiones con participantes, sin tocar la lógica del RAG
# Fecha: 2026-10-04 | Rama: fix/interview-readiness (worktree .worktrees/interview-readiness) | HEAD esperado: 7f10d7a
# Modo: AUTÓNOMO DE PRINCIPIO A FIN, sin privilegios de administrador. Al terminar, una auditoría de cierre independiente verificará cada criterio de este prompt.
# Pegar TODO este contenido a Codex tal cual.

---

## §0. Contexto y decisiones de Enzo (2026-10-04, noche), definitivas para esta iteración

La iteración 3 dejó un GO técnico de latencia (R1, auxiliares en la L4). Una auditoría independiente lo revisó después: Claude Code, con el paquete `C:/CloudRAG/audit-iteration3-20261004T204327Z/`, manifiesto `164497e927096b0181eab400bef6ba8a648c34cf3d66521c9aa2d1703d2b285a`.

**Lo que la auditoría sostuvo:**
- el sello (45 075/45 075 hashes);
- el recálculo exacto de los 240 intentos;
- los 12 pares R1 idénticos;
- los preregistros anclados antes de medir;
- la ausencia de parada opcional sesgada;
- la ruta `per_claim`;
- el firewall con solo el 443 abierto;
- TLS 1.3;
- un operador que funciona de punta a punta.

**Su dictamen global:** NO APTO para la primera sesión con participantes hasta cerrar 9 hallazgos ALTO (tabla de §1.2). Se aceptan como base, con las dos correcciones de §1.1.

Enzo quiere que esta fase del despliegue quede bien hecha en todos sus atributos de calidad, para que no haya problemas con los participantes en las sesiones en vivo. Un MVP no justifica fallas conocidas.

Decisiones cerradas de Enzo; no las reabras:

1. **Objetivo:** cerrar los 9 hallazgos ALTO y los MEDIO y BAJO que §4 asigna a esta iteración. Cada atributo de calidad debe cumplir su criterio medible (cláusula 63).
2. **La lógica del RAG queda congelada** (cláusula 57). No cambian:
   - recuperación, fusión, reordenamiento;
   - plantillas y ruteo de prompts, balanceo entre proveedores;
   - modelo, `num_predict` (1024), temperatura, NLI;
   - índices, corpus, pesos.

   La razón es metodológica: el estudio con usuarios debe evaluar el mismo procedimiento que mide el artículo, con el único ajuste ya declarado en su sección 4.8. Las mejoras de lógica quedan para después del estudio con usuarios, no antes.

   Solo se puede cambiar:
   - la capa de servicio (cómo se invoca a Ollama y cómo se maneja su estado entre consultas);
   - la infraestructura;
   - el operador;
   - los registros;
   - IAM;
   - el borrado;
   - los textos de la app que §4 indica.
3. **Estímulo (A-ALTO-01): se arregla en vivo.** Diagnosticas la causa de que la respuesta dependa de la consulta anterior. Después implementas en la capa de servicio un mecanismo que vuelva la respuesta independiente del historial, con hipótesis preregistrada (cláusulas 17, 18, 20 y 58). Prohibido servir respuestas grabadas o una tabla de respuestas por tarea. Si ningún mecanismo cumple el criterio de la cláusula 58, la rama queda `BLOQUEADO-HUMANO` con la evidencia; no se declara la variación por cuenta propia.
4. **Compuerta nueva completa sobre la imagen final.** Enmienda preregistrada, smoke, piloto nuevo de 20 y dos ventanas de 60 intentos. El umbral es p95 ≤ 60 s por condición, con cero fallos e inválidos y decisión solo con el agregado. El GO de R1 no cubre una imagen con otra capa de servicio.
5. **Presupuesto de nube: 100 USD acumulados**, con corte propio a 90 (reemplaza los 50 y 45 anteriores).
   - Punto de partida conservador: USD 9,02 estimados, de los cuales unos 3,0 son márgenes y no gasto.
   - La factura real de octubre conciliada por Claude (§1.1) cuadra con las horas de VM.
   - Tu ledger separa el costo estimado de los márgenes.
   - La moneda de la cuenta de facturación es PEN.
6. **TLS (A-ALTO-07): IP externa estática regional en `us-central1`.**
   - El certificado se emite una sola vez por IP y se conserva en almacenamiento persistente; las renovaciones las hace Caddy.
   - No se crea ninguna cuenta nueva con otro emisor: ZeroSSL exige cuenta, y eso es una parada 24b.
   - Enzo aún no conoce la fecha de la primera sesión. Por eso reservas la IP para probar el mecanismo (cláusula 63) y la liberas al cerrar, salvo que una sesión ya esté agendada.
   - El operador debe traer los comandos para reservarla y preparar el certificado con días de anticipación, y el runbook debe indicar cuándo hacerlo.
7. **Capacidad (A-ALTO-08): zona alterna lista.**
   - En los días de sesión, encendido 60 minutos antes.
   - Si `us-central1-a` no tiene L4, el operador levanta una VM equivalente en `us-central1-b` o `us-central1-c`, desde una imagen o instantánea preparada en esta iteración: misma máquina, misma IP estática regional e identidad verificada.
   - Si ninguna zona tiene capacidad, aplica el protocolo de reprogramación escrito.
   - Nunca hay dos VM con GPU encendidas a la vez.
   - El simulacro de conmutación se ejecuta una vez en real.
8. **B.4: Claude preparará la V6 después de esta iteración.**
   - Tú implementas los mecanismos para que las promesas de B.4 V5 (§1.1) sean ciertas.
   - Entregas `DATOS_B4_V6.md` con lo que realmente quedó: dónde se guarda cada dato, cuánto tiempo, cómo se borra, qué registra cada componente y qué no.
   - Las tres comparativas cerradas, la pregunta abierta sobre la diferencia y la pregunta de cegamiento no se cambian en la app; se describirán en la V6.
   - El país de almacenamiento (EE. UU.) lo declarará Enzo en la V6.
9. **Propósito `study`.** El operador lo admite solo si existe un registro de aprobación ética creado a mano por Enzo (cláusula 62). Tú nunca creas ese registro: lo pruebas con fixtures.
10. **Ética:** sin participantes ni invitaciones reales. Todo dato de prueba es sintético. Un GO no autoriza reclutar.
11. **Documentos de Enzo: no los toques.** Dejas la lista de cambios documentales pendientes.
12. **Sin privilegios de administrador y sin cerrar aplicaciones del usuario.** Esta iteración no mide nada en el equipo local. Si una medición local se volviera imprescindible, esa rama queda `BLOQUEADO-HUMANO`.
13. **Plazo: 48 horas de reloj** desde tu primer comando, con presupuestos por fase según la cláusula 33.
14. **El operador de la iteración 3 queda congelado** en `C:/CloudRAG/operator-iteration3`. Instalas uno nuevo en `C:/CloudRAG/operator-iteration4`.

## §1. Estado heredado

### 1.1 VERIFICADO

**Por la auditoría independiente** (rutas relativas a su paquete):
- **Compuerta R1:** p95 híbrido 31,304683 s y sin RAG 22,549157 s; baseline 113,366841 s y 22,436311 s. Censo de 240 intentos, sin fallos ni inválidos (`results/f2-gate-census.json`).
- **Estímulo:** en 10 de 12 combinaciones tarea × condición hay 2 textos, separados de forma perfecta por la consulta inmediatamente anterior. Cambia la clase v2 en `q016|no_rag` y cambian las citas en `q172|hybrid`. Los contextos recuperados son idénticos. Dentro de una misma secuencia es reproducible entre arranques y GPU físicas (8/8) (`results/f2-stimulus-variants.json`, `results/f5-smoke-vs-codex.json`).
- **Operador:** READY en 650,1 s; `stop` en 361,7 s. Histórico de start a READY: 5,1 a 10,8 min.
- **IAM:**
  - owner, solo Enzo;
  - la VM usa `103950017681-compute@developer.gserviceaccount.com`, con `roles/editor` y scope `cloud-platform`;
  - 0 claves de SA de usuario;
  - los contenedores usan red host y alcanzan el servidor de metadatos.
- **Bucket `cloudrag-study-103950017681-20261002`:** `us-central1`, UBLA y PAP, soft delete de 7 días, sin versionado ni ciclo de vida. Cada sesión deja copias en el prefijo de sesiones, en `iteration3/<job>/evidence/` (el invitado sube todo `deployment_root` al apagar) y en el disco de la VM.
- **Caddy:** el log acumula 11 arranques, con 99 entradas `http.log.error` que contienen `remote_ip`, `client_ip`, User-Agent, URI y encabezados. Incluyen escáneres atraídos por la transparencia de certificados y la IP del equipo del operador. Se sube a `evidence/`.
- **Invitaciones:**
  - `secrets.token_urlsafe(32)`, guardadas solo como SHA-256 e ingresadas en un campo de contraseña;
  - sin TTL;
  - la operación `invite` de `cloud_entrypoint.py` solo es alcanzable con `docker exec` por SSH;
  - `session_storage.release()` revoca sin borrar.
- **Streamlit:** `enableXsrfProtection=false`, `enableCORS=false`, `gatherUsageStats=false`.
- **Consigna de la consulta libre (`study.json`):** «… No incluyas información confidencial de tu empresa.» No menciona datos personales.
- **Red `default`:** `default-allow-ssh` y `default-allow-rdp` abiertas a 0.0.0.0/0. La VM está en `cloudrag-study-20261002`, que solo abre el 443.
- **Presupuesto:** «Presupuesto seminario 50 USD», PEN 180, a nivel de cuenta, sin filtro de proyecto.
- **Preflight:** durante el arranque falla con un traceback `StopIteration` (`iteration3_operator.py:128`).
- **Push:** GitHub recibió un único push con los 10 commits; los anexos `8a0b24b` y `7ad7aed` no tienen ancla externa.

**Por Claude** (2026-10-04), con dos correcciones a la auditoría:
- **A-MEDIO-07 era casi entero un falso positivo.**
  - La auditoría comparó contra `docs/Paper_IEEE_RAG_Hibrido_LACCI_v9.tex` (LACCI, Llama 3.1), no contra el artículo de la tesis A.3 V2.16.
  - El ruteo de prompts por tipo de consulta es el mismo del experimento principal. `scripts/run_generation_matrix.py` tipa cada consulta con `QueryProcessor` y usa `get_template(query_type)`.
  - A.3 V2.16 §4.8 ya declara el balance de cobertura por proveedor como único ajuste de despliegue.
  - Conclusión: el despliegue no se cambia por este hallazgo.
- **A-MEDIO-06 no es una desviación.**
  - En `experiments/results/exp12_matrix/checkpoint__granite4.1-8b__hibrido.json`, q070 ya tenía `tokens.output = 1024` y q068 967. El tope de 1024 es el mismo del experimento principal.
  - No se cambia; se documenta.
- **Factura de octubre:** S/ 9,073294 en total (cómputo 7,351244; disco 1,615777; Network Analyzer, Topology y Performance Dashboards 0,102710; Interconnect 0,003563).
  - Cuadra con las 3,0485 h de VM encendida hasta el 2026-10-04 05:22Z, a la tarifa de USD 0,711832/h y un cambio implícito de 3,39 PEN/USD.
  - Y con unos 1,44 días de disco.
  - Es decir, la factura todavía no incluye lo posterior al 2026-10-04 en la mañana (UTC).
  - Gasto real estimado del proyecto: unos USD 6,3 más la retención desde entonces.
  - Network Intelligence Center cobra centavos que ningún ledger registraba.
- **Promesas de B.4 V5** (texto literal del documento entregado, sección de confidencialidad):
  - (a) solo el investigador y su asesor tendrán acceso a los datos;
  - (b) a cada participante se le asigna un código, y la lista que vincula código y nombre se custodia por separado;
  - (c) consultas, respuestas, tiempos de respuesta y cuestionarios se guardan asociados solo a ese código en el servidor de Google Cloud durante el periodo de sesiones, y al terminarlo se descargan a un almacenamiento del investigador y **se eliminan del servidor**;
  - (d) los resultados se reportan agregados, y los datos anonimizados y agregados podrán publicarse en el repositorio público;
  - (e) el participante usa los sistemas desde su navegador, mediante un enlace que se le entrega al iniciar, sin instalar programas ni dar acceso a su equipo;
  - (f) la consulta libre se pide preferentemente en inglés, sin datos personales y sin información confidencial de su empleador;
  - (g) una consulta de familiarización, tres tareas y una consulta libre por sistema, SUS de diez ítems y preguntas Likert, tres preguntas comparativas y una entrevista breve, en 45 a 60 minutos;
  - (h) tratamiento conforme a la Ley 29733;
  - (i) el participante puede pedir que se eliminen sus datos mientras sigan vinculados a su código, es decir, hasta el cierre del análisis.

### 1.2 Hallazgos que esta iteración debe cerrar o documentar

| ID | Severidad | Hallazgo | Destino en esta iteración |
|---|---|---|---|
| A-ALTO-01 | ALTO | Estímulo dependiente de la consulta previa | Fase 1 (cláusula 58) |
| A-ALTO-02 | ALTO | SA por defecto con `roles/editor` | Fase 2 |
| A-ALTO-03 | ALTO | Sin mecanismo de borrado del servidor | Fase 2 |
| A-ALTO-04 | ALTO | Sin retiro de un participante a pedido | Fase 2 |
| A-ALTO-05 | ALTO | Consigna sin «datos personales» | Fase 2 |
| A-ALTO-06 | ALTO | Sin invitaciones reales en el operador | Fase 3 |
| A-ALTO-07 | ALTO | TLS dependiente de una emisión por arranque | Fase 3 |
| A-ALTO-08 | ALTO | Sin contingencia de capacidad | Fase 3 |
| A-ALTO-09 | ALTO | Caddy registra IP y UA y los sube a `evidence/` | Fase 2 |
| A-MEDIO-01 | MEDIO | Preregistros sin ancla externa | Cláusula 65 |
| A-MEDIO-02 | MEDIO | Comparativas frente a B.4 | `DATOS_B4_V6.md` (sin cambio en la app) |
| A-MEDIO-03 | MEDIO | Sin anonimización; datos en EE. UU. | Fase 2 (script de exportación) y `DATOS_B4_V6.md` |
| A-MEDIO-04 | MEDIO | Sin guarda de tiempo restante | Fase 3 |
| A-MEDIO-05 | MEDIO | Presupuesto y retención | Fase 3 (proyección con techo 100) |
| A-MEDIO-06 | MEDIO | `tokens_out=1024` | Solo documentar (§1.1) |
| A-MEDIO-07 | MEDIO | SURVEY_DEPLOY frente al artículo | Cerrado por §1.1; sin cambio |
| A-BAJO-01 | BAJO | XSRF desactivado; invitaciones sin TTL | Fase 2 |
| A-BAJO-02 | BAJO | SSH y RDP abiertos en la red `default` | Fase 2: deshabilitar las reglas, no borrarlas |
| A-BAJO-03 | BAJO | Paquetes ajenos en la imagen | **No se hace en esta iteración** (cambiaría el entorno sin beneficio para la sesión); se documenta |
| A-BAJO-04 | BAJO | Alerta de presupuesto de cuenta | Fase 3 |
| A-BAJO-05 | BAJO | `rerank_score` no persistido | Fase 1, solo en el registro del runner, sin efecto en la respuesta |
| A-BAJO-06 | BAJO | Traceback en preflight | Fase 3 |
| Nuevo (Claude) | BAJO | Cargos de Network Intelligence Center fuera del ledger | Fase 3: cuantificar y desactivar las APIs si no se usan |

## §2. Autorizaciones de esta iteración

- **A1. Código, pruebas y documentación** del worktree, con commits atómicos, dentro de lo permitido por la cláusula 57.
- **A2. Google Cloud** sobre `pure-loop-474323-a8`, dentro de §0.5:
  - encender y detener la VM original;
  - crear una cuenta de servicio dedicada y sus bindings;
  - quitar `roles/editor` a la SA por defecto, registrando el estado previo para revertirlo;
  - crear un bucket nuevo y privado para sesiones en `us-central1` (UBLA, PAP, sin versionado, soft delete en 0);
  - reservar y liberar una IP externa estática regional;
  - crear una imagen o instantánea del disco y una VM de prueba en otra zona de `us-central1` (nunca dos GPU encendidas);
  - deshabilitar `default-allow-ssh` y `default-allow-rdp`;
  - presupuestos y alertas del proyecto;
  - reglas IAP temporales;
  - desactivar APIs de Network Intelligence Center que no se usen.

  Prohibido:
  - borrar la VM original, su disco, el bucket original o evidencia;
  - reservar capacidad;
  - crear recursos fuera de `us-central1`.
- **A3. Mediciones reales en la nube:** diagnóstico y aceptación del estímulo, ensayos de §4, smoke, piloto y compuerta.
- **A4. Let's Encrypt:** sigue vigente la aceptación de Enzo para la cuenta ACME de este despliegue (contacto `20221789@aloe.ulima.edu.pe`), incluida la emisión para el nombre de la IP estática.
- **A5. Borrado sintético.** Puedes ejecutar los comandos de purga y retiro solo sobre datos sintéticos creados por ti en esta iteración, en el bucket nuevo de sesiones y en la VM. También puedes borrar los recursos de prueba que creaste en esta iteración y declaraste desechables antes de crearlos (cláusula 34), como la VM de prueba en la zona alterna. Nada más.
- **A6. Push** de `fix/interview-readiness` sin force ni merge, nunca a `main`. Se permiten pushes intermedios para anclar preregistros (cláusula 65).
- **A7. Instalación** del operador nuevo en `C:/CloudRAG/operator-iteration4`. Todo lo de iteraciones anteriores, el operador de la iteración 3 y el paquete de auditoría quedan en solo lectura.
- **No se autoriza:** administrador, cierre de aplicaciones de Enzo, cambios en la lógica RAG (cláusula 57), creación del registro de aprobación ética, invitaciones a personas, ni editar documentos de Enzo.

## Cómo rigen las cláusulas en esta iteración

Las cláusulas 1 a 44 se copian sin cambios de la iteración 3, con cuatro precisiones:

- **Autorizaciones:** las referencias a §0.4, §0.5, §0.6 y §0.7 se leen como las autorizaciones de §0 y §2 de este prompt.
- **Cláusula 24a:** el corte es 90 USD.
- **Cláusula 41:** la escalera de capacidad se reemplaza por la contingencia de §0.7 y la cláusula 64.
- **Cláusula 43:** no aplica en esta iteración, porque no se cierran aplicaciones.

Las cláusulas 45 a 56 pertenecen al prompt de auditoría y no rigen aquí, salvo las que las cláusulas 57 a 66 incorporan. Ante cualquier conflicto prevalecen las cláusulas 57 a 66 y las decisiones de §0.

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

## §4. Plan por fases (criterios escritos antes de ejecutar; tiempo según la cláusula 33)

### Fase 0 · Arranque (≤ 1 h)
1. Registra:
   - el baseline: `git status`, `git log`, suite Windows, Ruff, `git diff --check` y escaneo de secretos;
   - el estado de la VM y del bucket;
   - el presupuesto.
2. Lee el paquete de auditoría y su manifiesto.
3. Copia y commitea este prompt (cláusula 65) y genera `rag_freeze_baseline.json` (cláusula 57).
4. Presenta una lista consolidada de efectos pagados y destructivos previstos (cláusula 13), con su costo estimado.

### Fase 1 · Estímulo (≤ 8 h de comandos)
1. Diagnóstico prospectivo en la VM y preregistro con push (cláusulas 58 y 65).
2. Implementación en la capa de servicio, con tests.
3. Persistir `rerank_score` en el registro del runner (A-BAJO-05), sin cambiar la respuesta.

**Criterio:** la aceptación de la cláusula 58 se mide en la Fase 5 con la imagen final.

### Fase 2 · Privacidad, IAM y borrado (≤ 8 h)
1. Crea la SA dedicada, quita `roles/editor` a la SA por defecto y bloquea el acceso a metadatos (cláusula 61).
2. Crea el bucket nuevo de sesiones (cláusula 60).
3. Configura los registros de Caddy y la app según la cláusula 59 y excluye el contenido de sesiones de `evidence/`.
4. Implementa `purge-study` y `withdraw` (cláusula 60).
5. Crea un script de exportación anonimizada por código. La consulta libre queda marcada para revisión manual, porque puede contener datos personales aunque la consigna los prohíba.
6. Cambia la consigna de la consulta libre a un texto que pida no incluir datos personales ni información confidencial del empleador, con nota UX (cláusula 16). Es el único texto visible que cambia.
7. Activa XSRF y prueba la app detrás de Caddy.
8. Agrega TTL a las invitaciones.
9. Deshabilita `default-allow-ssh` y `default-allow-rdp`.

**Criterio:** tests de cada punto y verificación por lectura de GCP.

### Fase 3 · Operador iteración 4 (≤ 8 h)
1. Comandos `invite`, propósito `study` con registro ético (cláusula 62) y los comandos de la cláusula 64.
2. Guarda de tiempo restante: no se admite una sesión con menos de 70 min de margen del invitado, con tests para 69 y 71.
3. Mensajes legibles en el preflight.
4. Imagen o instantánea para la zona alterna, con su costo en reposo.
5. Presupuesto del proyecto en PEN, equivalente a USD 90 con el tipo de cambio registrado y umbrales del 50, 75, 90 y 100 %.
6. Cuantificar los cargos de Network Intelligence Center y desactivar las APIs que no se usen.
7. Proyección de costo:
   - retención por día;
   - IP estática reservada desde 3 días antes de la primera sesión;
   - piloto más 20 sesiones;
   - escenario de espera de 30, 60 y 90 días.
8. README y runbook en español.

### Fase 4 · Imagen e identidad (≤ 3 h; máximo 3 builds)
1. Build de la imagen final.
2. `pip check`, suites Windows y Linux, y las 7 pruebas POSIX sobre el disco persistente.
3. `environment_identity.json` nuevo.
4. Anexo de entorno commiteado con push antes de medir (cláusulas 42 y 65).
5. `rag_freeze_final.json` (cláusula 57).

### Fase 5 · Ensayos de aceptación (≤ 6 h)
1. Aceptación del estímulo (cláusula 58).
2. Tres arranques con la IP estática sin emisión nueva de certificado.
3. Simulacro real de conmutación y vuelta.
4. Purga y retiro sobre datos sintéticos, con recibos.
5. Restauración de un respaldo.
6. Prueba de privacidad de registros (cláusula 59).
7. Verificación de puertos y TLS.
8. Smoke real con el automatizador validado (cláusula 36).

**Criterio:** la tabla de la cláusula 63 completa, salvo Rendimiento.

### Fase 6 · Compuerta nueva
1. Enmienda preregistrada con push.
2. Piloto nuevo de 20 posiciones.
3. Dos ventanas de 60 intentos con la identidad única de la Fase 4, con el mismo umbral y las mismas reglas de la iteración 3 (cláusulas 15, 21 y 32).
4. Si es NO-GO, no hay remedios automáticos: la rama queda `BLOQUEADO-HUMANO` con el diagnóstico por etapa, porque cualquier remedio que no sea de infraestructura tocaría la lógica congelada.

### Fase 7 · Cierre (≤ 2 h)
1. Ejecuta el runbook de forma literal (cláusula 66).
2. Libera la IP estática, salvo que Enzo haya agendado una sesión.
3. Verifica la VM `TERMINATED`, el disco retenido y que no queden recursos de prueba sin declarar.
4. Ledger final.
5. Push.
6. Arma el paquete del auditor (cláusula 31) y `DATOS_B4_V6.md` (§0.8).

## §5. Reporte final (obligatorio; si falta una sección, el reporte está incompleto)

## Resumen ejecutivo         (6 líneas: estímulo, privacidad y borrado, seguridad, TLS y capacidad, compuerta nueva, costo)
## Auditoría                 (commits, suites, secretos, Ruff, diff --check, manifiesto con SHA-256)
## Congelamiento RAG         (rag_freeze_baseline frente a rag_freeze_final; única diferencia admitida)
## Estímulo                  (diagnóstico, preregistro, mecanismo, aceptación 12/12 con historiales y arranques)
## Privacidad y B.4          (matriz de promesas a–i: mecanismo, prueba y estado; resultado de la cláusula 59)
## Borrado y retiro          (inventario de rutas, recibos de purga y retiro sintéticos)
## IAM y seguridad           (SA, bindings antes y después, metadatos, XSRF, firewall, secretos)
## Invitaciones y propósito study
## TLS, IP estática y capacidad (tres arranques, simulacro de conmutación, costos)
## Compuerta nueva           (piloto, ventanas, p50/p95 por condición, fallos, inválidos)
## Atributos de calidad      (tabla de la cláusula 63)
## Plazos y tiempo           (cotas inferiores por fase, huecos, uso del techo de 48 h)
## Decisiones autónomas      (resumen de DECISIONS_LOG con enlaces)
## Costos                    (estimado y márgenes por separado, en reposo por recurso, proyección del periodo de sesiones)
## Matriz de riesgos residuales
## Checklist humano          (lo que Enzo debe hacer antes de la primera sesión, incluidos la aprobación ética, el registro ético, la IP y tls-prepare)
## Cambios documentales pendientes (para Enzo y Claude: B.4 V6, 4.8, Declaraciones; no aplicados)
## Veredicto                 (por fase y por atributo)
## Bloqueos
## Falta                     (incluye lo que NO hiciste aunque el prompt lo sugiriera)