# Notas UX — protocolo de dos condiciones (2026-09-27)

Implementación autorizada; validación con personas pendiente. Se conserva el cegamiento
de etiqueta, no funcional: las fuentes pueden revelar la condición. No se afirma que
la interfaz elimine ese sesgo; el chequeo al cierre lo mide.

| Momento | Texto/presentación exacta | Motivo y límite |
|---|---|---|
| Entrada | «Sesión», «Invitación», «Entrar» | Sin nombre técnico ni navegación del operador. No solicita datos personales. |
| Configuración incompleta | «La sesión necesita preparación. Contacta al coordinador.» | No admite sesiones con SUS vacío o asignación inválida. El diagnóstico detallado pertenece al operador. |
| Antes de consultas | «El coordinador debe preparar la sesión antes de continuar.» / «Preparar sesión» | Calienta ambos sistemas por sesión; no depende de la familiarización. Misma acción para ambas etiquetas. |
| Bloques | «Sistema A» / «Sistema B» | Mapeo global congelado. Sin valoraciones por consulta. |
| Familiarización | «Prueba de familiarización», «What is cloud computing?», «Probar consulta», «Comenzar tareas» | Propuesta: misma consulta genérica en ambos bloques y para todos. Permite aprender la interfaz sin usar una tarea puntuada. El segundo contacto puede facilitar el uso; el orden contrabalanceado distribuye ese aprendizaje. Solo evento de avance persistido. |
| Espera | «Preparando la respuesta. Tiempo transcurrido: MM:SS.» | Reloj de navegador igual en ambas condiciones. Lectura HTTP de 180 s; no es límite total. |
| 60 s | «La consulta está tardando más de lo previsto. Sigue en curso; no necesitas enviarla otra vez.» | Reconoce demora sin atribuir calidad ni identificar condición. |
| 120 s | «La consulta continúa. Puedes esperar o avisar al coordinador. Recargar la página no cancela la consulta.» | Conserva aviso autorizado. La demora sigue contando como tal; no altera el umbral propuesto. |
| Respuesta | Respuesta original; «Fuentes» / «Abrir fuente» solo si hay fuentes presentables | Se eliminan metadatos vacíos None/N/A de las citas, nunca del texto original. No hay bloque de fuentes vacío en la condición sin consulta documental. No se muestran puntajes de confianza. |
| Error de consulta | «No se pudo completar la consulta. Puedes reintentar o avisar al coordinador.» | El intento queda como error, sin valoración; no habilita avance. |
| Formulario desactualizado / fallo de guardado | «La sesión cambió o no se pudo guardar. Contacta al coordinador y vuelve a comprobar las respuestas.» | No duplica instrumentos ni muestra stacktraces técnicos; no se atribuye el fallo a una condición. |
| Consulta libre | Texto literal de `config/study.example.json:free_instruction`, antes de «Tu consulta» | Advierte no incluir información confidencial. Se registra completa para análisis cualitativo, fuera del desenlace primario. |
| Instrumentos | SUS literal por completar por el usuario; F/U/R/I literal de configuración. «1 = Totalmente en desacuerdo · 5 = Totalmente de acuerdo» | Sin respuestas preseleccionadas. Al terminar cada bloque y antes del siguiente. No se inventó texto SUS. |
| Cierre | C1–C4 y chequeo literal de configuración; «La sesión ha terminado. Gracias por participar.» | Comparación después de ambos bloques. No se expone la exportación ni el mapeo interno al participante. |

La prohibición de pistas se verifica sobre textos propios de la plantilla. No se
reescriben las respuestas del modelo ni los títulos de documentos para censurar
vocabulario técnico: eso alteraría el tratamiento y la respuesta original.
La familiarización vive únicamente en memoria de la conexión; reconectar puede
exigir repetirla. Sus reintentos no se registran. Un bloqueo técnico persistente
se registra mediante el operador como incidente de sesión sin contenido.
