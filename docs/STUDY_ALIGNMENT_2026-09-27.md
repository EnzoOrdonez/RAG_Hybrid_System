# Alineación del estudio — decisiones previas a implementar

Baseline VERIFICADO: `521f525`, 674 tests / 5 excluidos / 9 subpruebas.
Auditoría: `C:/CloudRAG/study-alignment-20260927T172349490Z/`.
Ruff global tiene 149 incidencias previas; secretos sin hallazgos. No se modifica
ningún pre-registro aprobado. Artículo V2.11 §4.8 leído en su DOCX original.

| Requisito | Código previo | Cambio y prueba prevista |
|---|---|---|
| Dos condiciones | session_manager.py:26, index_loader.py:55 | Configuración del estudio separada; doble que falla al acceder al índice en no_rag |
| Asignación 2×2 estratificada | session_manager.py:140 | CSV completo, cuotas y mapeo global; rechazo de celdas/perfiles incorrectos |
| Seis tareas emparejadas | session_manager.py:62 | Configuración editable validada contra las 194 consultas y denylist |
| Práctica por bloque sin registro | evaluation_page.py:110 | Sólo evento de avance; sentinel que no aparece en disco/exportación |
| SUS y Likert por bloque | session_manager.py:187, evaluation_page.py | Dos instrumentos por sesión antes de cambiar condición; puntuación 0/50/100 |
| Consulta libre y cierre | evaluation_page.py | Flujo explícito y pruebas de transición/recarga |
| Exportación y análisis por sujeto | session_manager.py:486, analyze_user_sessions.py:130 | Esquema nuevo, validación estricta, tests sintéticos pareados |
| Citas sin None/N/A | evaluation_page.py:239 | Proyección de presentación sin alterar respuesta; prueba con campos vacíos |
| Cegamiento de etiqueta | app.py:31, evaluation_page.py | Ruta y título neutrales; AppTest y léxico prohibido en textos propios |

## Crítica y límites

La asimetría de fuentes y latencias impide prometer cegamiento funcional. No se
añaden esperas artificiales ni se ocultan respuestas lentas. Los textos propios
de UI serán neutrales; no se censura contenido generado ni documentación citada.
La práctica fija será la misma en ambos bloques, fuera de T1/T2, para aprender la
interfaz; se reconoce aprendizaje de esa respuesta genérica, no de tareas puntuadas.

El usuario confirmó: mapeo A/B global congelado; eliminar ratings por consulta;
BH sobre SUS, promedio F1–F4 (F3 inverso) y promedio U1–U3 conjuntamente.
R1/R2 e I1 descriptivos separados. No hay nuevas decisiones inferenciales por perfil.

Aclaración del prompt, no cambio de protocolo: familiarización no persiste contenido,
fuentes, latencia ni errores de reintentos. Sólo `familiarization_done` y timestamp
por bloque. Si impide continuar, incidente de sesión sin contenido. La consulta libre
sí persiste completa con `analysis_role=free_query`; no entra al desenlace primario.
El calentamiento es independiente y obligatorio antes de cada participante.

NLI queda aislado, por autorización explícita, en `.worktrees/nli-521f525`, con
su propia rama `fix/interview-readiness`, HEAD `521f525` y bundle verificado.
Los primeros enlaces de bundle fueron rechazados por confinamiento de rutas;
se reemplazaron por copias físicas verificadas, sin alterar originales.
La venv compartida no se actualiza. La cohorte real sigue siendo tarea humana.

Riesgos: cambiar defaults de componentes compartidos rompería la cohorte NLI;
se preservan defaults históricos y se separa el protocolo de estudio. Los datos
históricos no se migran. La lista de asignación no debe mutar: se sella por hash,
los reemplazos se registran aparte y conservan celda/perfil. No se puede garantizar
que texto libre no contenga datos personales: no se añaden campos identificables,
se advierte al participante y se exige revisión humana de exportaciones.
