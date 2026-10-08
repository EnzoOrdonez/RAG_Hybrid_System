# Preparación del colector de historiales fríos

La preparación de la compuerta genera respuestas de calentamiento para ambos
sistemas. Reutilizarla para el primer objetivo posterior al arranque consumiría
ese historial. El colector de estímulo carga los pipelines sin llamar `query`;
comprueba STARTING/sequence=1 antes y después de prepararlos. La compuerta
mantiene su preparación caliente vigente y sus fronteras sin cambios.

El colector llama al mismo `study_service.execute_query` y a la misma presentación
de la aplicación. El observador mide el request en la frontera del cliente
Ollama después de añadir num_ctx; devuelve el mismo objeto y restaura el cliente
y sus overrides. Conserva solo hash y opciones del request, texto crudo y
presentado, clase v2, citas, IDs de contexto, etapas y rerank_score en el registro
privado de un arranque sintético técnico. No registra prompts fuera de ese
registro. No sirve respuestas grabadas ni cambia ningún módulo RAG congelado.

La aceptación sigue exigiendo el calendario de 144 llamadas, 120 objetivos,
10 repeticiones por combinación y 12 arranques distintos. No cambia umbrales,
selección ni el mecanismo fresh_runner preregistrado en la iteración 4.

El supervisor conserva los límites de 900 s de preparación, 600 s por llamada
y 120 min por arranque. Su llamador de host debe suministrar admisión fría y
telemetría continua verificadas; sin ellas rechaza iniciar. Muere el worker si
muere su padre en Linux. Una llamada fallida, vencida o contaminada deja el
arranque terminal; solo un arranque completo verificado se puede publicar.

Estado: componentes en implementación y pruebas sintéticas. Falta integrar la
orquestación de host, la admisión fría y las observaciones GPU fuera del namespace
del contenedor, incluir el controlador en la imagen final y anclar su identidad.
No existe todavía aceptación LIVE por este documento o por esos tests.
