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

El controlador de host usa el sampler Linux fuera del namespace de la app.
Comprueba identidad de los contenedores, sus PIDs y el acceso bloqueado a metadatos;
mantiene el calendario privado por stdin/stdout y guarda solo metadatos en el
progreso técnico. La admisión de 60 s se enlaza por hash, y cada llamada queda
ligada a sus fronteras y filas de telemetría. El verificador LIVE exige esas
pruebas además de las observaciones del runner. La cola privada está acotada y
su cierre no queda bloqueado por una cola llena.

`stimulus-start` valida la sintaxis de la unidad con el systemd instalado, cierra
admisión y lanza un job independiente con RuntimeMaxSec=7200 y apagado final.
Rechaza study, datos existentes, historial consumido, margen menor de 125 min y
la repetición de un intento registrado, aunque su lanzamiento haya fallado.
`stimulus-collect` descarga en privado, verifica SHA-256 antes y después de guardar
y confirma el apagado; no muestra ni registra el texto. Un arranque parcial no se
sustituye. El recibo dice BOOT_COMPLETE_UNANALYZED hasta el análisis del censo.

Estado: flujo de host y controles verificados con fixtures, sin modelo local.
Falta incluirlos en la imagen final, validar sus binarios en la VM, anclar la
identidad, ejecutar y descargar los doce arranques reales. No existe todavía
aceptación LIVE por este documento o por esos tests.
