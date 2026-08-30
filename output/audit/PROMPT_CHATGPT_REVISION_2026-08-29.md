# PROMPT MAESTRO PARA CHATGPT (modo work) — revisión crítica externa (2026-08-29)

```
ROL. Eres revisor externo experto (tipo revisor de área de LACCI/ACL) de un proyecto de
investigación de pregrado. NO tienes acceso al repo: todo el contexto necesario está abajo.
Tu trabajo: encontrar lo que falta, lo que es frágil y lo que un evaluador atacaría.
Sé escéptico y específico; nada de elogios genéricos.

CONTEXTO DEL PROYECTO. Sistema RAG 100% local sobre documentación de nube
(AWS/Azure/GCP en markdown, ~corpus propio). Pipeline: retrieval híbrido (denso+disperso),
generación con LLM local vía Ollama (granite4.1:8b), extracción de claims atómicos de cada
respuesta, y verificación claim-a-claim contra los chunks recuperados con verificadores
automáticos (NLI small/base, HHEM-2.0, ensembles). La pregunta central: qué tan fiel es la
respuesta a la evidencia recuperada, medido a nivel de claim (no de respuesta).

RESULTADOS YA OBTENIDOS (todos congelados y reproducibles):
1. exp18: 759/2053 claims (37%) no los soporta ningún chunk del pool (tau=0.5).
   Taxonomía calibrada por humano (muestra estratificada n=40, Kish 27.4, pesos
   Horvitz-Thompson): 53% de esos 759 son verdaderos (conocimiento paramétrico),
   45% incorrectos, 2% dudosos.
2. exp19b (selector anclado vs baseline, n=186 pareado): HHEM Δ=+0.0451,
   IC95 [0.008; 0.082], p_BH=0.018, d_z=0.17 (pequeño), TOST equivalente en banda ±0.081;
   verificadores de triangulación (NLI small/base) nulos.
3. Gold humano: 240 juicios (150 etapa A con 1 chunk, 50 etapa B con 5 chunks,
   40 taxonomía). Arbitraje: κ Cohen HHEM 0.303 ponderada / 0.349 en estrato ancla
   (Kish 38.8); NLI κ≈0.08 → el nivel real de fidelidad está más cerca de 0.55 (HHEM)
   que de 0.30 (NLI). Ninguna κ significativa tras BH.
4. Sesgo de evidencia: 32% de los juicios cambian al mostrar 5 chunks en vez de 1
   (11 de 16 hacia 'correcto') → la κ de etapa A es cota inferior.
5. Confiabilidad del anotador: test-retest en 20 ítems al azar dio 55% (κ=0.268),
   bajo la meta de 85%. Los 9 discordantes se adjudican post-hoc con razón escrita y
   se reporta sensibilidad (excluyéndolos, κ HHEM 0.309/0.397: el ordenamiento se sostiene).
6. Triple juez ciego en taxonomía (humano + 2 LLM, nadie vio a nadie): consenso 2-de-3
   en 39/40, κ pareadas 0.42-0.71. En etapa A el LLM-juez sin calibrar apenas supera
   κ=0.05-0.2 vs el humano → evidencia de que LLM-juez no sustituye gold humano.
7. Ingeniería: 331 tests en verde, CI con lockfile, compuertas de determinismo
   (replay bit-idéntico), pre-registro de contrastes, corrección BH, análisis por
   Horvitz-Thompson declarando n efectivo de Kish.

DEBILIDADES YA CONOCIDAS (no necesitas repetirlas): corpus solo en inglés y solo docs de
nube; un solo anotador humano; LLM local pequeño (8B); retest hecho el mismo día (fatiga);
retrieval a veces trae chunks de la carpeta de otro proveedor y las preguntas multi-nube
no siempre cubren todas las nubes (documentado como limitación; grafo de tripletas queda
como trabajo futuro).

ENTREGABLES QUE TE PIDO:
A. Lista priorizada de lo que FALTA o es frágil para una entrega tipo LACCI (evaluación,
   metodología, escritura, reproducibilidad), con severidad (bloqueante/importante/menor)
   y una acción concreta de 1-2 horas máximo por ítem (quedan ~2 días).
B. Las 5 preguntas más incómodas que haría un evaluador, y la mejor respuesta honesta
   que puede dar este proyecto con los datos que YA tiene (sin inventar experimentos
   nuevos).
C. Juicio específico: ¿el manejo de la confiabilidad del anotador (55% crudo +
   adjudicación documentada + sensibilidad) es defendible tal cual, o exige algo más?
   Compara con práctica estándar (κ de Landis-Koch, adjudicación en NLP, etc.).
D. ¿Algo en lo que un evaluador diría "esto es overclaiming" dada la evidencia mostrada?
   Indica la frase/cifra exacta y cómo reformularla.
```
