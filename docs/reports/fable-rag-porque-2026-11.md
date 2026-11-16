[Fable 5.2] | 2026-11-16 04:40 | rama `fable/rag-auditoria` @ `bf8a18d` | tarea F-propuesta (sin número, ver §9) | modo: ejecución
Capas: 1 secretos=exit 0 | 2 tests=exit 0 (2406 pasan, 0 fallan, 6 omitidas, 8 xfail estrictos) | 3 docs=exit 0
Cambios: `scripts/audit/fable_why_analysis.py`, sus tests, el reporte generado en `output/audit/` y este informe | Sin verificar: este análisis no ejecuta modelos; exp20 (firmada el 2026-11-16) está en el informe de conceptos; `init.sh` completo corrió con la evidencia firmada presente y antes del commit que solo añade documentación (capas 1 y 3 repetidas después); ver §5

# Por qué el híbrido no siempre gana y por qué unos modelos puntúan más

Encargo de Enzo (2026-11-15): que cada explicación salga de código y sea exacta, no una atribución
genérica «al modelo». Este informe interpreta la salida de
[`scripts/audit/fable_why_analysis.py`](../../scripts/audit/fable_why_analysis.py), guardada en
[`output/audit/fable_why_analysis_2026-11-15.md`](../../output/audit/fable_why_analysis_2026-11-15.md).

No se ejecutó ningún modelo (decisión D1 vigente). Todo sale de la evidencia firmada, leída sin
modificarla, y de los índices en disco. Cada afirmación lleva una de tres etiquetas:

- **Demostrado:** se sigue de la aritmética del código o de una identidad exacta.
- **Medido:** es un conteo o un contraste sobre la evidencia firmada.
- **No resuelto:** el código no alcanza; hace falta ejecutar un modelo (§5).

## 1. Respuesta corta

1. **La fusión RRF no supera a la búsqueda densa porque, con los parámetros usados, actúa como un
   filtro de consenso.** Con `k = 60` y 50 candidatos por pierna, cualquier fragmento que esté en
   las dos listas queda por encima de cualquier fragmento que esté en una sola (demostrado). El
   primer resultado denso se pierde en 82 de 194 consultas, y en 79 de ellas porque BM25 no lo
   tenía entre sus 50 (medido).
2. **La ventaja del híbrido aparece después del reranker y depende del pool, no de la fusión.** El
   64,7 % del top-5 final sube desde los puestos 6 a 50 del pool (medido).
3. **En fidelidad el híbrido no gana porque el diseño no puede detectar efectos del tamaño
   observado.** Ninguno de los 12 contrastes entre escenarios alcanza el mínimo efecto detectable
   con su número de pares (medido). Esto coincide con exp18, que ya probó de forma causal que dar
   la mejor evidencia no sube la fidelidad.
4. **Gemma encabeza la tabla porque afirma mucho menos, no porque apoye más.** En las mismas
   consultas emite unas 4 afirmaciones por respuesta frente a 14 a 18 de Granite, y obtiene 1,4
   apoyadas frente a 4,0 a 4,6 (medido). El cociente premia el denominador pequeño.
5. **La mitad de cada brecha entre modelos es selección de consultas.** Cada modelo contesta un
   conjunto distinto de preguntas; en promedio el 50 % de la brecha (verificador small) viene de
   ahí y no de cómo responde (medido).

## 2. Recuperación

### 2.1 La propiedad (demostrado)

El código puntúa cada fragmento con `suma de 1/(k + puesto)` sobre las listas donde aparece
(`src/embedding/index/hybrid_index.py`, `_rrf_fusion`), con puestos desde 1 y 50 candidatos por
pierna.

- Peor fragmento presente en las dos listas (puesto 50 en ambas): `2/(60+50) = 0,018182`.
- Mejor fragmento presente en una sola lista (puesto 1): `1/(60+1) = 0,016393`.

Como `0,018182 > 0,016393`, todo fragmento de consenso supera a todo fragmento de una sola pierna.
La desigualdad `2/(k+N) > 1/(k+1)` equivale a `N < k + 2`: vale hasta 61 candidatos por pierna y
deja de valer en 62. Con `k = 60` y `N = 50` la fusión ordena primero por «está en las dos
listas» y solo después por puesto.

Consecuencia: el top-5 fusionado se llena con la intersección de las dos listas de 50. El primer
resultado denso entra solo si BM25 también lo recuperó, o si la intersección tiene menos de 5
fragmentos.

### 2.2 Lo que hace con las listas firmadas (medido, exp11, 194 consultas)

| Medida | Valor |
|---|---|
| Fragmentos compartidos entre el top-5 léxico y el top-5 denso | media 0,75 de 5; en 104 consultas, ninguno |
| Top-5 denso que sobrevive en el top-5 fusionado, si BM25 lo tenía entre sus 50 | 334 de 417 (80,1 %) |
| Top-5 denso que sobrevive, si BM25 no lo tenía | 29 de 553 (5,2 %) |
| Consultas donde el primer resultado denso sale del top-5 fusionado | 82 de 194 |
| De esas, por no estar en el top-50 de BM25 | 79 |
| Consultas cuyo top-5 fusionado no contiene ningún fragmento solo-denso | 174 de 194 |
| Top-5 final (tras el reranker) que ya estaba en el top-5 fusionado | 35,3 % |
| Top-5 final que sube desde los puestos 6 a 50 del pool | 64,7 % |
| Top-5 final formado por fragmentos fuera del top-50 de BM25 | 31,1 % |

Lectura: las dos piernas casi nunca coinciden arriba, así que el consenso que RRF premia está en
puestos profundos. La pierna densa es la más fuerte (NDCG@5 0,624 frente a 0,442 de BM25, oráculo
independiente, cifra firmada), y la fusión reemplaza sus mejores resultados exclusivos por
fragmentos que BM25 también vio. El resultado firmado es coherente con eso: híbrido antes del
reranker 0,603, por debajo del denso. El reranker deshace parte del efecto: casi un tercio del
top-5 final son fragmentos solo-densos que la fusión había dejado por debajo del consenso.

### 2.3 Dos comprobaciones de que la evidencia firmada obedece esta aritmética (medido)

- El brazo léxico se recalculó hoy con el índice real y coincide con el firmado en 194 de 194
  consultas (mismos identificadores, mismo orden).
- En las listas fusionadas firmadas no hay ninguna violación del orden que predice RRF: 0 casos de
  un fragmento solo-denso por delante de uno de consenso conocido, y 0 inversiones entre los 210
  fragmentos cuyo puntaje exacto se puede reconstruir (115 consultas sin filtro de proveedor).

## 3. Fidelidad entre escenarios: por qué el híbrido no siempre gana

Tabla citable v4 (verificador small): Granite 0,235 / 0,247 / 0,299 (léxico, denso, híbrido);
Gemma 0,413 / 0,377 / 0,317. El script reproduce las 24 celdas de los dos verificadores con n
exacto.

**Potencia (medido).** Las consultas que un modelo contesta en dos escenarios a la vez son entre
18 y 109. Con ese n, el mínimo `|d_z|` detectable al 80 % de potencia va de 0,27 a 0,66. Ningún
contraste observado lo alcanza en ninguno de los dos verificadores (0 de 12). El mejor caso es
Granite híbrido frente a léxico: `d_z = +0,31` con 53 pares; el mínimo detectable con 53 pares es
0,38, y harían falta 82 pares.

**Selección (medido).** La media de cada celda se calcula sobre las consultas que ese modelo no
declinó en ese escenario, y ese conjunto cambia con el contexto. En 2 de los 8 contrastes del
híbrido (small) la brecha publicada y la brecha dentro de las mismas consultas tienen signo
opuesto. Ejemplo: Qwen híbrido frente a denso muestra +0,029 en la tabla, pero en las 29 consultas
comunes es −0,114; la diferencia (+0,143) es composición del conjunto contestado.

**El caso de Gemma (medido).** En sus 30 consultas comunes, híbrido menos léxico es −0,145
(`d_z = −0,30`, no significativo). La descomposición dice de dónde sale: −0,210 por menos
afirmaciones apoyadas y +0,065 por menos afirmaciones emitidas. Con contexto híbrido Gemma obtiene
menos afirmaciones apoyadas en las mismas preguntas. Con 30 pares no se distingue del azar.

**Dosis y respuesta (medido).** Si la fidelidad respondiera al contexto, cambiar más fragmentos
debería mover más la fidelidad. La correlación entre fragmentos distintos y `|Δf|` va de −0,22 a
+0,44, sin patrón estable: 2 de 12 contrastes con p < 0,05 sin corregir bajo small y 1 de 12 bajo
base, y no son los mismos. No hay una relación consistente.

Esto no añade un resultado nuevo al de la fase de verano; lo acota. exp18 ya contrastó de forma
causal que sustituir la evidencia por la mejor disponible deja la fidelidad equivalente (TOST
firmado). Lo que aporta este análisis es el tamaño del efecto que el diseño de exp12 podía ver.

## 4. Fidelidad entre modelos: por qué unos puntúan más

**No es la recuperación (demostrado sobre los datos).** Los cuatro modelos recibieron los mismos
cinco fragmentos, en el mismo orden, en las 582 celdas (escenario, consulta). Quedan tres fuentes:
qué consultas contesta cada modelo, cuántas afirmaciones emite y cuántas apoya el verificador.

**Identidad usada (demostrado).** Para dos modelos A y B:

```
media_A − media_B = selección + dentro
dentro            = efecto de apoyadas + efecto de emitidas
```

`selección` es la parte debida a que contestan consultas distintas. `dentro` se mide solo en las
consultas comunes y se reparte entre numerador y denominador del cociente `apoyadas / emitidas`.
Las 36 descomposiciones entre modelos (18 por verificador) cierran exactamente.

**Gemma frente a Granite, escenario léxico, verificador small (medido):**

| Componente | Valor |
|---|---|
| Brecha en la tabla | +0,178 |
| Selección | +0,108 |
| Dentro de las 42 consultas comunes | +0,071 (`d_z = +0,24`, no significativo) |
| · por afirmaciones apoyadas | −0,626 |
| · por afirmaciones emitidas | +0,696 |
| Apoyadas por respuesta, Gemma frente a Granite | 1,45 frente a 4,05 |
| Emitidas por respuesta, Gemma frente a Granite | 3,88 frente a 14,00 |

Gemma no apoya más: apoya casi un tercio de lo que apoya Granite. Gana el cociente porque emite
poco más de una cuarta parte. El patrón se repite en los tres escenarios y con los dos
verificadores.

**Hipótesis contrastada y no confirmada (medido).** «Con la misma pregunta y el mismo contexto,
emitir más afirmaciones baja la fidelidad». Regresión con efectos fijos de (escenario, consulta) y
errores agrupados por consulta, 763 respuestas: `beta = −0,027` (p = 0,18) con small y
`beta = +0,001` (p = 0,97) con base. No se sostiene. La razón es que el número de afirmaciones es
casi una constante de cada modelo (Gemma unas 4, Mistral y Qwen de 9 a 12, Granite de 14 a 19), así que
longitud y modelo no se pueden separar con estos datos. La descomposición de arriba es contabilidad
exacta; no prueba que alargar la respuesta cause la caída.

**Hallazgo exploratorio retirado (corrección del 2026-11-15).** La primera versión de este informe
decía que, juntando los tres escenarios, Gemma supera a Granite en 0,08 con los dos verificadores
(Holm sobre 6 pares: p = 0,016 con small y p = 0,022 con base). Ese resultado sale de una regresión
conjunta de los cuatro modelos. Con diferencias pareadas solo en las celdas donde ambos modelos
contestaron, la brecha es +0,057 (Holm 0,30) con small y +0,078 (Holm 0,066) con base; agrupando
por plantilla de consulta, 0,36 y 0,14. Ningún par de modelos es significativo con los dos
verificadores. El hallazgo no resiste el cambio de estimador y se retira: el «0 de 18 pares
robustos» del ledger N9 se mantiene sin matices. Detalle en
[el informe de conceptos](fable-rag-conceptos-2026-11.md), C11.

## 5. Lo que el código no puede responder sin ejecutar modelos

Cada punto necesita una corrida nueva con ID `exp20+` y tu autorización (D1, ADR-0004).

**Actualización del 2026-11-15:** autorizaste exp20 y las dos primeras filas ya tienen respuesta
en [el informe de conceptos](fable-rag-conceptos-2026-11.md), §2. Las cuatro restantes siguen
abiertas.

**Actualización del 2026-11-16:** exp20 quedó firmada en `experiments/results/exp20_rerank_arms/`
y replica bit a bit la corrida de auditoría (informe de conceptos, §2.5).

| Pregunta | Por qué no se puede hoy | Corrida mínima |
|---|---|---|
| ¿Cuánto NDCG cuestan los 79 primeros resultados densos que la fusión expulsa? | Los puntajes del oráculo por (consulta, fragmento) de exp11 no se guardaron; solo sus agregados | Puntuar con `bge-reranker-large` los top-5 de los cuatro sistemas. Sin Ollama |
| ¿El híbrido con reranker supera a un denso con el mismo reranker? | Los brazos léxico y denso firmados no pasan por el reranker. Hoy «híbrido» y «reranker» están confundidos | Brazo nuevo: denso top-50 más reranker, y léxico top-50 más reranker |
| ¿Mejora la fusión con 62 o más candidatos por pierna, o con otro `k`? | Cambia rankings: clase B | Barrido de `N` y `k` en recuperación sola |
| ¿Cuánto ruido propio tiene la generación? | En exp12 no hay ningún par (modelo, consulta) con contexto idéntico entre escenarios | Réplicas con el mismo prompt (exp14 quedó abortado) |
| ¿Alargar la respuesta causa menor fidelidad? | La longitud es casi constante por modelo | Mismo contexto y modelo, con límite de afirmaciones en el prompt |
| Prueba de extremo a extremo con búsqueda densa, reranker y generación reales | D1 | Las 20 consultas fijas de la alternativa D1 |

## 6. Decisiones que te tocan

1. **¿Autorizas alguna corrida de §5?** La más barata y de más valor es la segunda: sin ella, la
   frase «el híbrido supera al denso» mezcla dos componentes.
2. ~~¿Se anota en el ledger el contraste exploratorio Gemma frente a Granite?~~ Retirada: el
   contraste no es robusto (§4). No hay nada que anotar.
3. **¿Quieres un parche de clase B que suba los candidatos por pierna o pondere la pierna densa?**
   Cambiaría rankings y el camino desplegado; solo lo propongo si lo pides.

## 7. Verificaciones hechas en esta ronda

| Comprobación | Resultado |
|---|---|
| Integridad de índices: identificadores, dimensiones, normas, auto-recuperación de 512 vectores | 8 de 8 |
| Reproducción de extremo a extremo del brazo léxico con el índice real | 194 de 194 |
| Listas fusionadas firmadas frente a la aritmética de RRF | 0 violaciones |
| Celdas de la tabla citable v4 recalculadas | 24 de 24 |
| Descomposiciones entre modelos que cierran exactamente | 36 de 36 (las 24 entre escenarios usan la misma identidad; el script no las comprueba aparte) |
| RRF reimplementado frente al de producción, 100 casos aleatorios con empates | idéntico |
| Tests nuevos (`tests/test_fable_why_analysis.py`) | 12 rápidos y 1 marcado `slow` que recorre todo el análisis |
| Revisión independiente de los commits de B1, B2 y B6 | aprobada; 0 críticos, 0 altos, 1 medio, 3 bajos (detalle en el informe de auditoría, §13) |

## 8. Límites

- El oráculo de relevancia es un modelo (`bge-reranker-large`); no hay juicios humanos de ranking.
- La fidelidad es relativa al verificador. Por eso cada contraste se da con small y con base.
- Los contrastes de §3 y §4 son exploratorios. Las familias BH y Holm son las de este script, no
  las de la tabla citable. Los p de la regresión usan aproximación normal con errores agrupados en
  128 consultas.
- Las comparaciones «dentro» condicionan a que ambos brazos contestaron la consulta.
- En 79 de 194 consultas hay filtro de proveedor; ahí el puesto crudo de la pierna densa no se
  conoce y las comprobaciones de puntaje exacto las excluyen.

## 9. Cómo reproducirlo

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 PYTHONUTF8=1 .venv-app/Scripts/python.exe scripts/audit/fable_why_analysis.py
```

Termina con exit 1 si alguna comprobación deja de cumplirse. Con `--write` vuelve a guardar el
reporte y el JSON en `output/audit/`.
