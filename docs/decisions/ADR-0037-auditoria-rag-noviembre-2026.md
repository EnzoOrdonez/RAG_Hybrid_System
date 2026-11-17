# ADR-0037: Auditoría del código RAG de noviembre de 2026 y exp20

[DECLARADO] Registrado el 2026-11-17 al integrar la rama `fable/rag-auditoria` (base `1574d22`).
Tarea F-022. La auditoría la hizo Fable 5.2 entre el 15 y el 16/11; Claude revisó el diff y
registró las decisiones. Las decisiones sobre B3, B4, B5, B7 y B8 siguen la delegación de Enzo del
17/11 ("Puedes hacer todas tu excepto la 7 con las opciones mas recomendadas?"). Las entradas de
ledger, la nota de erratas y las frases para el A.3 tienen OK explícito de Enzo del mismo día.

## Contexto

Con la lógica RAG congelada (ADR-0004) y el despliegue I7 en fase humana, se auditó `src/` en una
rama aislada. La auditoría separó los cambios que no alteran ningún resultado (clase A) de los que
sí (clase B). Después, con autorización de Enzo, corrió exp20: el control que faltaba para
separar el aporte de la fusión del aporte del reranker. Informes:
[fable-rag-auditoria-2026-11.md](../reports/fable-rag-auditoria-2026-11.md),
[fable-rag-conceptos-2026-11.md](../reports/fable-rag-conceptos-2026-11.md) y
[fable-rag-afirmacion-principal-2026-11.md](../reports/fable-rag-afirmacion-principal-2026-11.md).

## Decisión

[DECLARADO]
1. **Clase A, integrada:** caché LLM atómica y con aviso, docstring veraz del runner sin timeout
   por consulta y docstring de `balance()`, con los tests `tests/test_fable_*.py`.
2. **Clase B aplicada** (Enzo, 15/11): B1 (`BenchmarkRunner` no reescribe evidencia protegida),
   B2 (`p = 1` para pares totalmente empatados) y B6 (sin recorte de latencias antes de p95, p99 y
   máximo).
3. **Clase B sin aplicar:** B3, B4, B5, B7 y B8. Sus tests quedan como `xfail` estricto y sus
   parches en `docs/reports/fable-rag-2026-11/patches/`.
   - B4 y B5 tocan el balanceo de proveedores del camino desplegado. Aplicarlas cambia el sistema
     que evalúa el estudio con personas y exige otra Fase B. Quedan como límite del estudio.
   - B3 depende de una pregunta previa (si una consulta sin fragmentos es un error o un cero de
     recuperación) y cambiaría la puntuación de corridas futuras.
   - B7 no tiene efecto práctico hoy y cambiaría el comportamiento de `exp2`.
   - B8 solo afecta a corridas futuras con caché. No se regeneran las respuestas vacías de exp12.
   - Las cinco se reconsideran antes de cualquier corrida `exp21+` o de una imagen nueva.
4. **exp20 conserva su nombre.** `exp20_rerank_arms` está firmada desde el 16/11 con escritura
   única y manifiesto de procedencia, y cae en el rango «exp20 o superior» previsto para
   experimentos nuevos. El nombre tentativo «exp20» de un selector de diversidad que nunca se
   corrió pasa a `exp21+`.
5. **Ledgers** (OK de Enzo del 17/11): la entrada 36 de `paper/summer_ablation_log.md` registra
   exp20, y `paper/correction_log.md` registra la fe de erratas S6 («42 multi-cloud queries» son
   25). El README publica la nota de erratas en la sección «Publication». El `.tex` y el PDF
   certificado no se tocan.
6. **Afirmación principal** (OK de Enzo frase por frase, 17/11): se aceptan las ocho propuestas
   S1, S5, S7c, S11a, S11c, S13, S6 y S12 para el A.3, la defensa y cualquier versión posterior.
   Las aplica Enzo en esos documentos.

## Consecuencias

- Los tests de caracterización fijan el texto de los prompts, la fusión y el camino de
  `RAGPipeline.query`. Un cambio en cualquiera de ellos rompe un test y obliga a declararlo.
- Un parche de clase B que toque el camino desplegado no llega a la web sin una Fase B nueva.
- La afirmación principal queda así: la ventaja del pipeline sobre el denso sin reranker procede
  del reranker; con el mismo reranker, el híbrido supera al denso en +0,016 de NDCG@5, sin
  significación (p = 0,075).

## Decisiones menores del 2026-11-17

[DECLARADO] Por delegación de Enzo («las decisiones menores … con las opciones recomendadas»):
- **Respuestas vacías de exp12:** recontadas (399 de 3 104; 177 de qwen sin RAG, 168 de qwen con
  RAG y 54 de gemma con RAG) y registradas en `paper/correction_log.md` como complemento de N6.4.
- **Aviso del Flag 33:** añadido a las limitaciones del A.3 V2.21, con OK de Enzo a la frase.
- **Validación humana del oráculo (C2):** no se hace ahora. Queda como límite declarado (S12) y
  como trabajo futuro, que el A.3 ya lista.
- **Fidelidad con ventanas en lugar de truncado (C5):** no se corre ahora; sería una corrida
  `exp21+` con modelos. El aviso del Flag 33 declara el truncado medido.
- **Aviso a LACCI o a IEEE de la fe de erratas S6:** documentado fuera del repositorio para que
  decida Enzo.
- **A.3:** la V2.21 aplica S1, S5, S7c, S11c, S12 y S13 y el aviso del Flag 33. S6 no cambia porque
  el A.3 ya decía 25 consultas multiproveedor, y S11a no aplica porque el A.3 no tiene la cifra del
  +67 %. Los archivos viven fuera del repositorio, junto a la V2.20, que no se sobrescribe.

## Verificación

[VERIFICADO]
- Revisión del diff de `src/` (5 archivos): retornos consistentes en `paired_comparison`, sin
  referencias al `TimeoutError` eliminado e imports completos.
- Recuento de S6 repetido el 17/11 sobre `data/evaluation/test_queries.json`: 194 consultas, 169
  con un proveedor, 8 con dos y 17 con tres; 25 multi-proveedor, idénticas a
  `data/evaluation/cross_cloud_subset.json`.
- Diff de la rama sin IPs, hosts, correos ni identificadores de proyecto.
- `init.sh` con exit 0 en las tres capas sobre la punta integrada.
