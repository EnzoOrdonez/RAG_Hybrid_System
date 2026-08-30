# Prompt maestro para ChatGPT (modo work) — Crítica constructiva final pre-entrega LACCI

> Copiar y pegar TODO el bloque siguiente en ChatGPT work. Es autocontenido.
> Fecha: 2026-08-30. Entrega LACCI: 30-31 de agosto.

---

Actúa como revisor externo senior (track de sistemas de recuperación aumentada / evaluación de LLMs). Ya hiciste una primera revisión de este proyecto el 2026-08-29; esta es la revisión FINAL antes de la entrega a LACCI. NO escribas código ni edites archivos: solo dictamen crítico, en español, accionable.

## Qué es el estudio (encuadre adoptado tras tu primera revisión)

El trabajo se re-encuadró como **auditoría de la medición de fidelidad en un RAG local** (no como benchmark de sistemas). Terminología adoptada: `unsupported@τ`, *human-judged*, *externally incorrect*; la referencia humana se presenta como **piloto** (n=200), con pre-adjudicación como versión principal y adjudicación como reconciliación documentada. Tu tabla de riesgos de overclaiming se aplicó a la redacción.

## Cifras finales congeladas (verificadas en el repo, rama summer/taxonomia-759)

**Comparación de verificadores automáticos contra el gold humano (200 ítems, post-adjudicación):**
- HHEM: κ ponderada **0,315** [IC95 0,0137; 0,5712], κ anchor **0,3966**
- E5_base_and_hhem: 0,160 / anchor 0,170 — small: 0,092 / 0,016 — base: 0,084 / 0,078 — E1_mean: −0,029 / −0,080
- **Sensibilidad a la adjudicación**: ordenamiento idéntico en 3 variantes (post / sin-los-9 / pre), Δκ máx 0,012
- **Sensibilidad de umbral**: unsupported@τ = 636 / 759 / 880 claims para τ = 0,4 / 0,5 / 0,6

**Sensibilidad al conjunto de evidencia (etapa A: 1 chunk → etapa B: 5 chunks, mismos 50 claims):**
- 18/50 juicios cambian (36,0 %; IC95 Wilson [24,1 %; 49,9 %]), 12 hacia `correcto`; McNemar descriptivo p=0,0352 (fuera de BH)

**Paquete de confiabilidad del anotador humano:**
- Retest intra-anotador: acuerdo crudo 11/20 (55,0 %; Wilson [34,2 %; 74,2 %]), κ=0,2683 — meta 85 % declarada como incumplida
- Taxonomía de 759 claims con muestreo Horvitz-Thompson, Kish efectivo 27,4
- Auditoría del extractor: 15 respuestas, 127 claims brutos → 121 retenidos

**NUEVO desde tu última revisión — acuerdo triple de jueces (humano vs 2 LLM ciegos independientes, mismos 200 ítems, cegamiento total):**

| Par | κ (3 clases) | κ binario | Acuerdo binario |
|---|---|---|---|
| Kimi–Codex (global) | +0,667 | +0,754 | 89,8 % |
| Humano–Kimi (global) | +0,138 | +0,204 | 54,8 % |
| Humano–Codex (global) | +0,166 | +0,171 | 54,5 % |
| Humano–LLM, etapa A → B | κ₂ 0,10 → 0,32 | | 49 % → 72 % |

Distribuciones etapa A (C/I/D): humano 88/44/18 · Codex 27/109/14 · Kimi 20/124/6. Etapa B: humano 39/4/7 · Codex 37/13/0 · Kimi 31/17/2.
Interpretación declarada: los jueces LLM son **mutuamente reproducibles pero sistemáticamente más estrictos** que el humano (sesgo concentrado en claims multi-parte, meta-claims y fragmentos); la evidencia completa (5 chunks) hace converger los criterios. Conclusión: el juicio LLM no puede sustituir la referencia humana; se reporta como instrumento reproducible pero más estricto.

**Infraestructura experimental:**
- exp19b (selector anclado): veredicto con compuerta de replay bit-idéntica 5/5; HHEM +0,0451
- Sonda de suelo de ruido: 140 réplicas re-puntuadas, |Δ| media 0,0616, p90 0,2005, 23,3 % de pares exceden la banda ±0,081 dentro del mismo runtime
- Probe de cobertura de proveedor (evidencia exp17, descriptivo): baseline 8 % vs balanced 80 % cobertura estricta; 0/125 chunks de proveedores ajenos
- Suite: 341 tests CPU verdes; CI con lockfile y escáner de secretos con baseline

## Lo que te pido (responde en este orden, etiquetando cada hallazgo [BLOQUEANTE] / [RECOMENDABLE] / [OPCIONAL])

1. **Verificación de overclaiming**: con el encuadre y las cifras de arriba, ¿queda alguna afirmación que un revisor de LACCI pueda leer como sobre-interpretación? Indica la frase/afirmación exacta y la reformulación propuesta (máx. 1 h de trabajo cada una).
2. **El hallazgo triple-juez**: ¿lo presentarías como fortaleza (triangulación, reproducibilidad del instrumento) o es un riesgo (κ humano–LLM bajo podría leerse como "el gold es poco fiable")? ¿Cómo lo redactarías en 1 párrafo para la sección de validación humana?
3. **Coherencia interna**: revisa que las cifras reportadas no se contradigan entre sí (κ, ICs, flips, sensibilidad, retest). Señala cualquier inconsistencia aritmética o de encuadre.
4. **Veredicto final**: ¿el paquete es presentable para LACCI como estudio de medición, sí/no/con-condiciones? Si es "con-condiciones", lista SOLO condiciones ejecutables antes del 30 de agosto (nada de nuevos experimentos con GPU ni nuevas anotaciones humanas).
5. **Una sola cosa**: si solo pudieras exigir UN cambio antes de la entrega, ¿cuál sería y por qué?

Sé tan duro como en tu primera revisión. Prefiero un [BLOQUEANTE] incómodo ahora que una objeción del comité después.

---

*Fin del prompt. Guardado en `output/audit/PROMPT_CHATGPT_CRITICA_FINAL_2026-08-30.md` — [Kimi Work].*
