# Audit findings — Claude Code addenda

Hallazgos relacionados encontrados durante ejecución Tier 0 que NO están en `paper/audit_findings.md`. **No se corrigen sin aprobación explícita del usuario** (disciplina zero-trust bidireccional, mission spec §"Regla de oro").

---

## Fase 1 addenda

### A1 — `.get(..., 0)` silent zeros adicionales en `results_exporter.py`

El audit §21.5 Tier 0 (T0.7 / Flag 76) cita explícitamente las líneas **540** y **655** de `src/evaluation/results_exporter.py`. Un grep exhaustivo durante la verificación zero-trust muestra el mismo patrón en **9 líneas adicionales**:

```
241:                    matrix[i][j] = data[key].get("ret_ndcg@5_mean", 0)
287:            values = [data[method].get(m, 0) for m in metrics]
313:        ndcg5 = [data[c].get("ret_ndcg@5_mean", 0) for c in configs]
314:        recall5 = [data[c].get("ret_recall@5_mean", 0) for c in configs]
353:        values = [data.get(s, {}).get("ret_ndcg@5_mean", 0) for s in stages]
410:            if any(data[m].get(key, 0) > 0 for m in models):
425:            values = [data[model].get(m[0], 0) for m in available_metrics]
463:            values = np.array([data[c].get(key, 0) for c in configs])
495:            values = [data[config].get(m, 0) for m in metrics]
```

Cada una está en una figura distinta (`fig_statistical_heatmap`, `fig_method_comparison_radar`, distintos charts de exp7/exp5/etc.). Misma patología que Flag 76 / Flag 91: si la métrica falta, se fabrica `0` silenciosamente, produciendo barras/heatmaps en cero en lugar de crashear.

**Consecuencia si NO se corrigen**: tras aplicar el fix a 540/655, las regeneraciones en Fase 3 podrían aún producir figuras vacías en otros charts sin diagnóstico. El fix parcial pospone el problema.

**Recomendación (pendiente de aprobación)**: extender el fix a las 9 líneas adicionales, con mismo formato de `KeyError` diagnóstico. Effort estimado: 30-45 min. Ideal de aplicar en Fase 1 (mismo commit o commit separado `fix(flag-76-extended): raise on missing keys across all exporter figures`). **No lo aplico hasta que el usuario apruebe.**

### A2 — Menciones residuales de "normalization" en placeholders de main.tex

El plan Fase 1 paso 1.3 lista estos 6 puntos para renombrar a "query expansion": L68-69, L98, L145, L146, L199, L204. Todos aplicados en el commit `fix(flag-160,124)`.

Tras aplicar, un `grep -n -i 'normaliz' paper/overleaf_ready/main.tex` muestra 3 menciones residuales, todas dentro de `[PLACEHOLDER: ...]` que serán reescritos en Fase 5:

```
93 : ...benchmarks... do not evaluate cross-cloud terminology normalization.
122: ...Position the gap: none address cross-cloud terminology normalization.
134: Describe 5 stages: (1) ingestion + normalization, (2) hybrid retrieval...
```

- L93 y L122 son **afirmaciones de gap de literatura** (Related Work). Si el framing del paper cambia de "terminology normalization" a "cross-provider query expansion", el gap-statement debería reformularse en Fase 5 o bien declararse que el gap es sobre ambos conceptos.
- L134 es el **label del primer stage del pipeline** en el §III.A overview. Con la nueva narrativa (metadata tagging a nivel de índice, BM25 expansion a nivel query), el stage 1 debería llamarse "ingestion + dictionary-based tagging" o similar.

No se modifican en esta fase — son prose nueva que corresponde a Fase 5. **Pendiente de redacción por el usuario en Fase 5; yo solo flagging.**

### A3 — Abstract L63 también mencionaba "cross-cloud terminology normalization module"

Durante 1.3, al renombrar la prosa "Cross-cloud terminology normalization contributes..." (L68-69) observé que la misma oración en el abstract (L62-64) describía el módulo con el mismo compound: *"a cross-cloud terminology normalization module"*. Esa línea específica no estaba en la enumeración de 6 puntos del plan. La renombré igualmente a *"a dictionary-based cross-provider query expansion module"* porque es el mismo claim arquitectónico en la misma oración del abstract; dejar solo 5 renombres y uno coherente hubiera dejado el abstract internamente inconsistente.

Decisión documentada aquí por transparencia; si el usuario considera que fue fuera de alcance, revertir con `git revert` el commit `fix(flag-160,124)` o hacer patch puntual.

---
