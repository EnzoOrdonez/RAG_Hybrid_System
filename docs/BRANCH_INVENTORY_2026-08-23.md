# Inventario de ramas — refrescado el 2026-08-30

Instantánea de solo lectura tomada después del `fetch` autorizado, con
`summer/taxonomia-759` en `50c8ff3` como base y `origin/main` en `670f8e5` como rama
principal pública. Los conteos `adelante` y `atrás` son relativos a esa punta exacta de
la base. «Mergeada» significa que la punta de la fila es ancestro de la rama indicada.

Las filas `origin/*` son **referencias locales de seguimiento remoto**, no ramas remotas
por sí mismas. El `fetch` actualizó las ramas presentes, pero no usó `--prune`: las tres
refs marcadas «obsoleta» corresponden a ramas remotas cuya eliminación fue registrada el
2026-08-23. `origin/HEAD -> origin/main` es una ref simbólica y no se cuenta como rama.
Ninguna clasificación ejecuta una acción; todas quedan sujetas a decisión humana.

| Ref | Tipo / estado remoto | Último commit | Fecha | Mergeada a taxonomía | Mergeada a `origin/main` | Adelante | Atrás | Propuesta | Razón |
|---|---|---:|---:|:---:|:---:|---:|---:|---|---|
| `codex/plan-a-thesis-safe` | local | `42b95ba` | 2026-03-06 | sí | sí | 0 | 220 | candidata a archivar | Hito antiguo totalmente integrado. |
| `fase-3-regenerate-figures` | local | `f6d345b` | 2026-04-23 | no | no | 3 | 194 | candidata a archivar | Conserva 3 commits no integrados; requiere revisión humana antes de cualquier retiro. |
| `fase-3.5-nli-recompute-saved-answers` | local | `e8d2e2e` | 2026-04-24 | no | no | 9 | 194 | candidata a archivar | Conserva 9 commits no integrados; requiere revisión humana antes de cualquier retiro. |
| `main` | local | `670f8e5` | 2026-08-30 | no | sí | 3 | 6 | activa | Rama principal local sincronizada con el merge camera-ready; diverge por trabajo posterior de esta rama. |
| `pre-corpus-rebuild-2026-05-21` | local | `66cd0cf` | 2026-06-30 | sí | sí | 0 | 147 | candidata a archivar | Hito previo al rebuild, totalmente integrado. |
| `summer/ablacion` | local | `3b60ed3` | 2026-07-24 | sí | sí | 0 | 107 | candidata a archivar | Punta integrada en ambas líneas vigentes. |
| `summer/exp19b` | local | `cea75a1` | 2026-08-21 | sí | sí | 0 | 48 | candidata a archivar | Hito de exp19b integrado; útil para trazabilidad. |
| `summer/mejoras` | local | `89dc654` | 2026-08-04 | sí | sí | 0 | 62 | candidata a archivar | Punta integrada en ambas líneas vigentes. |
| `summer/taxonomia-759` | local | `50c8ff3` | 2026-08-30 | sí | no | 0 | 0 | activa | Rama de trabajo de esta instantánea. |
| `origin/fase-2.5-recompute-retrieval-stats` | seguimiento remoto obsoleto; rama remota borrada | `7f9eede` | 2026-04-23 | sí | sí | 0 | 198 | candidata a borrar | Solo queda la ref local obsoleta; la rama remota fue eliminada el 2026-08-23. |
| `origin/fase-3-regenerate-figures` | seguimiento de rama remota | `f6d345b` | 2026-04-23 | no | no | 3 | 194 | candidata a archivar | Conserva 3 commits remotos no integrados. |
| `origin/fase-3.5-nli-recompute-saved-answers` | seguimiento de rama remota | `c119c99` | 2026-04-23 | no | no | 3 | 194 | candidata a archivar | Conserva 3 commits remotos no integrados; además difiere de la rama local homónima. |
| `origin/fix/phase-1-no-rerun` | seguimiento remoto obsoleto; rama remota borrada | `270ee58` | 2026-04-23 | sí | sí | 0 | 209 | candidata a borrar | Solo queda la ref local obsoleta; la rama remota fue eliminada el 2026-08-23. |
| `origin/fix/phase-2-nli-and-seeds` | seguimiento remoto obsoleto; rama remota borrada | `e6a599b` | 2026-04-23 | sí | sí | 0 | 203 | candidata a borrar | Solo queda la ref local obsoleta; la rama remota fue eliminada el 2026-08-23. |
| `origin/main` | seguimiento de rama remota | `670f8e5` | 2026-08-30 | no | sí | 3 | 6 | activa | Ref pública actual; contiene el merge camera-ready PR #3. |
| `origin/pre-corpus-rebuild-2026-05-21` | seguimiento de rama remota | `b5c597e` | 2026-06-11 | sí | sí | 0 | 153 | candidata a archivar | Hito remoto anterior y totalmente integrado. |
| `origin/summer/exp19b` | seguimiento de rama remota | `cea75a1` | 2026-08-21 | sí | sí | 0 | 48 | candidata a archivar | Hito remoto de exp19b integrado. |
| `origin/summer/mejoras` | seguimiento de rama remota | `89dc654` | 2026-08-04 | sí | sí | 0 | 62 | candidata a archivar | Punta remota integrada. |
| `origin/summer/taxonomia-759` | seguimiento de rama remota | `293cc0e` | 2026-08-30 | sí | no | 0 | 4 | activa | Ref remota de la rama de trabajo, cuatro commits detrás de la instantánea local. |

## Lectura propuesta

- Mantener activas `main`, `summer/taxonomia-759` y sus refs de seguimiento.
- Conservar como archivo los hitos integrados y revisar manualmente las dos familias
  `fase-3*`, porque guardan commits no integrados.
- Considerar retirar únicamente las tres refs de seguimiento obsoletas después de una
  decisión humana. Este inventario no borró ramas ni refs locales o remotas.
