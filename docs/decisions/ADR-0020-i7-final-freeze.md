# ADR-0020: Ajustes finales y pendientes de I7

[DECLARADO] Aceptado el 2026-10-28 para cierre local de F-020 sobre e2335ed.
Complementa [ADR-0019](ADR-0019-i7-ledger-corrections.md), que se conserva como
registro histórico. No autoriza publicación, despliegue ni gasto.

## Decisión y comprobación local

[DECLARADO] H3 exige approved_b4_version exactamente igual a
`nombre-del-directorio@fingerprint-del-protocolo-verificado`. No cambia el
esquema de ethics_approval.json; los registros con solo el nombre bloquean study.
La autenticidad y autoridad del PDF siguen siendo un atestado humano de Enzo.

[VERIFICADO] Se prueban coincidencia, dos sellos válidos con igual nombre y
fingerprints distintos, y aprobación sin fingerprint. Reducir la comparación
al nombre causa 2 fallos; retirarla causa 5. Ambos mutantes se restauraron byte
por byte; después pasan 87 tests de policy/human.

[DECLARADO] H9 usa NO_PAID_EFFECT_BEFORE_2026_10_28 y fecha ambos mensajes CLI
del piso como 2026-10-28 UTC. La autorización independiente de readiness permanece.

[VERIFICADO] El positivo integrado ejecuta start('study') con ética sintética
válida, GO, fase humana y reserva. Se conservan política y generación del startup
reales; solo se simula el proveedor. Verifica STARTED_NOT_READY, perfil standard,
estado persistido y llamadas de arranque. Los seis archivos de tests de la matriz
de revisión pasan: 159 casos, con `-o addopts= -q -p no:cacheprovider`.

## Fe de erratas de los conteos de ADR-0019

[VERIFICADO por revisión independiente] REPORTE_REVISION2.md sobre e2335ed
documenta al menos 26 mutantes detectados y restaurados, con 154 tests de base.
Fuente local: `C:/CloudRAG/iteration7-revision-cruzada-20261027/REPORTE_REVISION2.md`,
secciones «Mutación por barrera» y «ADR-0019 frente al código».

[DECLARADO] Los conteos históricos dependen de la suite y de la definición del
mutante. Para el árbol revisado se sustituyen las cifras discrepantes de ADR-0019
por estos límites inferiores; el orden es el de la matriz del reporte:

| Barrera | Fallos por mutante en e2335ed |
|---|---|
| F5: base, tarifa guard, tarifa reconcile | al menos 5 / al menos 1 / al menos 1 |
| F6: rehearsal, pilot | al menos 20 / al menos 21 |
| F9: límite por TTL | al menos 4 |
| F2: extra, SKU, prospectivo, durable, retención, validar extra, validar contador | al menos 2 / al menos 7 / al menos 1 / al menos 6 / al menos 5 / al menos 5 / al menos 3 |
| F7: etiqueta, fingerprint, sello, llamada I7, PDF | al menos 3 / al menos 1 / al menos 1 / al menos 2 / al menos 1 |

No son nuevas ejecuciones propias de los 26 mutantes ni conteos del árbol final.
Las dos mutaciones H3 descritas arriba son verificaciones nuevas separadas.

## Backlog aceptado, sin correcciones en este cierre

| Hallazgo | Severidad | Pendiente |
|---|---|---|
| H1 | Medio, latente | Reserva sin cost_basis bloquea guard y cierre por código. |
| H2 | Bajo, latente | UNKNOWN antiguo sin base bloquea reconcile y reserve. |
| H4 | Bajo | Exposición extra de Fase B arrastrada al ledger humano; sobrecuenta. |
| H5 | Bajo, observación | Sin tope acumulado de horas GPU humanas; rige el ledger USD 70. |
| H6 | Bajo | Plan de costos malformado falla cerrado con excepciones no codificadas. |
| H7 | Bajo, sin cambio | Escrituras GCS sin admisión monetaria por escritura. |
| H8 | Bajo | Guarda heredada por método, dependiente de installation.json. |
| P7 | Alto para la exclusividad global afirmada | Existen entradas GCP fuera del runbook; inventario en REPORTE.md, mantenido por REPORTE_REVISION2.md. |

[DECLARADO] F-020 sigue active, passes=false. El cierre exige init.sh 0/0/0
después del commit documental, con hash exacto y diff --stat contra e2335ed en el
traspaso final. No equivale a GO real ni a aprobación ética real.
