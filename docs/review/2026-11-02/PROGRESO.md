# Progreso (reanudable)
- [x] Fase 0 base verificada (b470d3e).
- [x] Línea base: capa 1 exit 0, capa 3 exit 0; capa 2 exit 1 por 11 fallos de entorno (R-08) con `PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers`; sin esa variable 2159 tests, 0 fallos, 58 omitidos/xfail.
- [x] P1 F-01 period_id tras smoke: reproducido, parche 001 verificado en worktree.
- [x] P5 re-validación R-01 (refutado), R-02 (obsoleto + residual documentado), R-03/04/05/06/09/15/16 (confirmados).
- [x] P4 seguridad (subagente C): C-1, C-2 S2 (= R-10, R-14), C-3, C-4 S3.
- [x] Ruff F/E9/B en alcance: 15 avisos, 0 defectos reales.
- [ ] P2 dinero/TTL (subagente A) en curso.
- [ ] P3 recuperación (subagente B) en curso.
- [ ] P6 refutación adversarial de F-01 en curso; pendientes los S0/S1 de A y B.
