# Rama `claude/i8-hardened`: revisión final (2026-11-05)

La rama parte de `3bb9ca8` y suma tres commits de código más este documento. Cada commit se puede adoptar o descartar por separado.

| Commit | Contenido | Respaldo |
|---|---|---|
| `3f12fd9` | 001: rama local antes de pagar. 002: `start('smoke')` dentro del `try`. 003: redacción del host sslip. 004: `diagnostics.json`. I8-09: peticiones fallidas y pendientes. | 8 tests; los 8 fallan sobre `3bb9ca8` |
| `fdc872a` | Solo transporte: `protocols h1 h2` en Caddy y `--disable-quic` en Chromium | **Sin evidencia**. 2 tests de configuración |
| `9ba4db7` | Plazo de 5 s para `observe()` dentro del volcado | 2 tests; sin el cambio, el principal falla con `assert 30.02 < 10` |

Tests: `tests/test_i8_hardened.py`. Las tres capas de `init.sh` dan exit 0 sin `PLAYWRIGHT_BROWSERS_PATH` (2134 pasan y 54 se omiten en `fdc872a`).

## Validación del Caddyfile [EJECUTADO]

- **Binario:** Caddy oficial v2.10.2, con checksum verificado.
- **Versión de producción:** el repo no la fija. `caddy_image` es un digest que vive en la config local del operador, así que **no está verificado** que la versión mayor coincida.
- **Entrada:** `caddyfile('203.0.113.7.sslip.io')`. Es una dirección de documentación (TEST-NET-3), no la de una VM.

| Fuente | `caddy validate --adapter caddyfile` | `protocols` en `caddy adapt` |
|---|---|---|
| `fdc872a` | exit 0, `Valid configuration` | `['h1', 'h2']` |
| `3bb9ca8` (control) | exit 0, `Valid configuration` | `None` (valores por defecto, con h3) |

Las dos salidas incluyen el aviso `Caddyfile input is not formatted`, que es cosmético.

## Experimento HTTP/3 con UDP/443 bloqueado [EJECUTADO, NO REPRODUCE]

- **Montaje:** Caddy 2.10.2 con TLS local y Chromium 1243 (Playwright 1.58).
  - La página hace tres `import()` tardíos de 60 KB.
  - UDP/443 está bloqueado con DROP en iptables.
  - Cada variante son 30 cargas con un contexto nuevo cada vez.
- **Resultado:** (a) por defecto, (b) `--disable-quic` y (c) Caddy con `protocols h1 h2` dan los tres 0/90 fallos, con `nextHopProtocol` h2 en 90/90.
- **Por qué no es concluyente:** Chromium no adoptó h3 por Alt-Svc en ninguna variante. Solo lo usó forzando QUIC con `--origin-to-force-quic-on`. El experimento no confirma ni descarta HTTP/3 como causa de F-020.
- **Hipótesis no verificada:** Chromium solo adopta h3 por Alt-Svc con una raíz pública, como el certificado de Let's Encrypt de producción.

## Revisión de `observe()`

`observe()` del smoke (`i7_smoke.py`) llama a `operator.bridge(..., private=True)` (`lifecycle.py:48`): un `gcloud compute ssh` por IAP con `timeout=180`.

- **Windows:** la llamada principal pasa por `windows_ssh.native_run`. Hace `communicate(timeout)` y, si vence, mata el árbol de procesos con `taskkill /T /F` y espera 15 s más.
- **Cota:** unos 210 s por llamada. El volcado de fallo podía quedarse esperando ese tiempo; `9ba4db7` lo limita a 5 s.
- **Riesgo residual, no verificado:** los pasos previos de `gcloud` usan `subprocess.run`, que en Windows llama a `communicate()` sin timeout después de matar el proceso. Si un proceso hijo sigue reteniendo el pipe, la llamada podría no volver.

## Qué exige adoptar la rama (inferido del código, no ejecutado)

1. **Publicar la fuente:** los commits tienen que estar en `fix/interview-readiness` y `config['commit']` tiene que apuntar a ellos. El guard 001 y la comprobación `remote == HEAD` lo exigen.
2. **Imagen nueva:** el build de CPU clona esa rama, así que hace falta un `image_id` nuevo. El Caddyfile se genera en el host a partir del checkout (`host_runtime.py:246`), por lo que `fdc872a` solo llega a la VM con la fuente nueva.
3. **Recibo del navegador nuevo:** `i7_gate_guest.py:31` e `i7_phase_b.py:127` fijan el sha256 de `i7_browser.py`, y los tres commits lo cambian.
4. **Readiness y smoke nuevos** para la identidad nueva de commit e imagen.
5. **Raíz de intento nueva:** si `operator.root/cpu` ya existe, el build rechaza con `CPU_ATTEMPT_EXISTS_RECONCILE_WITHOUT_REPLAY`.
6. **Descartar el transporte:** basta con `git revert fdc872a`, que es independiente de los otros commits.

## No verificado

- La VM real, Let's Encrypt y QUIC por Alt-Svc con una raíz pública.
- Chromium 1208 en Windows y la ruta de Internet desde Windows.
- Si `fdc872a` cambia algo en F-020.
- Que la versión mayor de Caddy coincida con la de `caddy_image`.
