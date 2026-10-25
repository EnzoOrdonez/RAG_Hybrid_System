"""Browser-owned clock: synchronous inference must not prevent elapsed feedback."""
import json
import math
import time


def wait_html(elapsed):
    elapsed = float(elapsed)
    if not math.isfinite(elapsed):
        raise ValueError('Elapsed time must be finite')
    initial = json.dumps(max(0.0, elapsed))
    return '''<!doctype html><html lang="es"><meta charset="utf-8">
<style>body{font:16px system-ui;margin:4px;color:#31333f}
@media(prefers-color-scheme:dark){body{color:#fafafa}}</style>
<p>Buscando y verificando la respuesta. Tiempo transcurrido:
<span id="elapsed" role="timer" aria-live="off"></span>.</p>
<p id="notice" role="status" aria-live="polite"></p>
<script>
const initial = ''' + initial + ''';
const origin = performance.now();
const clock = document.getElementById('elapsed');
const notice = document.getElementById('notice');
function update() {
  const seconds = Math.floor(initial + (performance.now() - origin) / 1000);
  clock.textContent = String(Math.floor(seconds / 60)).padStart(2, '0') + ':' + String(seconds % 60).padStart(2, '0');
  const text = seconds >= 120
    ? 'La consulta continúa. Puedes esperar o avisar al coordinador. Recargar la página no cancela la consulta.'
    : seconds >= 60
      ? 'La consulta está tardando más de lo previsto. Sigue en curso; no necesitas enviarla otra vez.'
      : '';
  if (notice.textContent !== text) notice.textContent = text;
}
update();
setInterval(update, 1000);
</script></html>'''


def render_wait(started_at):
    import streamlit.components.v1 as components
    components.html(wait_html(time.time() - started_at), height=155)
