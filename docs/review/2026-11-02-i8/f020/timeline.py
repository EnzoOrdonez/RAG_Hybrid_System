"""After goto+settled (b470d3e i7_browser.py:173-174): when do the 'Invitación' field and the 'Entrar' button (by role) appear?"""
import json, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H
from scripts.study_operator.i7_browser import StreamlitSync

def main(out, loads=20, server='app'):
    from playwright.sync_api import sync_playwright
    root = Path(tempfile.mkdtemp(prefix='f020t-'))
    config, assignments, store, token = H.make_store(root)
    proc, url = H.serve(root, config, assignments, server)
    rows = []
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(executable_path=H.CHROME, headless=True)
            for i in range(loads):
                page = b.new_context().new_page()
                sync = StreamlitSync(page, 25)
                page.goto(url, wait_until='domcontentloaded'); sync.settled()
                t0 = time.monotonic(); seen = {}
                field = page.get_by_label('Invitación', exact=True)
                button = page.get_by_role('button', name='Entrar', exact=True)
                samples = []
                while time.monotonic() - t0 < 5:
                    f, bt = field.count(), button.count()
                    samples.append((round(time.monotonic()-t0, 3), f, bt))
                    if f and 'field' not in seen: seen['field'] = samples[-1][0]
                    if bt and 'button' not in seen: seen['button'] = samples[-1][0]
                    if len(seen) == 2: break
                    page.wait_for_timeout(5)
                first = samples[0]
                rows.append({'load': i, 'at_settled': {'field': first[1], 'button': first[2]}, 'first_seen_s': seen,
                             'completed': sync.completed, 'requests': sync.requests,
                             'field_without_button_window_s': (seen.get('button', 99) - seen.get('field', 0)) if seen.get('field') is not None else None})
                page.context.close()
            b.close()
    finally:
        proc.terminate(); proc.wait(timeout=15)
    Path(out).write_text(json.dumps(rows, indent=1), encoding='utf-8')
    for r in rows: print(r)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20, sys.argv[3] if len(sys.argv) > 3 else 'app')
