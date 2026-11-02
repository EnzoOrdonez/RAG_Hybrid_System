"""F-020b: the form-submit-button JS chunk fails or hangs; same sequence as b470d3e i7_browser.py:173-176."""
import json, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H
from scripts.study_operator.i7_browser import StreamlitSync

DUMMY = 'dummy-not-a-token'
def chunk_of(testid):
    import streamlit
    js = Path(streamlit.__file__).parent/'static'/'static'/'js'
    hits = [p.name for p in js.glob('*.js') if testid in p.read_text(encoding='utf-8', errors='ignore')]
    assert len(hits) == 1, hits
    return hits[0]

def main(out, server='app'):
    from playwright.sync_api import sync_playwright
    target = chunk_of('stFormSubmitButton')
    root = Path(tempfile.mkdtemp(prefix='f020c-'))
    config, assignments, store, token = H.make_store(root)
    proc, url = H.serve(root, config, assignments, server)
    rows = []
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(executable_path=H.CHROME, headless=True)
            for mode in ('none', 'abort', 'hang'):
                page = b.new_context().new_page()
                requests = []
                page.on('requestfailed', lambda r: requests.append({'failed': r.url.rsplit('/', 1)[-1], 'why': r.failure}))
                console = []
                page.on('console', lambda m: console.append(m.type + ': ' + m.text[:160]))
                if mode == 'abort':
                    page.route('**/' + target, lambda route: route.abort())
                elif mode == 'hang':
                    page.route('**/' + target, lambda route: None)     # never fulfilled: request stays pending
                sync = StreamlitSync(page, 25)
                page.set_default_timeout(20000)
                r = {'mode': mode, 'chunk': target}
                page.goto(url, wait_until='domcontentloaded')
                try:
                    sync.settled(); r['settled'] = True
                except Exception as e:
                    r['settled'] = str(e)
                try:
                    page.get_by_label('Invitación', exact=True).fill(DUMMY); r['fill'] = 'ok'
                except Exception as e:
                    r['fill'] = str(e).splitlines()[0][:120]
                try:
                    sync.until(lambda: sync.completed > 0 and sync.idle(), 3); r['settled_before_click'] = True
                except Exception as e:
                    r['settled_before_click'] = str(e)
                try:
                    page.get_by_role('button', name='Entrar', exact=True).click(timeout=8000); r['click'] = 'ok'
                except Exception as e:
                    msg = str(e)
                    r['click'] = msg.splitlines()[0][:120]
                    r['call_log'] = [l.strip() for l in msg.split('Call log:', 1)[-1].splitlines() if l.strip()][:5]
                    r['locator_resolved'] = 'locator resolved' in msg
                r['aria'] = page.locator('body').aria_snapshot(timeout=5000)
                r['console'] = console[:6]; r['requestfailed'] = requests[:4]
                r['exact_match_attempt7'] = (r.get('fill') == 'ok' and r.get('settled_before_click') is True
                                            and 'Timeout' in r.get('click', '') and r.get('locator_resolved') is False)
                rows.append(r)
                page.unroute_all(behavior='ignoreErrors')   # the hung route is never fulfilled
                page.context.close()
            b.close()
    finally:
        proc.terminate(); proc.wait(timeout=15)
    blob = json.dumps(rows, indent=1, ensure_ascii=False).replace(DUMMY, '<dummy>')
    assert token not in blob
    Path(out).write_text(blob, encoding='utf-8')
    for r in rows:
        print(r['mode'], '| settled', r['settled'], '| fill', r['fill'], '| pre-click settled', r['settled_before_click'],
              '| click', r['click'], '| resolved', r.get('locator_resolved'), '| MATCH' if r['exact_match_attempt7'] else '')
        print('   call_log', r.get('call_log')); print('   aria', r['aria'].replace('\n', ' / ')[:400])
        print('   console', r['console'][:3], r['requestfailed'][:2])

if __name__ == '__main__':
    main(sys.argv[1], *(sys.argv[2:3]))
