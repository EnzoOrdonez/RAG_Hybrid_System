"""F-020b: conditions under which get_by_role('button','Entrar') never resolves while get_by_label('Invitación').fill works.
Same sequence as b470d3e i7_browser.py:174-176 (goto, settled, fill, action->settled->click). Synthetic store, src/ untouched."""
import json, os, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H                       # docs/review/2026-11-02-i8/f020/harness.py (same server launcher)
from scripts.study_operator.i7_browser import StreamlitSync

DUMMY = 'dummy-not-a-token'
FORM = "document.querySelector('[data-testid=stForm]')"
BTN = "[...document.querySelectorAll('button')].find(b=>b.innerText.trim()==='Entrar')"
JS = {
 'baseline': None,
 'aria_hidden_form': f"{FORM}.setAttribute('aria-hidden','true')",
 'aria_hidden_button': f"{BTN}.closest('[data-testid=stFormSubmitButton]').setAttribute('aria-hidden','true')",
 'aria_hidden_app_root': "document.querySelector('[data-testid=stAppViewContainer]').setAttribute('aria-hidden','true')",
 'inert_form': f"{FORM}.inert=true",
 'button_display_none': f"{BTN}.style.display='none'",
 'button_visibility_hidden': f"{BTN}.style.visibility='hidden'",
 'button_removed': f"{BTN}.closest('[data-testid=stFormSubmitButton]').remove()",
 'name_lowercase': f"{BTN}.querySelector('p').textContent='entrar'",
 'name_extra_text': f"{BTN}.querySelector('p').textContent='Entrar →'",
 'name_aria_label': f"{BTN}.setAttribute('aria-label','Ingresar')",
 'name_translated': "document.documentElement.lang='en';" + f"{BTN}.querySelector('p').textContent='Enter'",
 'button_disabled': f"{BTN}.disabled=true",
 'role_presentation': f"{BTN}.setAttribute('role','presentation')",
}

def run(conditions, out):
    from playwright.sync_api import sync_playwright
    root = Path(tempfile.mkdtemp(prefix='f020b-'))
    config, assignments, store, token = H.make_store(root)
    proc, url = H.serve(root, config, assignments, 'app')
    results = []
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(executable_path=H.CHROME, headless=True)
            for name in conditions:
                page = b.new_context().new_page()
                sync = StreamlitSync(page, 25)
                page.set_default_timeout(4000)                       # 600000 in prod; shortened here
                r = {'condition': name}
                try:
                    page.goto(url, wait_until='domcontentloaded'); sync.settled()
                    if name == 'settings_dialog':                    # Streamlit's own modal
                        page.get_by_test_id('stMainMenu').click(); page.get_by_text('Settings', exact=True).click()
                        page.wait_for_timeout(500)
                    elif name == 'connection_error_dialog':
                        proc.terminate(); proc.wait(timeout=15); page.wait_for_timeout(8000)
                    elif JS[name]:
                        page.evaluate(JS[name])
                    try:
                        page.get_by_label('Invitación', exact=True).fill(DUMMY); r['fill'] = 'ok'
                    except Exception as e:
                        r['fill'] = type(e).__name__ + ': ' + str(e).splitlines()[0][:120]
                    try:
                        r['idle_before_click'] = sync.idle()
                        sync.until(lambda: sync.completed > 0 and sync.idle(), 3)   # what action() does first
                        r['settled_before_click'] = True
                    except Exception as e:
                        r['settled_before_click'] = str(e)
                    try:
                        page.get_by_role('button', name='Entrar', exact=True).click(timeout=4000)
                        r['click'] = 'ok'
                    except Exception as e:
                        msg = str(e).replace(DUMMY, '<dummy>')
                        r['click'] = msg.splitlines()[0][:120]
                        r['call_log'] = [l.strip() for l in msg.split('Call log:', 1)[-1].splitlines() if l.strip()][:6]
                        r['locator_resolved'] = 'locator resolved' in msg
                    r['aria'] = page.locator('body').aria_snapshot(timeout=4000).replace(DUMMY, '<dummy>')
                except Exception as e:
                    r['error'] = type(e).__name__ + ': ' + str(e)[:200]
                r['exact_match_attempt7'] = (r.get('fill') == 'ok' and r.get('settled_before_click') is True
                                            and 'Timeout' in r.get('click', '') and r.get('locator_resolved') is False)
                results.append(r)
                page.context.close()
            b.close()
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=15)
    blob = json.dumps(results, indent=1, ensure_ascii=False)
    assert token not in blob
    Path(out).write_text(blob, encoding='utf-8')
    return results

if __name__ == '__main__':
    names = sys.argv[2:] or list(JS) + ['settings_dialog']
    for r in run(names, sys.argv[1]):
        print(r['condition'], '| fill:', r.get('fill'), '| settled:', r.get('settled_before_click'), '| click:', r.get('click'),
              '| resolved:', r.get('locator_resolved'), '| MATCH' if r['exact_match_attempt7'] else '', r.get('error', ''))
