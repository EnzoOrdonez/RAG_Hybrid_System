"""Automated participant UI; authoritative acknowledgements come from private app state."""
import base64
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from scripts.study_operator.stimulus_host_evidence import sha

PIN=Path(__file__).resolve().parents[2]/'config/i7-browser-pin.json'
DUMP_TIMEOUT_MS=5000
BUTTON_WAIT_S=90   # A normal button appears within one rerun; 'Probar consulta' passes 900 s explicitly.
TEXT_FIELDS='input:not([type=radio]):not([type=checkbox]):not([type=button]):not([type=submit]), textarea'
# ARIA snapshots print field values, password included: redact the value and any child lines.
VALUE_ROLE=re.compile(r'''^(\s*- '?(?:textbox|searchbox|combobox|spinbutton|slider)(?: "(?:[^"\\]|\\.)*")?(?: \[[^\]]*\])*'?)(?::.*)?$''')


def runtime_identity(playwright):
    import playwright as package
    pin=json.loads(PIN.read_bytes())
    root=Path(package.__file__).parent
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    executable=Path(playwright.chromium.executable_path)
    platform='win32' if sys.platform=='win32' else 'linux'
    if (importlib.metadata.version('playwright')!=pin['version']
            or sha(files)!=pin['platforms'][platform]['package_tree_sha256']
            or hashlib.sha256(executable.read_bytes()).hexdigest()!=pin['platforms'][platform]['browser_sha256']):
        raise ValueError('PINNED_BROWSER_RUNTIME_DIFFERS')
    return {'version':pin['version'],'pin_sha256':hashlib.sha256(PIN.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'browser_sha256':pin['platforms'][platform]['browser_sha256']}


class StreamlitSync:
    """One UI action, one observed rerun; form edits never submit implicitly."""
    def __init__(self,page,timeout_s):
        self.page,self.timeout_s=page,timeout_s
        self.requests=self.completed=self.form_text_entries=0
        self.error=None
        page.on('websocket',self.socket)

    def socket(self,socket):
        if '/_stcore/stream' not in socket.url:
            return
        socket.on('framesent',self.sent)
        socket.on('framereceived',self.received)
        socket.on('close',lambda:self.failed('STREAMLIT_SOCKET_CLOSED'))
        socket.on('socketerror',lambda error:self.failed('STREAMLIT_SOCKET_ERROR'))

    def failed(self,reason):
        self.error=reason

    def sent(self,payload):
        from streamlit.proto.BackMsg_pb2 import BackMsg
        if isinstance(payload,bytes) and BackMsg.FromString(payload).WhichOneof('type')=='rerun_script':
            self.requests+=1

    def received(self,payload):
        from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
        if not isinstance(payload,bytes):
            return
        message=ForwardMsg.FromString(payload)
        if message.WhichOneof('type')=='script_finished':
            if message.script_finished==ForwardMsg.FINISHED_SUCCESSFULLY:
                self.completed+=1
            elif message.script_finished==ForwardMsg.FINISHED_WITH_COMPILE_ERROR:
                self.failed('STREAMLIT_SCRIPT_FAILED')

    def idle(self):
        return self.page.locator('[data-testid=stApp][data-test-script-state=notRunning][data-test-connection-state=CONNECTED]').count()==1

    def until(self,predicate,timeout_s=None,code='STREAMLIT_RERUN_NOT_ACKNOWLEDGED'):
        end=time.monotonic()+(self.timeout_s if timeout_s is None else timeout_s)
        while time.monotonic()<end:
            if self.error:
                raise ValueError(self.error)
            if predicate():
                return
            self.page.wait_for_timeout(20)
        raise ValueError(code)

    def settled(self):
        self.until(lambda:self.completed>0 and self.idle())

    def action(self,perform,timeout_s=None):
        self.settled()
        sent,completed=self.requests,self.completed
        perform()
        self.until(lambda:self.requests>sent and self.completed>completed and self.idle(),timeout_s)

    def option(self,group,value):
        self.settled()
        group.get_by_text(str(value),exact=True).click()
        if not group.get_by_role('radio',name=str(value),exact=True).is_checked():
            raise ValueError('APP_VALUE_NOT_REGISTERED')

    def text(self,label,value):
        self.settled()
        control=self.page.get_by_label(label,exact=True)
        form=control.evaluate("el => !!el.closest('[data-testid=stForm]')")
        if form:
            control.fill(value)
            control.blur()
            self.form_text_entries+=1
        else:
            self.action(lambda:(control.fill(value),control.press('Control+Enter')))
        if control.input_value()!=value:
            raise ValueError('APP_VALUE_NOT_REGISTERED')


def visible(page,sync,label,timeout_s=None):
    """Exact-name button, polled so a closed Streamlit socket fails now, not at the click budget."""
    button=page.get_by_role('button',name=label,exact=True)
    wait=min(sync.timeout_s,BUTTON_WAIT_S) if timeout_s is None else timeout_s
    sync.until(button.is_visible,wait,'BUTTON_NOT_VISIBLE')
    return button


def redact_aria(text):
    lines,depth=[],None
    for line in text.splitlines():
        indent=len(line)-len(line.lstrip())
        if depth is not None and indent>depth:
            lines.append(' '*indent+'- <redacted>')
            continue
        match=VALUE_ROLE.match(line)
        depth=indent if match else None
        lines.append(match.group(1)+': <redacted>' if match else line)
    return '\n'.join(lines)


def scrub(value,secrets):
    if isinstance(value,str):
        for secret in secrets:
            value=value.replace(secret,'<redacted>')
        return value
    if isinstance(value,dict):
        return {k:scrub(v,secrets) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [scrub(v,secrets) for v in value]
    return value


def dump_failure(page,root,private,events,error):
    """Owner-only diagnostics. Token, canary IP/UA and field values are redacted; dump errors never escape."""
    try:
        folder=Path(root)/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        folder.parent.mkdir(parents=True,exist_ok=True)
        folder.mkdir(exist_ok=False)
        page.set_default_timeout(DUMP_TIMEOUT_MS)
    except Exception:
        return None
    failed,secrets=[],set(private)
    try:
        fields=page.locator(TEXT_FIELDS).all()
        for value in (field.input_value() for field in fields):
            secrets|={value,' '.join(value.split())} if value else set()
    except Exception:
        fields=[]
        failed.append('input_values')
    secrets=sorted(filter(None,secrets),key=len,reverse=True)
    def save(name,value):
        (folder/name).write_text(value if isinstance(value,str) else
            json.dumps(scrub(value,secrets),ensure_ascii=False,indent=1),encoding='utf-8')
    def controls():
        return {'path':urlsplit(page.url).path,
            'buttons':[{'text':b.inner_text(),'disabled':b.is_disabled()} for b in page.get_by_role('button').all() if b.is_visible()],
            'labels':page.locator('label').all_inner_texts(),
            'inputs':[{'type':f.get_attribute('type'),'aria_label':f.get_attribute('aria-label'),
                       'disabled':f.is_disabled()} for f in fields]}
    for name,produce in (('screenshot.png',lambda:page.screenshot(path=str(folder/'screenshot.png'),full_page=True)),
            ('aria.yaml',lambda:save('aria.yaml',scrub(redact_aria(page.locator('body').aria_snapshot()),secrets))),
            ('controls.json',lambda:save('controls.json',controls())),
            ('console.json',lambda:save('console.json',dict(events,sync_error=error)))):
        try:
            produce()
        except Exception:
            failed.append(name)
    if failed:
        save('dump-errors.json',{'failed':failed})
    return folder


def run(url, token, protocol, identity, observe, *, local_test=False, user_agent=None, ip=None,
        participant='P999', failure_dir=None):
    from playwright.sync_api import sync_playwright
    parsed=urlsplit(url)
    if parsed.scheme!='https' or (local_test and parsed.hostname not in ('localhost','127.0.0.1')):
        raise ValueError('HTTPS_BROWSER_REQUIRED')
    config=protocol['config']
    confirmations=[]
    with sync_playwright() as pw:
        version=runtime_identity(pw)
        browser=pw.chromium.launch(headless=True,executable_path=pw.chromium.executable_path)
        page=sync=None
        events={'console':[],'pageerror':[],'websockets':[],'websocket_closed':[]}
        try:
            headers={'X-Forwarded-For':ip,'X-Real-IP':ip,'Forwarded':'for='+ip} if ip else {}
            # Browser WebSocket APIs do not forward arbitrary HTTP headers.
            # Carry the IP canary in the UA too; Caddy strips this entire header.
            wire_ua=(user_agent+' CloudRAG-IP/'+ip) if user_agent and ip else user_agent
            context=browser.new_context(ignore_https_errors=local_test,user_agent=wire_ua,extra_http_headers=headers)
            page=context.new_page()
            page.on('console',lambda message:events['console'].append({'type':message.type,'text':message.text}))
            page.on('pageerror',lambda error:events['pageerror'].append(str(error)))
            def watch(ws):
                events['websockets'].append(ws.url)
                ws.on('close',lambda *_:events['websocket_closed'].append(ws.url))
            page.on('websocket',watch)
            sync=StreamlitSync(page,25 if local_test else 600)
            transport={'browser_context':bool(user_agent and ip),'websocket_ip':False,'websocket_user_agent':False}
            if user_agent and ip:
                cdp=context.new_cdp_session(page)
                cdp.send('Network.enable')
                cdp.send('Network.setExtraHTTPHeaders',{'headers':headers})
                def handshake(event):
                    values={k.lower():v for k,v in event['request']['headers'].items()}
                    transport['websocket_ip'] |= ip in values.get('user-agent','')
                    transport['websocket_user_agent'] |= values.get('user-agent')==wire_ua
                cdp.on('Network.webSocketWillSendHandshakeRequest',handshake)
            page.set_default_timeout(20000 if local_test else 60000)
            sockets=[]
            page.on('websocket',lambda ws:sockets.append(ws.url))
            def snapshot():
                for attempt in range(20):
                    try:
                        result=observe()
                        break
                    except PermissionError:
                        # Windows may temporarily deny a read during atomic replacement.
                        if attempt==19:
                            raise ValueError('APP_STATE_READ_UNAVAILABLE') from None
                        page.wait_for_timeout(50)
                if result.get('identity')!=identity:
                    raise ValueError('BROWSER_BOOT_IDENTITY_DIFFERS')
                return result
            def wait(predicate):
                end=time.monotonic()+(25 if local_test else 600)
                while time.monotonic()<end:
                    result=snapshot()
                    if result.get('data') and predicate(result['data']):
                        return result
                    page.wait_for_timeout(50)
                raise ValueError('APP_VALUE_NOT_REGISTERED')
            def confirm(actual,expected):
                if actual!=expected:
                    raise ValueError('APP_VALUE_NOT_REGISTERED')
                confirmations.append(sha(expected))
            def click(label):
                timeout=900000 if label=='Probar consulta' and not local_test else None
                budget=900 if timeout else None
                sync.action(lambda:visible(page,sync,label,budget).click(timeout=timeout),timeout_s=budget)
            def text(label,value):
                sync.text(label,value)
            snapshot()
            page.goto(url,wait_until='domcontentloaded')
            sync.settled()
            page.get_by_label('Invitación',exact=True).fill(token)
            click('Entrar')
            first=wait(lambda d:d['stage']=='familiarization')['data']
            session_id=first['session_id']
            confirm(first['assignment']['participant_id'],participant)
            click('Preparar sesión')
            for block in (0,1):
                click('Probar consulta')
                click('Comenzar tareas')
                wait(lambda d:d['stage']=='tasks' and d['block_index']==block)
                for task in range(4):
                    if task==3:
                        question=f'Synthetic cloud verification question {block+1}?'
                        text('Tu consulta',question)
                    else:
                        block_data=snapshot()['data']
                        from src.ui.components.study_protocol import CELLS
                        task_set=CELLS[block_data['assignment']['cell']][block][1]
                        qid=config['tasks'][task_set][task]
                        question=protocol['queries'][qid]['question']
                    click('Consultar')
                    visible(page,sync,'Continuar')
                    attempt=wait(lambda d:len(d['attempts'])==block*4+task+1)['data']['attempts'][-1]
                    confirm(attempt['question'],question)
                    confirm(attempt['status'],'success')
                    click('Continuar')
                    wait(lambda d: d['attempts'][-1]['status']=='acknowledged')
                visible(page,sync,'Guardar respuestas del bloque')
                sus=[1+i%5 for i in range(10)]
                ueq=[1+i%7 for i in range(8)]
                likert={item['id']:1+i%5 for i,item in enumerate(config['likert'])}
                values=sus+ueq+list(likert.values())
                groups=page.get_by_test_id('stRadio')
                groups.nth(len(values)-1).wait_for(state='visible')
                if groups.count()!=len(values):
                    raise ValueError('FULL_INSTRUMENT_UI_REQUIRED')
                for i,value in enumerate(values):
                    sync.option(groups.nth(i),value)
                click('Guardar respuestas del bloque')
                data=wait(lambda d:len(d['instruments'])==block+1)['data']['instruments'][-1]
                confirm(data['sus'],sus)
                confirm(data['ueq_s'],ueq)
                confirm(data['likert'],likert)
            comparative={}
            for i,item in enumerate(config['comparative']):
                if item['choices']:
                    value=item['choices'][0]
                    sync.option(page.get_by_test_id('stRadio').nth(i),value)
                else:
                    value='Synthetic comparison comment.'
                    text(item['text'],value)
                comparative[item['id']]=value
            click('Continuar al cierre')
            confirm(wait(lambda d:d['stage']=='blinding')['data']['comparative'],comparative)
            choice=config['blinding_choices'][0]
            sync.option(page.get_by_test_id('stRadio'),choice)
            reason='Synthetic closing reason.'
            text(config['blinding_reason'],reason)
            click('Finalizar')
            final=wait(lambda d:d['stage']=='complete')
            confirm(final['data']['blinding']['choice'],choice)
            confirm(final['data']['blinding']['reason'],reason)
            deadline=time.monotonic()+(25 if local_test else 600)
            while 'export_base64' not in final:
                if time.monotonic()>=deadline:
                    raise ValueError('BROWSER_EXPORT_NOT_BACKED_UP')
                page.wait_for_timeout(50)
                final=snapshot()
            raw=base64.b64decode(final['export_base64'],validate=True)
            exported=json.loads(raw)
            digest=hashlib.sha256(raw).hexdigest()
            if (exported['session_id']!=session_id or exported['stage']!='complete'
                    or len(exported['attempts'])!=8 or len(exported['instruments'])!=2
                    or digest!=final['export_sha256'] or digest!=final['backup']['sha256']
                    or final['backup']['status']!='complete' or not sockets):
                raise ValueError('BROWSER_EXPORT_INCOMPLETE_OR_CHANGED')
            return {'status':'BROWSER_EXPORT_VERIFIED','identity':identity,'session_id':session_id,
                'automator':version,'export_sha256':digest,'attempts':8,'instruments':2,
                'confirmed_inputs_sha256':confirmations,'websocket_seen':True,'local_validation':local_test,
                'canary_transport':transport,'ui_sync':{'requests':sync.requests,'completed':sync.completed,
                    'form_text_entries':sync.form_text_entries},
                'objects':final['backup']['objects']}
        except Exception:
            if failure_dir and page is not None:
                dump_failure(page,failure_dir,(token,ip,user_agent,wire_ua),events,getattr(sync,'error',None))
            raise
        finally:
            browser.close()
