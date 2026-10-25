"""User-visible browser actions wait for one acknowledged Streamlit rerun."""
import pytest

from tests.i7_browser_guard import require_i7_browser
from tests.test_study_i7_browser import local_app


def test_form_text_does_not_submit_before_explicit_button(tmp_path,monkeypatch):
    require_i7_browser()
    from playwright.sync_api import Locator
    from scripts.study_operator.i7_browser import run
    original=Locator.press
    def press(control,key,**kwargs):
        if key=='Control+Enter':
            assert not control.evaluate("el => !!el.closest('[data-testid=stForm]')"), 'FORM_SUBMITTED_BEFORE_EXPLICIT_BUTTON'
        return original(control,key,**kwargs)
    monkeypatch.setattr(Locator,'press',press)
    with local_app(tmp_path) as (url,token,protocol,identity,observe):
        result=run(url,token,protocol,identity,observe,local_test=True)
    assert result['status']=='BROWSER_EXPORT_VERIFIED'


# Delay delivery of successful script-finished messages to the actual frontend.
# Keep all later messages queued, preserving WebSocket order. No app source changes.
DELAY_FINISH = r"""
(() => {
  window.__i7DelayedFinishes = 0;
  function finished(data) {
    if (!(data instanceof ArrayBuffer)) return false;
    const b = new Uint8Array(data); let p = 0;
    function integer() {
      let n = 0, scale = 1, x;
      do { x = b[p++]; n += (x & 127) * scale; scale *= 128; } while (x & 128);
      return n;
    }
    while (p < b.length) {
      const tag = integer(), field = Math.floor(tag / 8), wire = tag % 8;
      if (wire === 0) { const n = integer(); if (field === 6 && n === 0) return true; }
      else if (wire === 1) p += 8;
      else if (wire === 2) { const size = integer(); p += size; }
      else if (wire === 5) p += 4;
      else return false;
    }
    return false;
  }
  function delayed(listener) {
    let busy = false; const queue = [];
    function deliver(target,event) {
      if (typeof listener === 'function') listener.call(target,event);
      else listener.handleEvent(event);
    }
    function drain() {
      if (busy || !queue.length) return;
      const [target,event] = queue.shift();
      if (finished(event.data)) {
        busy = true; window.__i7DelayedFinishes++;
        setTimeout(() => { deliver(target,event); busy = false; drain(); }, 350);
      } else { deliver(target,event); drain(); }
    }
    return function(event) { queue.push([this,event]); drain(); };
  }
  const add = WebSocket.prototype.addEventListener;
  WebSocket.prototype.addEventListener = function(type,listener,options) {
    return add.call(this,type,type === 'message' ? delayed(listener) : listener,options);
  };
  const property = Object.getOwnPropertyDescriptor(WebSocket.prototype,'onmessage');
  Object.defineProperty(WebSocket.prototype,'onmessage', {
    ...property, set(listener) { property.set.call(this,listener ? delayed(listener) : listener); }
  });
})();
"""


def test_edits_wait_for_finished_rerun_and_rendered_app(tmp_path,monkeypatch):
    require_i7_browser()
    from playwright.sync_api import BrowserContext,Locator
    from scripts.study_operator.i7_browser import run
    original_page,original_fill=BrowserContext.new_page,Locator.fill
    observed=[]
    def new_page(context,*args,**kwargs):
        page=original_page(context,*args,**kwargs)
        page.add_init_script(DELAY_FINISH)
        return page
    def fill(control,value,**kwargs):
        control.wait_for(state='visible')
        state=control.page.get_by_test_id('stApp').get_attribute('data-test-script-state')
        observed.append(control.page.evaluate('window.__i7DelayedFinishes'))
        assert state=='notRunning','ACTION_BEFORE_STREAMLIT_RERUN_FINISHED'
        return original_fill(control,value,**kwargs)
    monkeypatch.setattr(BrowserContext,'new_page',new_page)
    monkeypatch.setattr(Locator,'fill',fill)
    with local_app(tmp_path) as (url,token,protocol,identity,observe):
        result=run(url,token,protocol,identity,observe,local_test=True)
    assert observed and max(observed)>0
    assert result['status']=='BROWSER_EXPORT_VERIFIED'


@pytest.mark.parametrize('missing',['request','finished','render'])
def test_missing_ack_blocks_without_replaying_action(missing):
    from scripts.study_operator.i7_browser import StreamlitSync
    from streamlit.proto.BackMsg_pb2 import BackMsg
    from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
    class Page:
        ready=True
        def on(self,*args):
            pass
        def locator(self,*args):
            return self
        def count(self):
            return int(self.ready)
        def wait_for_timeout(self,*args):
            pass
    page=Page()
    sync=StreamlitSync(page,.001)
    request=BackMsg()
    request.rerun_script.SetInParent()
    finished=ForwardMsg(script_finished=ForwardMsg.FINISHED_SUCCESSFULLY)
    sync.sent(request.SerializeToString())
    sync.received(finished.SerializeToString())
    actions=[]
    def action():
        actions.append('one user action')
        if missing!='request':
            sync.sent(request.SerializeToString())
        if missing!='finished':
            sync.received(finished.SerializeToString())
        page.ready=missing!='render'
    with pytest.raises(ValueError,match='STREAMLIT_RERUN_NOT_ACKNOWLEDGED'):
        sync.action(action)
    assert actions==['one user action']
