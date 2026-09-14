"""A live, visual transcript of the marking workflow.

``mark_script`` writes, under ``outputs/<book>/script/``:
- ``live.jsonl``      append-only chat events (dictionary mining, every `edit`, summary).
- ``chNNN.json``      the chapter's current inline ``fragments`` (narration / speech /
                      deleted / inserted), overwritten after every edit.
- ``live_index.json`` ``{"current": cid, "chapters": {cid: {chars, done}}}`` for the picker.

``live.html`` shows the selected chapter as ONE continuous article (paragraph breaks kept)
with colour blocks appearing in place -- speech -> role card, deletions -> red strike,
inserted punctuation -> green. The chat panel on the right is a fixed-length stream that
keeps feeding upward.

Open it via the file server: http://<host>:8899/live
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

HTML = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mark_script · live</title>
<style>
:root{color-scheme:dark;--bg:#0d1017;--panel:#151a23;--panel2:#1b2230;--line:#26303f;--fg:#e6ecf3;--dim:#8b97a8;
  --accent:#5aa9ff;--ok:#43c785;--warn:#ffb454}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;overflow:hidden}
header{height:46px;display:flex;gap:12px;align-items:center;padding:0 18px;border-bottom:1px solid var(--line)}
.brand{font-weight:700}.brand small{color:var(--dim);font-weight:400;margin-left:8px}
header .grow{flex:1}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 8px var(--ok)}
#main{display:grid;grid-template-columns:1fr 380px;height:calc(100dvh - 46px)}
#left{display:flex;flex-direction:column;min-width:0;min-height:0;border-right:1px solid var(--line)}
#chbar{display:flex;gap:10px;align-items:center;padding:7px 16px;border-bottom:1px solid var(--line);font-size:12.5px;color:var(--dim)}
#chbar select{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:3px 8px;font-size:13px}
#chbar button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:3px 10px;cursor:pointer;font-size:12.5px}
#chbar button.on{background:var(--accent);color:#0b0f15;border-color:var(--accent);font-weight:700}
#article{overflow:auto;min-height:0;padding:20px 30px 60px;flex:1;font-size:15.5px;line-height:2.05;white-space:pre-wrap;word-break:break-word}
.who{display:inline-block;font-size:11.5px;font-weight:700;color:#9fd0ff;background:#16314f;border-radius:6px;padding:0 7px;margin:0 3px 0 2px;vertical-align:1px;line-height:1.7}
.speech{background:#152238;border-radius:8px;padding:2px 6px;box-shadow:inset 0 0 0 1px #2b4a72}
del{color:#ff9d9d;background:#2a1414;text-decoration:line-through;border-radius:5px;padding:1px 3px}
ins{color:#9fe6c1;background:#122a1c;text-decoration:none;border-radius:5px;padding:1px 5px}
.new{animation:appear .65s cubic-bezier(.2,.9,.3,1.1) both}
@keyframes appear{0%{opacity:0;filter:blur(4px)}55%{opacity:1}100%{opacity:1;filter:blur(0)}}
#chat{overflow:hidden;min-height:0;padding:10px 12px;display:flex;flex-direction:column;justify-content:flex-end;gap:5px}
.ev{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:5px 9px;font-size:12.5px;line-height:1.55;word-break:break-word}
.ico{display:inline-block;width:15px;margin-right:5px;text-align:center;opacity:.9}
.think{color:var(--dim);background:#12161d}.think summary{cursor:pointer;font-size:12px}
.call{border-color:#25415f;background:#101a26}
.dict{border-color:#5c4620;background:#241d10}
.res{border-color:#234;background:#111720;color:var(--dim);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px}
.res.ok{border-color:#1f4a33;color:#9fe6c1}.res.bad{border-color:#5a2626;color:#ffb3b3;background:#1d1212}
.sum{border-color:#3b3560;background:linear-gradient(180deg,#1b1830,#151a23)}
.ev code{font-size:11px;padding:0 3px}
.badge{display:inline-block;font-size:10.5px;font-weight:700;padding:0 5px;border-radius:6px;margin-right:5px}
.b-speak{background:#17345a;color:#8fc4ff}.b-delete{background:#4a2a12;color:#ffba75}.b-replace{background:#33234a;color:#c9a6ff}
code{font-family:ui-monospace,Menlo,Consolas,monospace;background:#0d1219;border:1px solid var(--line);border-radius:5px;padding:0 4px;font-size:12px}
.role{color:var(--accent);font-weight:600}.sp{color:var(--dim)}.chap{color:var(--accent);font-weight:700;font-size:11.5px;margin-right:6px}
/* phone / narrow: stack vertically -- article on top, chat as a fixed-height stream below */
@media (max-width:820px){
  #main{grid-template-columns:1fr;grid-template-rows:1fr minmax(28dvh,38dvh)}
  #left{border-right:none;border-bottom:1px solid var(--line)}
  #article{padding:14px 16px 40px;font-size:16px;line-height:1.95}
  #chat{padding:8px 10px;gap:4px}
  .ev{font-size:12px;padding:4px 8px}
  #chbar{gap:8px;padding:6px 12px;font-size:12px;overflow-x:auto;white-space:nowrap}
  header{padding:0 12px}.brand{font-size:14px}
}
</style></head><body>
<header>
  <div class="brand">mark_script <small id="book">· live</small></div>
  <span class="grow"></span><span class="dot"></span>
</header>
<div id="main">
  <div id="left">
    <div id="chbar">
      <select id="chs"></select>
      <button id="follow" class="on">跟随最新</button>
      <span id="finfo"></span>
    </div>
    <div id="article"><span class="sp">等待正文……</span></div>
  </div>
  <div id="chat"></div>
</div>
<script>
const BASE='__LIVE_BASE__', MAXCHAT=32;
const article=document.getElementById('article'), chat=document.getElementById('chat'), chs=document.getElementById('chs');
const book=document.getElementById('book'), followBtn=document.getElementById('follow'), finfo=document.getElementById('finfo');
let offset=0, follow=true, selected=null, current=null, prevSig='', idxSig='', seen=new Set();
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function fragHTML(f,isNew){const c=isNew?' new':'';
  if(f.kind==='speech')return '<span class="speech'+c+'"><span class="who">'+esc(f.role||'?')+'</span>'+esc(f.text)+'</span>';
  if(f.kind==='deleted')return '<del'+c+'>'+esc(f.text)+'</del>';
  if(f.kind==='inserted')return '<ins'+c+'>'+esc(f.text)+'</ins>';
  return '<span'+c+'>'+esc(f.text)+'</span>';}
function renderArticle(s){
  if(!s||!s.fragments||document.hidden)return;
  const sig=JSON.stringify(s.fragments); if(sig===prevSig)return;
  if(seen.chapter!==s.chapter){seen=new Set();seen.chapter=s.chapter;}
  const top=article.scrollTop; let html='';
  for(const f of s.fragments){const key=f.kind+'|'+(f.role||'')+'|'+f.text;const isNew=!seen.has(key);seen.add(key);html+=fragHTML(f,isNew);}
  article.innerHTML=html; article.scrollTop=top; prevSig=sig;
  finfo.textContent=(s.chars||0)+' 字 · '+s.fragments.length+' 片段';
}
let pending=[];
function pushChat(e){
  const row=document.createElement('div'); row.className='ev '+e.cls;
  row.innerHTML=(e.icon?'<span class="ico">'+e.icon+'</span>':'')+(e.chapter?'<span class="chap">ch'+e.chapter+'</span>':'')+e.html;
  pending.push(row);
}
function flushChat(){
  if(!pending.length)return;
  const frag=document.createDocumentFragment();
  for(const row of pending)frag.appendChild(row);
  pending=[]; chat.appendChild(frag);
  let extra=chat.childElementCount-MAXCHAT;
  while(extra-->0)chat.removeChild(chat.firstChild);
}
function fmtCall(a){const x=a&&a.args||{};
  const op=x.op||(x.role?'speak':(x.find!==undefined||x.replace!==undefined)?'replace':'delete');
  if(op==='speak')return '<span class="badge b-speak">speak</span><span class="role">'+esc(x.role||'?')+'</span> <code>'+esc(x.text||'')+'</code>';
  if(op==='replace')return '<span class="badge b-replace">replace</span><code>'+esc(x.find||'')+'</code> <span class="sp">→</span> <code>'+esc(x.replace||'')+'</code>';
  return '<span class="badge b-delete">delete</span><code>'+esc(x.text||'')+'</code>';}
function handle(e){
  if(e.type==='start'){book.textContent='· '+(e.book||'live');return;}
  if(e.type==='chapter'){current=e.chapter;if(follow)selected=e.chapter;return;}
  if(e.type==='roster'){pushChat({cls:'dict',icon:'✦',chapter:e.chapter,html:'词典 <b>+'+e.added+'</b> 标签 · 词条 '+e.roles});return;}
  if(e.type==='assistant'){pushChat({cls:'think',icon:'🧠',chapter:e.chapter,html:'<details><summary>思考</summary>'+esc(e.content)+'</details>'});return;}
  if(e.type==='tool'){pushChat({cls:'call',chapter:e.chapter,html:fmtCall(e.args)});return;}
  if(e.type==='result'){pushChat({cls:'res '+(e.ok?'ok':'bad'),icon:e.ok?'✓':'✗',chapter:e.chapter,html:esc(e.result)});return;}
  if(e.type==='done'){pushChat({cls:'sum',icon:'📝',chapter:e.chapter,html:'<b>本章完成</b> <span class="sp">'+esc(e.summary||'')+'</span>'});return;}}
function handleText(txt){
  for(const line of txt.split('\n')){if(!line.trim())continue;let e;try{e=JSON.parse(line);}catch(_){continue;}handle(e);}
}
async function tickChat(){
  try{const r=await fetch(BASE+'live.jsonl',{headers:{'Range':'bytes='+offset+'-'},cache:'no-store'});
    if(r.status===416){                                   // 到了 EOF：可能只是没新数据
      const h=await fetch(BASE+'live.jsonl',{method:'HEAD',cache:'no-store'});
      const size=+(h.headers.get('Content-Length')||0);
      if(size<offset){                                    // 文件变小 -> 被截断（新一次运行）
        const fb=await (await fetch(BASE+'live.jsonl',{cache:'no-store'})).arrayBuffer();
        offset=0;chat.innerHTML='';pending=[];
        handleText(new TextDecoder().decode(fb)); offset=fb.byteLength;
      } else { offset=size; }
    }
    else if(r.status===206||offset===0){const buf=await r.arrayBuffer();offset+=buf.byteLength;
      handleText(new TextDecoder().decode(buf));}
  }catch(err){}
  flushChat();
  setTimeout(tickChat,1200);
}
async function tickIndex(){
  try{const r=await fetch(BASE+'live_index.json?t='+Date.now(),{cache:'no-store'});
    if(!r.ok)return; const idx=await r.json();
    const cur=idx.current; if(cur&&follow)selected=cur;
    const ids=Object.keys(idx.chapters||{}).map(Number).sort((a,b)=>a-b);
    const listKey=ids.join(',');
    if(listKey!==idxSig){idxSig=listKey;                  // rebuild only when the set changes
      chs.innerHTML=ids.map(function(c){var d=idx.chapters[c]&&idx.chapters[c].done?' ✔':'';return '<option value="'+c+'">第 '+c+' 章'+d+'</option>';}).join('');}
    const want=String(selected||cur||''); if(chs.value!==want)chs.value=want;
  }catch(err){}
  setTimeout(tickIndex,1500);
}
async function tickState(){
  const cid=selected||current; if(!cid){setTimeout(tickState,800);return;}
  try{const r=await fetch(BASE+'ch'+String(cid).padStart(3,'0')+'.json?t='+Date.now(),{cache:'no-store'});
    if(r.ok)renderArticle(await r.json());}catch(err){}
  setTimeout(tickState,800);
}
chs.addEventListener('change',function(){selected=+chs.value;follow=false;followBtn.classList.remove('on');prevSig='';});
followBtn.addEventListener('click',function(){follow=true;followBtn.classList.add('on');if(current)selected=current;prevSig='';});
tickChat();tickIndex();tickState();
</script></body></html>
"""


def write_page(directory: Path) -> Path:
    """Write ``live.html`` for a script directory (callable without touching the log)."""
    app_root = Path(__file__).resolve().parent.parent
    base = "/" + str(directory.resolve().relative_to(app_root)) + "/"
    page = directory / "live.html"
    page.write_text(HTML.replace("__LIVE_BASE__", base), encoding="utf-8")
    return page


class Live:
    """Event log + per-chapter fragments + a self-refreshing visual page."""

    def __init__(self, path: Path) -> None:
        self.dir = path.parent
        self.path = path
        self.index_path = self.dir / "live_index.json"
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self.index: dict = {"current": 0, "chapters": {}}
        self._write_index()
        write_page(self.dir)

    def _write_index(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def emit(self, type: str, **event) -> None:  # noqa: A002 - 'type' matches the wire format
        event["type"] = type
        event["ts"] = round(time.time(), 3)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def set_state(self, chapter: int, chars: int, fragments: list[dict], done: bool = False) -> None:
        """Overwrite this chapter's canvas state (atomic) and refresh the index."""
        entry = self.index["chapters"].setdefault(str(chapter), {"chars": chars, "done": done})
        entry["chars"] = chars
        entry["done"] = entry.get("done", False) or done
        self.index["current"] = chapter
        payload = json.dumps({"chapter": chapter, "chars": chars, "fragments": fragments}, ensure_ascii=False)
        tmp = (self.dir / f"ch{chapter:03d}.json").with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.dir / f"ch{chapter:03d}.json")
        self._write_index()
