"""A live, visual transcript of the marking workflow.

Two files, both consumed by ``live.html``:
- ``live.jsonl``       append-only chat events (dictionary mining, every `edit`, summary).
- ``live_state.json``  overwritten after every edit: the current chapter's parsed segments.

The page renders the chapter as a flow of blocks: one big narration block that splits,
like raindrops, into coloured speech blocks (each labelled with its role) as marking
proceeds; the chat feed runs alongside.

Open it via the file server, e.g.
http://192.168.0.104:8899/outputs/<book>/script/live.html
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
  --accent:#5aa9ff;--ok:#43c785;--warn:#ffb454;--err:#ff6b6b;--purple:#b07cff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;overflow:hidden}
header{height:52px;display:flex;gap:16px;align-items:center;padding:0 18px;border-bottom:1px solid var(--line);background:#0d1017cc;backdrop-filter:blur(10px)}
.brand{font-weight:700}.brand small{color:var(--dim);font-weight:400;margin-left:8px}
.bar{flex:1;max-width:340px;height:8px;border-radius:99px;background:#222b39;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#3d7bff,#5aa9ff,#7ee0ff);transition:width .4s}
.stat{display:flex;gap:14px;font-size:12.5px;color:var(--dim);white-space:nowrap}.stat b{color:var(--fg)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 8px var(--ok);display:inline-block}
#main{display:grid;grid-template-columns:1.15fr .85fr;height:calc(100vh - 52px)}
#canvas,#chat{overflow:auto;padding:16px}
#canvas{border-right:1px solid var(--line);background:radial-gradient(900px 400px at 30% -10%,#17203355,transparent)}
.canvas-head{position:sticky;top:0;z-index:3;margin:-16px -16px 12px;padding:10px 16px;background:#0d1017e6;backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line);font-weight:700;font-size:14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.chip{font-size:11.5px;font-weight:600;color:var(--dim);background:var(--panel2);border:1px solid var(--line);padding:2px 8px;border-radius:99px}
.chip.on{color:#0b0f15;background:var(--accent);border-color:var(--accent)}
#blocks{display:flex;flex-direction:column;gap:8px;padding-bottom:40px}
.block{border-radius:12px;padding:9px 13px;font-size:14px;line-height:1.8;border:1px solid var(--line);background:var(--panel);
  word-break:break-word;transition:background .3s,border-color .3s}
.block.narr{color:#aab6c5;background:#12161d;font-size:13.5px}
.block.speech{border-color:#2b4a72;background:linear-gradient(180deg,#152238,#121924);box-shadow:0 4px 14px #0006}
.block.speech .who{display:inline-block;font-size:11.5px;font-weight:700;color:#9fd0ff;background:#16314f;border-radius:6px;padding:1px 8px;margin-right:8px}
.block.drop{animation:drop .5s cubic-bezier(.2,.9,.3,1.2) both}
@keyframes drop{0%{opacity:0;transform:translateY(-14px) scale(.96);filter:blur(2px)}100%{opacity:1;transform:none;filter:none}}
.blocks-empty{color:var(--dim);font-size:13px}
.chapter{position:relative;margin:16px 0 22px;padding-left:16px;border-left:2px solid var(--line)}
.chapter::before{content:"";position:absolute;left:-6px;top:6px;width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px #5aa9ff22}
.chap-head{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-weight:700;font-size:14px;margin:2px 0 10px}
.ev{margin:7px 0;display:flex;gap:10px}
.avatar{flex:0 0 26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:13px;background:var(--panel2);border:1px solid var(--line)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 12px;font-size:13.5px;line-height:1.65}
.dict .card{border-color:#5c4620;background:#241d10}.dict .avatar{background:#2c2312;border-color:#5c4620}
.think .card{color:var(--dim);background:#12161d}.think summary{cursor:pointer;color:var(--dim);font-size:12.5px;margin-bottom:4px}
.call .card{border-color:#25415f;background:#101a26}.call .avatar{background:#12233a;border-color:#25415f}
.res .card{border-color:#234;background:#111720;color:var(--dim);font-size:12.5px;font-family:ui-monospace,Menlo,Consolas,monospace;padding:6px 10px}
.res.ok .card{border-color:#1f4a33;color:#9fe6c1}.res.bad .card{border-color:#5a2626;color:#ffb3b3;background:#1d1212}
.sum .card{border-color:#3b3560;background:linear-gradient(180deg,#1b1830,#151a23)}.sum .avatar{background:#221d40;border-color:#3b3560}
.badge{display:inline-block;font-size:11px;font-weight:700;padding:1px 7px;border-radius:6px;margin-right:7px}
.b-speak{background:#17345a;color:#8fc4ff}.b-delete{background:#4a2a12;color:#ffba75}.b-replace{background:#33234a;color:#c9a6ff}
code{font-family:ui-monospace,Menlo,Consolas,monospace;background:#0d1219;border:1px solid var(--line);border-radius:6px;padding:1px 5px;font-size:12.5px}
.role{color:var(--accent);font-weight:600}.sp{color:var(--dim)}
</style></head><body>
<header>
  <div class="brand">mark_script <small id="book">· live</small></div>
  <div class="bar"><i id="prog"></i></div>
  <div class="stat">
    <span>章 <b id="done">0</b>/<b id="total">?</b></span>
    <span>✎ <b id="edits">0</b></span>
    <span>⚠ <b id="errs">0</b></span>
    <span><span class="dot"></span><span id="status"> live</span></span>
  </div>
</header>
<div id="main">
  <section id="canvas">
    <div class="canvas-head">▶ <span id="ch-chapter">等待开始…</span>
      <span class="chip" id="ch-blocks">0 块</span><span class="chip" id="ch-speech">0 台词</span>
    </div>
    <div id="blocks"><div class="blocks-empty">正文会像雨点一样，逐块变成台词……</div></div>
  </section>
  <section id="chat"><div id="feed"></div></section>
</div>
<script>
const BASE='__LIVE_BASE__';
const feed=document.getElementById('feed'), blocks=document.getElementById('blocks');
let offset=0, chapterEl=null, done=0,total=0,edits=0,errs=0,t0=null;
let lastChapter=null, segCount=0;
const book=document.getElementById('book'),prog=document.getElementById('prog'),doneEl=document.getElementById('done'),
 totalEl=document.getElementById('total'),editsEl=document.getElementById('edits'),errsEl=document.getElementById('errs'),
 status=document.getElementById('status'),chChapter=document.getElementById('ch-chapter'),
 chBlocks=document.getElementById('ch-blocks'),chSpeech=document.getElementById('ch-speech');
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function add(cls,avatar,html,parent){const w=document.createElement('div');w.className='ev '+cls;
  w.innerHTML='<div class="avatar">'+avatar+'</div><div>'+html+'</div>';(parent||chapterEl||feed).appendChild(w);return w;}
function newChapter(ch){
  chapterEl=document.createElement('div');chapterEl.className='chapter';
  chapterEl.innerHTML='<div class="chap-head">第 '+ch+' 章</div>';feed.appendChild(chapterEl);return chapterEl;}
function fmtCall(a){const x=a&&a.args||{};
  if(x.op==='speak')return '<span class="badge b-speak">speak</span><span class="role">'+esc(x.role||'?')+'</span> <code>'+esc(x.text||'')+'</code>';
  if(x.op==='delete')return '<span class="badge b-delete">delete</span><code>'+esc(x.text||'')+'</code>';
  if(x.op==='replace')return '<span class="badge b-replace">replace</span><code>'+esc(x.find||'')+'</code> <span class="sp">→</span> <code>'+esc(x.replace||'')+'</code>';
  return '<span class="badge b-replace">edit</span><code>'+esc(JSON.stringify(x))+'</code>';}
function handle(e){
  if(e.type==='start'){total=e.total||0;t0=Date.now();book.textContent='· '+(e.book||'live');return;}
  if(e.type==='chapter'){newChapter(e.chapter);return;}
  if(e.type==='roster'){add('dict','✦','词典 <b>+'+e.added+'</b> 标签 · 词条 '+e.roles);return;}
  if(e.type==='assistant'){add('think','🧠','<details><summary>思考</summary>'+esc(e.content)+'</details>');return;}
  if(e.type==='tool'){add('call','✎',fmtCall(e.args));edits++;editsEl.textContent=edits;return;}
  if(e.type==='result'){add('res '+(e.ok?'ok':'bad'),e.ok?'✓':'✗',esc(e.result));if(!e.ok){errs++;errsEl.textContent=errs;}return;}
  if(e.type==='done'){done++;doneEl.textContent=done;prog.style.width=(total?Math.min(100,done*100/total):0)+'%';
    add('sum','📝','<b>本章完成</b><div class="sp" style="margin-top:6px">'+esc(e.summary||'')+'</div>');return;}}
function clearCanvas(){blocks.innerHTML='';segCount=0;chChapter.textContent='等待开始…';chBlocks.textContent='0 块';chSpeech.textContent='0 台词';}
function renderState(s){
  if(!s||!s.segments)return;
  if(s.chapter!==lastChapter){lastChapter=s.chapter;blocks.innerHTML='';segCount=0;}
  chChapter.textContent='第 '+s.chapter+' 章 · '+s.chars+' 字';
  const segs=s.segments;
  for(let i=segCount;i<segs.length;i++){
    const g=segs[i],b=document.createElement('div');
    if(g.kind==='speech'){b.className='block speech drop';b.innerHTML='<span class="who">'+esc(g.role_name)+'</span>'+esc(g.text);}
    else{b.className='block narr drop';b.innerHTML=esc(g.text);}
    blocks.appendChild(b);
  }
  if(segs.length<segCount){blocks.innerHTML='';segCount=0;renderState(s);return;}
  segCount=segs.length;
  const speech=s.segments.filter(x=>x.kind==='speech').length;
  chBlocks.textContent=s.segments.length+' 块';chSpeech.textContent=speech+' 台词';
}
async function tickChat(){
  try{
    const r=await fetch(BASE+'live.jsonl',{headers:{'Range':'bytes='+offset+'-'},cache:'no-store'});
    if(r.status===416){offset=0;feed.innerHTML='';chapterEl=null;done=edits=errs=0;doneEl.textContent=editsEl.textContent=errsEl.textContent=0;prog.style.width='0';}
    else{const buf=await r.arrayBuffer();if(r.status===200&&offset>0){offset=0;feed.innerHTML='';chapterEl=null;}
      offset+=buf.byteLength;const txt=new TextDecoder().decode(buf);
      for(const line of txt.split('\n')){if(line.trim()){let e;try{e=JSON.parse(line);}catch(_){continue;}handle(e);}}
      totalEl.textContent=total||'?';status.textContent=' live';}
  }catch(err){status.textContent=' waiting…';}
  setTimeout(tickChat,1500);
}
async function tickState(){
  try{const r=await fetch(BASE+'live_state.json?t='+Date.now(),{cache:'no-store'});
    if(r.ok)renderState(await r.json());}catch(err){}
  setTimeout(tickState,900);
}
clearCanvas();tickChat();tickState();
</script></body></html>
"""


class Live:
    """Event log + current-chapter state + a self-refreshing visual page."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.state_path = path.parent / "live_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self.state_path.write_text("{}", encoding="utf-8")
        app_root = Path(__file__).resolve().parent.parent
        base = "/" + str(path.parent.resolve().relative_to(app_root)) + "/"
        (path.parent / "live.html").write_text(HTML.replace("__LIVE_BASE__", base), encoding="utf-8")

    def emit(self, type: str, **event) -> None:  # noqa: A002 - 'type' matches the wire format
        event["type"] = type
        event["ts"] = round(time.time(), 3)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def set_state(self, chapter: int, chars: int, segments: list[dict]) -> None:
        """Overwrite the current-chapter canvas state (atomic, so readers never see half)."""
        payload = json.dumps({"chapter": chapter, "chars": chars, "segments": segments}, ensure_ascii=False)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.state_path)
