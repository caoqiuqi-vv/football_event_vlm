#!/usr/bin/env python3
"""Token-protected, read-only UI for inspecting GT/human discrepancies."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


LABELS = {"shot":"射门","save":"扑救","set_piece":"定位球","corner":"角球",
          "free_kick":"任意球","kickoff":"中圈开球","throw_in":"界外球","back_pass":"回传"}
KINDS = {"gt_missing":"GT 漏标","gt_removed":"GT 应删除","label_changed":"GT 类别误标",
         "time_changed":"GT 时间需修正","new_non_target":"人工确认的非目标事件","review_gap":"审核覆盖缺口"}


def page() -> bytes:
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GT 与人工确认差异分析</title><style>
:root{font-family:Inter,"PingFang SC",sans-serif;color:#e8edf5;background:#0d1117}body{margin:0}.top{padding:14px 20px;background:#161b22;position:sticky;top:0;z-index:2;border-bottom:1px solid #30363d}.layout{display:grid;grid-template-columns:minmax(520px,1.5fr) minmax(360px,1fr);gap:16px;padding:16px}.player{position:sticky;top:90px;align-self:start}.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px}video{width:100%;max-height:70vh;background:#000}.filters{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}button{background:#21262d;color:#e8edf5;border:1px solid #484f58;padding:7px 10px;border-radius:7px;cursor:pointer}button.on{background:#1f6feb}.item{margin-bottom:9px;cursor:pointer}.item.active{border-color:#58a6ff}.kind{font-weight:700}.time{color:#79c0ff}.old{color:#ff7b72}.new{color:#7ee787}.muted{color:#8b949e;font-size:13px}.stats{display:flex;gap:14px;flex-wrap:wrap}.detail{line-height:1.75;margin-top:12px}.nearby{margin-top:12px;padding:10px;border:1px solid #3b4654;border-radius:8px;background:#0f151d}.nearby-title{color:#f0c76a;font-weight:700}.nearby-row{display:grid;grid-template-columns:110px 1fr;gap:8px;margin-top:6px}.event-chip{display:inline-block;margin:2px 4px 2px 0;padding:2px 6px;border-radius:5px;background:#27313d;color:#dce6f2}.nearby-note{margin-top:7px;color:#8b949e;font-size:12px}@media(max-width:900px){.layout{grid-template-columns:1fr}.player{position:static}}
</style></head><body><div class="top"><h2 style="margin:0 0 8px">原 GT ↔ 人工确认差异（只读）</h2><div id="stats" class="stats"></div><div id="filters" class="filters"></div></div>
<main class="layout"><section class="player card"><video id="video" controls preload="metadata"></video><div id="detail" class="detail muted">请选择右侧差异条目</div></section><section id="list"></section></main>
<script>
const token=new URLSearchParams(location.search).get('token')||localStorage.getItem('diffToken')||''; if(token)localStorage.setItem('diffToken',token);
const q=(p)=>p+(p.includes('?')?'&':'?')+'token='+encodeURIComponent(token);let data,filter='all',active='';
const fmt=s=>{s=Number(s||0);return `${String(Math.floor(s/60)).padStart(2,'0')}:${(s%60).toFixed(3).padStart(6,'0')}`};
const label=x=>({shot:'射门',save:'扑救',set_piece:'定位球',corner:'角球',free_kick:'任意球',kickoff:'中圈开球',throw_in:'界外球',back_pass:'回传'}[x]||x||'无');
const kinds={gt_missing:'GT 漏标',gt_removed:'GT 应删除',label_changed:'GT 类别误标',time_changed:'GT 时间需修正',new_non_target:'人工确认的非目标事件',review_gap:'审核覆盖缺口'};
function describe(x){const o=x.original,h=x.human;return `<div><span class="old">原 GT：${o?label(o.semantic_label)+' @ '+fmt(o.time_sec):'无'}</span></div><div><span class="new">人工：${h?label(h.semantic_label)+' @ '+fmt(h.time_sec):'无'}</span></div>${x.delta_sec===undefined?'':`<div class="muted">时间差：${Number(x.delta_sec).toFixed(3)} 秒</div>`}${h?.note?`<div>备注：${h.note}</div>`:''}${x.kind==='review_gap'?`<div>候选：${(x.candidate_labels||[]).map(label).join('、')}</div>`:''}`}
function eventChips(rows){return rows?.length?rows.map(e=>`<span class="event-chip">${label(e.semantic_label)} @ ${fmt(e.time_sec)}</span>`).join(''):'<span class="muted">无事件</span>'}
function nearby(x){return `<section class="nearby"><div class="nearby-title">该播放片段的完整事件上下文</div><div class="nearby-row"><strong>原 GT 附近</strong><div>${eventChips(x.nearby_original)}</div></div><div class="nearby-row"><strong>人工最终附近</strong><div>${eventChips(x.nearby_human)}</div></div><div class="nearby-note">这里包含已正确匹配的事件；上方“原 GT：无”只表示当前类别没有对应 GT，不表示附近完全没有 GT。</div></section>`}
function select(x){active=x.id;document.querySelector('#video').currentTime=Math.max(0,x.start_sec);document.querySelector('#detail').innerHTML=`<div class="kind">${kinds[x.kind]}</div><div class="time">${fmt(x.start_sec)} – ${fmt(x.end_sec)}</div>${describe(x)}${nearby(x)}`;render()}
function render(){const rows=data.differences.filter(x=>filter==='all'||x.kind===filter);document.querySelector('#list').innerHTML=rows.map(x=>`<article class="item card ${active===x.id?'active':''}" data-id="${x.id}"><div><span class="kind">${kinds[x.kind]}</span> · <span class="time">${fmt(x.time_sec)}</span></div>${describe(x)}</article>`).join('')||'<div class="card muted">该筛选下无条目</div>';document.querySelectorAll('.item').forEach(e=>e.onclick=()=>select(data.differences.find(x=>x.id===e.dataset.id)));document.querySelectorAll('#filters button').forEach(e=>e.classList.toggle('on',e.dataset.kind===filter))}
fetch(q('/api/snapshot')).then(r=>{if(!r.ok)throw Error('unauthorized');return r.json()}).then(x=>{data=x;document.querySelector('#video').src=q('/video');const s=x.summary;document.querySelector('#stats').innerHTML=`<span>视频 ${x.video_id}</span><span>原 GT ${s.original_gt_events}</span><span>人工事件 ${s.human_final_events}</span><span>差异 ${s.differences}</span><span>已审核片段 ${s.committed_review_segments}/${s.all_review_segments}</span>`;const ks=['all',...Object.keys(s.by_kind)];document.querySelector('#filters').innerHTML=ks.map(k=>`<button data-kind="${k}">${k==='all'?'全部':kinds[k]}${k==='all'?'':` (${s.by_kind[k]})`}</button>`).join('');document.querySelectorAll('#filters button').forEach(e=>e.onclick=()=>{filter=e.dataset.kind;render()});render();if(x.differences.length)select(x.differences[0])}).catch(e=>document.body.innerHTML='<div class="card">访问令牌无效</div>');
</script></body></html>'''.encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "FootballDiscrepancyUI/1"
    def authorized(self) -> bool:
        query = parse_qs(urlparse(self.path).query)
        return not self.server.token or query.get("token", [""])[0] == self.server.token
    def do_GET(self) -> None:
        if not self.authorized():
            self.send_error(HTTPStatus.UNAUTHORIZED); return
        path = urlparse(self.path).path
        if path == "/": return self.send_bytes(page(), "text/html; charset=utf-8")
        if path == "/api/snapshot": return self.send_bytes(self.server.snapshot_bytes, "application/json")
        if path == "/video": return self.send_video(self.server.video_path)
        self.send_error(HTTPStatus.NOT_FOUND)
    def send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK); self.send_header("Content-Type",content_type)
        self.send_header("Content-Length",str(len(data))); self.send_header("Cache-Control","no-store")
        self.end_headers(); self.wfile.write(data)
    def send_video(self, path: Path) -> None:
        size=path.stat().st_size; start,end=0,size-1; status=HTTPStatus.OK
        value=self.headers.get("Range","")
        if value.startswith("bytes="):
            first,last=value[6:].split("-",1); start=int(first or 0); end=min(int(last) if last else size-1,size-1); status=HTTPStatus.PARTIAL_CONTENT
        if start<0 or start>end or start>=size: self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE); return
        self.send_response(status); self.send_header("Content-Type",mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges","bytes"); self.send_header("Content-Length",str(end-start+1))
        if status==HTTPStatus.PARTIAL_CONTENT:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start); remaining=end-start+1
            while remaining:
                chunk=f.read(min(1024*1024,remaining))
                if not chunk:break
                try:self.wfile.write(chunk)
                except (BrokenPipeError,ConnectionResetError):break
                remaining-=len(chunk)
    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--snapshot",type=Path,required=True);p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8774);p.add_argument("--token",default=os.environ.get("FOOTBALL_DISCREPANCY_TOKEN",""));a=p.parse_args()
    snapshot=json.loads(a.snapshot.read_text());server=ThreadingHTTPServer((a.host,a.port),Handler);server.snapshot_bytes=json.dumps(snapshot,ensure_ascii=False).encode();server.video_path=Path(snapshot["video_path"]);server.token=a.token
    print(f"serving discrepancy UI at http://{a.host}:{a.port}/",flush=True);server.serve_forever()


if __name__=="__main__":main()
