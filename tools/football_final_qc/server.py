#!/usr/bin/env python3
"""Independent provenance-aware final adjudication UI."""
from __future__ import annotations
import argparse, hashlib, hmac, json, mimetypes, sqlite3, ssl, threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

LABELS={"shot","save","corner","free_kick","kickoff","penalty","set_piece","throw_in","back_pass"}
LOCK=threading.RLock()
def now(): return datetime.now(timezone.utc).isoformat()
def dumps(x): return json.dumps(x,ensure_ascii=False,separators=(",",":"))

def init_db(path,cases):
 path.parent.mkdir(parents=True,exist_ok=True)
 with LOCK,sqlite3.connect(path) as c:
  c.executescript("""PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS final_decisions(case_id TEXT PRIMARY KEY,source_hash TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'unreviewed',selected_events_json TEXT NOT NULL DEFAULT '[]',note TEXT NOT NULL DEFAULT '',reviewer TEXT NOT NULL DEFAULT '',revision INTEGER NOT NULL DEFAULT 0,provisional INTEGER NOT NULL DEFAULT 1,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS final_history(id INTEGER PRIMARY KEY AUTOINCREMENT,case_id TEXT NOT NULL,revision INTEGER NOT NULL,action TEXT NOT NULL,state_json TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_final_history_case ON final_history(case_id,revision);""")
  ts=now()
  for x in cases:
   r=c.execute("SELECT source_hash,status,revision FROM final_decisions WHERE case_id=?",(x["id"],)).fetchone()
   provisional=int(not x.get("video_first_pass_complete",False))
   if not r:c.execute("INSERT INTO final_decisions(case_id,source_hash,updated_at,provisional) VALUES(?,?,?,?)",(x["id"],x["source_hash"],ts,provisional))
   elif r[0]!=x["source_hash"]:
    old=c.execute("SELECT * FROM final_decisions WHERE case_id=?",(x["id"],)).fetchone()
    c.execute("INSERT INTO final_history(case_id,revision,action,state_json,created_at) VALUES(?,?,?,?,?)",(x["id"],r[2],"source_changed",dumps(list(old)),ts))
    status="stale" if r[1] not in {"unreviewed","stale"} else "unreviewed"
    c.execute("UPDATE final_decisions SET source_hash=?,status=?,revision=revision+1,updated_at=?,provisional=? WHERE case_id=?",(x["source_hash"],status,ts,provisional,x["id"]))
   else:c.execute("UPDATE final_decisions SET provisional=? WHERE case_id=?",(provisional,x["id"]))
  c.commit()

def decisions(path):
 with LOCK,sqlite3.connect(path) as c:
  c.row_factory=sqlite3.Row;rows=c.execute("SELECT * FROM final_decisions").fetchall()
 out={}
 for r in rows:
  d=dict(r);d["selected_events"]=json.loads(d.pop("selected_events_json") or "[]");d["provisional"]=bool(d["provisional"]);out[d["case_id"]]=d
 return out

def clean_event(e,duration,i):
 label=str(e.get("semantic_label") or e.get("label") or "")
 if label not in LABELS:raise ValueError(f"不支持的类别: {label}")
 t=float(e.get("time_sec"))
 if not 0<=t<=duration:raise ValueError(f"事件时间超出视频范围: {t}")
 return {"source_id":str(e.get("source_id") or e.get("id") or f"manual_{i}")[:240],"semantic_label":label,"time_sec":round(t,3),"lineage_gt_ids":sorted({str(x) for x in e.get("lineage_gt_ids",[]) if x}),"origin":str(e.get("origin") or "manual")[:40]}
def near_pairs(events,tol):
 return [[a["source_id"],b["source_id"]] for i,a in enumerate(events) for b in events[i+1:] if a["semantic_label"]==b["semantic_label"] and abs(a["time_sec"]-b["time_sec"])<=tol]

HTML_PATH=Path(__file__).with_name("index.html")

class App:
 def __init__(self,a):self.a=a;self.file=a.cases.resolve();self.db=a.db.resolve();self.proxy=a.proxy_root.resolve();self.reload()
 def reload(self):
  self.data=json.loads(self.file.read_text());self.cases={x["id"]:x for x in self.data["cases"]};init_db(self.db,self.data["cases"])
 def bootstrap(self):
  self.reload();return {"schema_version":self.data["schema_version"],"created_at":self.data["created_at"],"summary":self.data["summary"],"videos":self.data["videos"],"cases":self.data["cases"],"decisions":{k:v for k,v in decisions(self.db).items() if k in self.cases},"policy":{"standard_nms":False,"match_tolerance_sec":self.data["source"]["match_tolerance_sec"],"duplicate_window_sec":self.data["source"]["duplicate_window_sec"],"formal_evaluation_tolerance_sec":self.data["source"].get("formal_evaluation_tolerance_sec",3.0),"prior_policy":self.data["source"].get("prior_policy",{}),"default_reviewer":self.a.default_reviewer}}
class Handler(BaseHTTPRequestHandler):
 @property
 def app(self):return self.server.app
 def auth(self):
  q=parse_qs(urlparse(self.path).query);got=q.get("token",[""])[0];got=got or (self.headers.get("Authorization","")[7:] if self.headers.get("Authorization","").startswith("Bearer ") else "")
  return bool(got) and hmac.compare_digest(hashlib.sha256(got.encode()).digest(),hashlib.sha256(self.app.a.token.encode()).digest())
 def security_headers(self,cache_control="no-store"):
  self.send_header("X-Content-Type-Options","nosniff");self.send_header("Referrer-Policy","no-referrer");self.send_header("Cache-Control",cache_control);self.send_header("Content-Security-Policy","default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; media-src 'self'; connect-src 'self'")
 def send_json(self,x,status=200):
  b=dumps(x).encode();self.send_response(status);self.security_headers();self.send_header("Content-Type","application/json; charset=utf-8");self.send_header("Content-Length",str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self):
  path=urlparse(self.path).path
  if path=="/healthz":return self.send_json({"ok":True})
  if not self.auth():return self.send_json({"error":"Authentication required"},401)
  if path=="/":
   b=HTML_PATH.read_bytes();self.send_response(200);self.security_headers();self.send_header("Content-Type","text/html; charset=utf-8");self.send_header("Content-Length",str(len(b)));self.end_headers();return self.wfile.write(b)
  if path=="/api/bootstrap":return self.send_json(self.app.bootstrap())
  if path.startswith("/media/"):return self.media(path.rsplit("/",1)[-1])
  return self.send_json({"error":"Not found"},404)
 def media(self,vid):
  if vid not in self.app.data["videos"]:return self.send_json({"error":"Unknown video"},404)
  proxy=self.app.proxy/f"{vid}.mp4";path=proxy if proxy.is_file() else Path(self.app.data["videos"][vid]["video_path"])
  if not path.is_file():return self.send_json({"error":"Video unavailable"},404)
  st=path.stat();size=st.st_size;start,end,status=0,size-1,200;r=self.headers.get("Range")
  if r and r.startswith("bytes="):
   try:
    a,b=r[6:].split("-",1)
    if not a:
     suffix=int(b);start=max(0,size-suffix);end=size-1
    else:
     start=int(a);end=min(int(b) if b else size-1,size-1)
    if start<0 or start>=size or end<start:raise ValueError
    status=206
   except (ValueError,TypeError):return self.send_json({"error":"Invalid range"},416)
  n=end-start+1;self.send_response(status);self.security_headers("private, max-age=86400");self.send_header("Content-Type",mimetypes.guess_type(path.name)[0] or "video/mp4");self.send_header("Accept-Ranges","bytes");self.send_header("ETag",f'"{size:x}-{st.st_mtime_ns:x}"');self.send_header("Content-Length",str(n));
  if status==206:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
  self.end_headers()
  with path.open("rb") as f:
   f.seek(start);left=n
   while left:
    buf=f.read(min(1024*1024,left))
    if not buf:break
    try:self.wfile.write(buf)
    except (BrokenPipeError,ConnectionResetError):break
    left-=len(buf)
 def do_POST(self):
  if not self.auth():return self.send_json({"error":"Authentication required"},401)
  if urlparse(self.path).path!="/api/decision":return self.send_json({"error":"Not found"},404)
  try:
   n=int(self.headers.get("Content-Length",0))
   if n>1024*1024:raise ValueError("请求过大")
   b=json.loads(self.rfile.read(n) or b"{}");self.app.reload();cid=str(b.get("case_id",''));case=self.app.cases.get(cid)
   if not case:raise ValueError("终审案例不存在或已变化，请刷新")
   if b.get("source_hash")!=case["source_hash"]:raise ValueError("初审来源已更新，请刷新后判断")
   action=str(b.get("action",''))
   if action not in {"confirm","keep_both","pending"}:raise ValueError("非法操作")
   duration=float(self.app.data["videos"][case["video_id"]]["duration_sec"]);events=[clean_event(e,duration,i) for i,e in enumerate(b.get("selected_events",[]))];pairs=near_pairs(events,float(self.app.data["source"]["duplicate_window_sec"]))
   if pairs and action=="confirm":raise ValueError("终稿仍含同类近邻事件；请合并一条，或明确保留近邻两条")
   if action=="keep_both" and not pairs:raise ValueError("当前没有需明确保留的同类近邻事件")
   status="pending" if action=="pending" else "keep_both" if action=="keep_both" else "deleted" if not events else "confirmed";reviewer=str(b.get("reviewer",'')).strip()[:100]
   if not reviewer:raise ValueError("请填写终审人员姓名")
   note=str(b.get("note",''))[:4000];rev=int(b.get("expected_revision",-1));ts=now()
   with LOCK,sqlite3.connect(self.app.db) as c:
    c.row_factory=sqlite3.Row;r=c.execute("SELECT * FROM final_decisions WHERE case_id=?",(cid,)).fetchone()
    if not r or r["source_hash"]!=case["source_hash"]:raise ValueError("初审来源已变化，请刷新")
    if r["revision"]!=rev:raise ValueError("此案例已被其他终审人员更新，请刷新")
    c.execute("INSERT INTO final_history(case_id,revision,action,state_json,created_at) VALUES(?,?,?,?,?)",(cid,r["revision"],action,dumps(dict(r)),ts));cur=c.execute("UPDATE final_decisions SET status=?,selected_events_json=?,note=?,reviewer=?,revision=revision+1,provisional=?,updated_at=? WHERE case_id=? AND revision=?",(status,dumps(events),note,reviewer,int(not case.get("video_first_pass_complete",False)),ts,cid,rev))
    if cur.rowcount!=1:raise ValueError("并发更新冲突，请刷新")
    c.commit()
   return self.send_json({"ok":True,"decision":decisions(self.app.db)[cid],"near_duplicate_pairs":pairs})
  except (ValueError,TypeError,json.JSONDecodeError) as e:return self.send_json({"error":str(e)},409)
  except Exception as e:return self.send_json({"error":f"server error: {e}"},500)
 def log_message(self,fmt,*args):print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt%args}")
class ReviewHTTPServer(ThreadingHTTPServer):
 request_queue_size=128
 daemon_threads=True
 allow_reuse_address=True
def main():
 p=argparse.ArgumentParser();p.add_argument("--cases",type=Path,required=True);p.add_argument("--db",type=Path,required=True);p.add_argument("--proxy-root",type=Path,required=True);p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8776);p.add_argument("--token",required=True);p.add_argument("--default-reviewer",default="终审员1");p.add_argument("--tls-cert",type=Path);p.add_argument("--tls-key",type=Path);a=p.parse_args();app=App(a);server=ReviewHTTPServer((a.host,a.port),Handler);server.app=app
 if a.tls_cert and a.tls_key:
  ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain(a.tls_cert,a.tls_key);server.socket=ctx.wrap_socket(server.socket,server_side=True)
 print(dumps({"status":"serving","host":a.host,"port":a.port,"cases":len(app.cases),"db":str(app.db)}),flush=True);server.serve_forever()
if __name__=="__main__":main()
