#!/usr/bin/env python3
"""Token-protected, read-only browser for missed football-event clips."""
from __future__ import annotations
import argparse,hashlib,hmac,json,mimetypes,ssl
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs,urlparse
HTML=Path(__file__).with_name("index.html")
def dumps(x):return json.dumps(x,ensure_ascii=False,separators=(",",":"))
class App:
 def __init__(self,a):
  self.a=a;self.root=a.root.resolve();self.manifest_path=(self.root/'manifest.json').resolve();self.reload()
 def reload(self):
  self.data=json.loads(self.manifest_path.read_text());self.files={x['filename'] for x in self.data['clips']}
 def public_manifest(self):
  self.reload();return {**self.data,'clips':[{k:v for k,v in x.items() if k!='source_video'} for x in self.data['clips']]}
class Handler(BaseHTTPRequestHandler):
 @property
 def app(self):return self.server.app
 def auth(self):
  q=parse_qs(urlparse(self.path).query);got=q.get('token',[''])[0]
  if not got and self.headers.get('Authorization','').startswith('Bearer '):got=self.headers['Authorization'][7:]
  return bool(got) and hmac.compare_digest(hashlib.sha256(got.encode()).digest(),hashlib.sha256(self.app.a.token.encode()).digest())
 def security_headers(self,cache='no-store'):
  self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer');self.send_header('Cache-Control',cache);self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; media-src 'self'; connect-src 'self'")
 def json(self,x,status=200):
  b=dumps(x).encode();self.send_response(status);self.security_headers();self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers()
  try:self.wfile.write(b)
  except (BrokenPipeError,ConnectionResetError):pass
 def do_GET(self):
  path=urlparse(self.path).path
  if path=='/healthz':return self.json({'ok':True,'clips':len(self.app.files)})
  if not self.auth():return self.json({'error':'Authentication required'},401)
  if path=='/':
   b=HTML.read_bytes();self.send_response(200);self.security_headers();self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();return self.wfile.write(b)
  if path=='/api/manifest':return self.json(self.app.public_manifest())
  if path.startswith('/clips/'):return self.media(path.rsplit('/',1)[-1])
  return self.json({'error':'Not found'},404)
 def media(self,name):
  if name not in self.app.files:return self.json({'error':'Unknown clip'},404)
  path=(self.app.root/'clips'/name).resolve()
  if path.parent!=(self.app.root/'clips').resolve() or not path.is_file():return self.json({'error':'Clip unavailable'},404)
  st=path.stat();size=st.st_size;start,end,status=0,size-1,200;r=self.headers_get_range()
  if r:
   try:
    a,b=r[6:].split('-',1)
    if not a:
     n=int(b);start=max(0,size-n);end=size-1
    else:start=int(a);end=min(int(b) if b else size-1,size-1)
    if start<0 or start>=size or end<start:raise ValueError
    status=206
   except (ValueError,TypeError):return self.json({'error':'Invalid range'},416)
  n=end-start+1;self.send_response(status);self.security_headers('private, max-age=86400');self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'video/mp4');self.send_header('Accept-Ranges','bytes');self.send_header('ETag',f'"{size:x}-{st.st_mtime_ns:x}"');self.send_header('Content-Length',str(n))
  if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  with path.open('rb') as f:
   f.seek(start);left=n
   while left:
    buf=f.read(min(1024*1024,left))
    if not buf:break
    try:self.wfile.write(buf)
    except (BrokenPipeError,ConnectionResetError):break
    left-=len(buf)
 def headers_get_range(self):
  r=self.headers.get('Range');return r if r and r.startswith('bytes=') else None
 def log_message(self,fmt,*args):print(f'[{self.log_date_time_string()}] {self.address_string()} {fmt%args}',flush=True)
class Server(ThreadingHTTPServer):request_queue_size=128;daemon_threads=True;allow_reuse_address=True
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8777);p.add_argument('--token',required=True);p.add_argument('--tls-cert',type=Path);p.add_argument('--tls-key',type=Path);a=p.parse_args();app=App(a);srv=Server((a.host,a.port),Handler);srv.app=app
 if a.tls_cert and a.tls_key:
  ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain(a.tls_cert,a.tls_key);srv.socket=ctx.wrap_socket(srv.socket,server_side=True)
 print(dumps({'status':'serving','host':a.host,'port':a.port,'clips':len(app.files),'root':str(app.root)}),flush=True);srv.serve_forever()
if __name__=='__main__':main()
