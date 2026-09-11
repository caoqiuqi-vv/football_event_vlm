#!/usr/bin/env python3
"""Create a new, read-only QC snapshot; existing versions are never overwritten."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sqlite3
from datetime import datetime,timezone
from pathlib import Path

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 return h.hexdigest()
def backup_sqlite(source:Path,target:Path)->None:
 with sqlite3.connect(f'file:{source.resolve()}?mode=ro',uri=True) as src,sqlite3.connect(target) as dst:src.backup(dst)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--tag',required=True)
 p.add_argument('--review-db',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
 p.add_argument('--final-db',type=Path,required=True);p.add_argument('--final-cases',type=Path,required=True);a=p.parse_args()
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');target=a.output_root/f'{a.tag}_{stamp}'
 target.mkdir(parents=True,exist_ok=False)
 outputs={'first_pass_reviews.sqlite3':('sqlite',a.review_db),'review_manifest.json':('file',a.manifest),'final_qc.sqlite3':('sqlite',a.final_db),'final_cases.json':('file',a.final_cases)}
 try:
  for name,(kind,source) in outputs.items():
   destination=target/name
   backup_sqlite(source,destination) if kind=='sqlite' else shutil.copy2(source,destination)
  files={name:{'bytes':(target/name).stat().st_size,'sha256':sha256(target/name),'source':str(source.resolve())} for name,(_,source) in outputs.items()}
  metadata={'schema_version':1,'snapshot_id':target.name,'created_at':datetime.now(timezone.utc).isoformat(),'immutability_policy':'create-only version directory; files chmod 0444; directory chmod 0555; modifications require a new snapshot','files':files}
  (target/'snapshot.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  for path in target.iterdir():os.chmod(path,0o444)
  os.chmod(target,0o555)
 except Exception:
  shutil.rmtree(target,ignore_errors=True);raise
 print(json.dumps({'snapshot':str(target.resolve()),'files':len(outputs)+1,'read_only':True},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
