#!/usr/bin/env python3
"""Periodically rebuild the atomic final-QC source snapshot."""
import argparse,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--builder",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--review-db",type=Path,required=True);p.add_argument("--gt-run-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--interval-sec",type=float,default=60);p.add_argument("--match-tolerance-sec",type=float,default=5.0);p.add_argument("--duplicate-window-sec",type=float,default=5.0);a=p.parse_args()
 cmd=[sys.executable,str(a.builder),"--manifest",str(a.manifest),"--review-db",str(a.review_db),"--gt-run-dir",str(a.gt_run_dir),"--output",str(a.output),"--match-tolerance-sec",str(a.match_tolerance_sec),"--duplicate-window-sec",str(a.duplicate_window_sec)]
 while True:
  ts=datetime.now(timezone.utc).isoformat()
  r=subprocess.run(cmd,text=True,capture_output=True)
  print(f"[{ts}] exit={r.returncode} {r.stdout.strip() or r.stderr.strip()}",flush=True)
  time.sleep(max(15,a.interval_sec))
if __name__=="__main__":main()
