#!/usr/bin/env python3
"""Pause the queue parent, let E1 finish, then exit without launching queued E2."""
import argparse,json,os,signal,time
from pathlib import Path
from datetime import datetime,timezone

def live(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return stat[0]!='Z'
    except FileNotFoundError:return False

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True);a=p.parse_args();root=Path(a.run_root).resolve()
    state=json.loads((root/'state.json').read_text());parent=state['pipeline_pid'];child=state['child_pid']
    if state['status']!='training' or state['stage']!='E1':raise RuntimeError('Only cancels a pending E2 while E1 is still running')
    parent_command=Path(f'/proc/{parent}/cmdline').read_bytes().decode();child_command=Path(f'/proc/{child}/cmdline').read_bytes().decode()
    parent_args=parent_command.split('\0')
    assert 'run_football_review_e1_e2.py' in parent_command
    assert Path(parent_args[parent_args.index('--run-root')+1]).resolve()==root
    assert 'train_football_events.py' in child_command and str(root/'E1.yaml') in child_command
    assert Path(f'/proc/{parent}').stat().st_uid==os.getuid()
    os.kill(parent,signal.SIGSTOP)
    control={'requested_at':datetime.now(timezone.utc).isoformat(),'reason':'User moves E2 to remote four-GPU training','queue_parent_pid':parent,'E1_child_pid':child,'watcher_pid':os.getpid(),'status':'E2_cancelled_waiting_for_E1','E1_continues':True}
    (root/'local_E2_cancelled.json').write_text(json.dumps(control,indent=2)+'\n')
    print(json.dumps(control),flush=True)
    while live(child):time.sleep(15)
    # E1 is finished. Deliver TERM before resuming the stopped parent so its
    # handler exits before it can launch E2; it never terminates an active E1.
    if live(parent):
        os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
    for _ in range(30):
        if not live(parent):break
        time.sleep(.2)
    control['status']='E1_finished_local_E2_cancelled';control['finished_at']=datetime.now(timezone.utc).isoformat();(root/'local_E2_cancelled.json').write_text(json.dumps(control,indent=2)+'\n')
    state=json.loads((root/'state.json').read_text());ok=(root/'E1/metrics_epoch_002.json').is_file()
    state.update(status='complete_local_training' if ok else 'E1_ended_check_logs',stage='E1',child_pid=None,completed=['E1_training'] if ok else [],cancelled=['E2_training','E2_test18'],external_E1_test18='E1_test18_dense_gpu4_epoch001_20260917',updated_at=datetime.now(timezone.utc).isoformat())
    temp=root/'state.json.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'state.json');print(json.dumps(state),flush=True)

if __name__=='__main__':main()
