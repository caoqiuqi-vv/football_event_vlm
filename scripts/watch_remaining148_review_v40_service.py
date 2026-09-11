#!/usr/bin/env python3
"""Guarded v40 activation and liveness monitoring for the existing 8775 service."""
from contextlib import closing
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import ssl
import subprocess
import sys
import time
import urllib.request

import watch_remaining148_review_v38_service as base

VERSION = 'v40-reliable-review'
MEDIA = base.PLATFORM / 'media_faststart_v40'
ENTRY = 'tools/football_event_review/server_multiuser_v40.py'
PREVIOUS = {f'watch_remaining148_review_v{i}_service.py' for i in range(36, 40)}


def health(path='/healthz'):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    with opener.open(f'https://119.147.202.180:{base.PORT}{path}', timeout=5) as response:
        return json.load(response)


def backup():
    folder = base.PLATFORM / 'backups'; folder.mkdir(exist_ok=True)
    path = folder / ('reviews_v40_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '.sqlite3')
    with closing(sqlite3.connect((base.PLATFORM/'reviews.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)) as source:
        with closing(sqlite3.connect(path)) as destination:
            source.backup(destination)
    path.chmod(0o600)
    print(f'BACKUP {path.name}', flush=True)
    return path


def expected_listener(allow_old=False):
    pid = base.listener_pid()
    if pid is None:
        return None
    command = base.command_for_pid(pid)
    versions = [f'server_multiuser_v{i}.py' for i in range(35, 41)] if allow_old else ['server_multiuser_v40.py']
    if 'review_platform_remaining148/reviews.sqlite3' not in command or not any(v in command for v in versions):
        raise RuntimeError(f'Refusing to replace unexpected port 8775 listener pid={pid}')
    return pid


def stop_old_monitors():
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            parts = (directory/'cmdline').read_bytes().split(b'\0')
            if any(Path(part.decode()).name in PREVIOUS for part in parts if part):
                if (directory/'cwd').resolve() == base.ROOT:
                    os.kill(int(directory.name), signal.SIGTERM)
                    print(f'STOP old_monitor={directory.name}', flush=True)
        except (OSError, UnicodeDecodeError):
            continue


def stop_listener(pid):
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while base.listener_pid() is not None:
        if time.monotonic() >= deadline:
            raise RuntimeError('Previous server did not release port 8775')
        time.sleep(.2)


def start_server():
    command = [sys.executable, ENTRY, '--manifest', str(base.PLATFORM/'review_manifest.json'),
        '--db', str(base.PLATFORM/'reviews.sqlite3'), '--access-config', str(base.PLATFORM/'access_config.json'),
        '--proxy-root', str(MEDIA), '--host', '119.147.202.180', '--port', str(base.PORT),
        '--tls-cert', str(base.TLS_ROOT/'server.crt'), '--tls-key', str(base.TLS_ROOT/'server.key'),
        '--require-faststart']
    with (base.PLATFORM/'server_v40.log').open('a') as log:
        process = subprocess.Popen(command, cwd=base.ROOT, env=base.direct_environment(),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (base.PLATFORM/'server_v40.pid').write_text(str(process.pid)+'\n')
    deadline = time.monotonic()+120
    while time.monotonic()<deadline:
        if process.poll() is not None:
            raise RuntimeError('v40 exited during startup; inspect server_v40.log')
        try:
            if health().get('version') == VERSION and health('/readyz').get('ok'):
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError('v40 startup did not become ready')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activate', action='store_true', help='Replace the verified previous 8775 service')
    args=parser.parse_args()
    with (base.PLATFORM/'.v40-monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, str(base.ROOT/'tools/football_event_review'))
        from mp4_layout import is_faststart
        manifest=json.loads((base.PLATFORM/'review_manifest.json').read_text())
        if not all((MEDIA/f"{v['video_id']}.mp4").is_file() and is_faststart(MEDIA/f"{v['video_id']}.mp4") for v in manifest['videos']):
            raise RuntimeError('Prepare and validate all faststart files before activation')
        pid=expected_listener(allow_old=args.activate)
        if pid is not None and 'server_multiuser_v40.py' not in base.command_for_pid(pid):
            if not args.activate:
                raise RuntimeError('Use --activate for an explicit old-version replacement')
            with closing(sqlite3.connect((base.PLATFORM/'reviews.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)) as connection:
                active=connection.execute('SELECT COUNT(*) FROM review_sessions WHERE expires_at>? AND last_seen_at>?',(time.time(),time.time()-60)).fetchone()[0]
            if active:
                raise RuntimeError('Reviewers active within 60 seconds; activate when idle')
            stop_old_monitors()
            backup()
            stop_listener(pid)
            backup()  # final consistent snapshot after all previous writers have stopped
            start_server()
        elif pid is None:
            start_server()
        failures=0; last_backup=time.monotonic()
        while True:
            status={'version':VERSION,'updated_at':datetime.now(timezone.utc).isoformat(), 'base_url':f'https://119.147.202.180:{base.PORT}'}
            try:
                if health().get('version')!=VERSION:
                    raise RuntimeError('Unexpected service version')
                ready=health('/readyz')
                failures=0
                status.update(status='ready',media=ready['media'])
            except Exception as error:
                failures+=1
                status.update(status='degraded',error=type(error).__name__,consecutive_failures=failures)
                if failures>=3:
                    pid=expected_listener()
                    if pid is not None:stop_listener(pid)
                    start_server();failures=0
            if time.monotonic()-last_backup>=600:
                try:
                    backup();last_backup=time.monotonic()
                except sqlite3.Error as error:
                    status['backup_error']=type(error).__name__
            tmp=base.PLATFORM/'READY_v40.json.tmp'
            tmp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
            tmp.replace(base.PLATFORM/'READY_v40.json')
            canonical=base.PLATFORM/'READY.json.tmp'
            canonical.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
            canonical.replace(base.PLATFORM/'READY.json')
            time.sleep(30)

if __name__=='__main__':main()
