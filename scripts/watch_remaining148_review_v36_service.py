#!/usr/bin/env python3
"""Atomically switch port 8775 to the prepared remaining148 v36 review UI.

The completed test18 database is never reused.  This watchdog starts only after
the new manifest, team palettes and four-user credentials have been validated,
then keeps versioned SQLite backups while the service is active.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import sqlite3
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/football_full_review/full166_fromlast_e8_best_dense_s5_20260902"
PLATFORM = RUN / "review_platform_remaining148"
TLS_ROOT = ROOT / "outputs/football_event_review/test18_multiuser_qc_20260903/tls"
PORT = 8775
SESSION = "football_remaining148_review_v36_server_20260907"


def tmux_alive() -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", SESSION],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def listener_pid() -> int | None:
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-t"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return int(values[0]) if values else None


def command_for_pid(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    return path.read_bytes().replace(b"\0", b" ").decode(errors="replace") if path.exists() else ""


def stop_expected_legacy_server() -> None:
    pid = listener_pid()
    if pid is None:
        return
    command = command_for_pid(pid)
    if (
        "server_multiuser_v36.py" in command
        and "review_platform_remaining148/reviews.sqlite3" in command
    ):
        return
    expected = "server_multiuser_v35.py"
    expected_db = "test18_full_annotation_repair_20260901/reviews.sqlite3"
    if expected not in command or expected_db not in command:
        raise RuntimeError(f"refusing to stop unexpected 8775 listener pid={pid}: {command}")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while listener_pid() is not None and time.monotonic() < deadline:
        time.sleep(0.25)
    if listener_pid() is not None:
        raise RuntimeError(f"legacy 8775 server did not stop: pid={pid}")


def start_server() -> None:
    command = [
        "/home/new_users/qiuqi/miniconda3/bin/python",
        "tools/football_event_review/server_multiuser_v36.py",
        "--manifest", str(PLATFORM / "review_manifest.json"),
        "--db", str(PLATFORM / "reviews.sqlite3"),
        "--access-config", str(PLATFORM / "access_config.json"),
        "--proxy-root", str(PLATFORM / "media_original_faststart"),
        "--host", "119.147.202.180", "--port", str(PORT),
        "--tls-cert", str(TLS_ROOT / "server.crt"),
        "--tls-key", str(TLS_ROOT / "server.key"),
    ]
    log = PLATFORM / "server.log"
    shell = f"{shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION, shell], cwd=ROOT, check=False
    )
    if result.returncode:
        raise RuntimeError("failed to start v36 server tmux")


def wait_health() -> None:
    context = __import__("ssl")._create_unverified_context()
    deadline = time.monotonic() + 600
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            data = urllib.request.urlopen(
                f"https://119.147.202.180:{PORT}/healthz", context=context, timeout=3
            ).read()
            if json.loads(data).get("version") == "v36-team-calibration":
                return
        except Exception as error:  # retained for startup diagnostics
            last_error = error
        time.sleep(1)
    raise RuntimeError(f"v36 health check failed: {last_error}")


def backup_database(source: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = PLATFORM / "backups" / f"reviews_{stamp}.sqlite3"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_db, target_db = sqlite3.connect(source), sqlite3.connect(destination)
    try:
        source_db.backup(target_db)
        if target_db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("review backup integrity check failed")
    finally:
        target_db.close(); source_db.close()
    return destination


def main() -> None:
    while not (PLATFORM / "PREPARED.json").is_file():
        time.sleep(30)
    readonly = ROOT / "outputs/football_event_review/test18_completed_readonly_snapshot_20260907"
    if not (readonly / "reviews.sqlite3").is_file():
        raise RuntimeError("completed test18 read-only snapshot is missing")
    stop_expected_legacy_server()
    if not tmux_alive():
        start_server()
    wait_health()
    last_revision = -1
    last_backup = 0.0
    while True:
        if not tmux_alive():
            start_server(); wait_health()
        database = PLATFORM / "reviews.sqlite3"
        if database.is_file():
            with sqlite3.connect(database) as connection:
                revision = int(connection.execute("SELECT COALESCE(SUM(revision),0) FROM reviews").fetchone()[0])
                revision += int(connection.execute("SELECT COALESCE(SUM(revision),0) FROM video_team_profiles").fetchone()[0])
            if revision != last_revision and time.monotonic() - last_backup >= 600:
                backup_database(database)
                last_revision, last_backup = revision, time.monotonic()
        ready = {
            "status": "ready_public",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "base_url": f"https://119.147.202.180:{PORT}",
            "credentials": str((PLATFORM / "credentials_private.json").resolve()),
            "database_revision_sum": last_revision,
        }
        temporary = PLATFORM / "READY.json.tmp"
        temporary.write_text(json.dumps(ready, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(PLATFORM / "READY.json")
        time.sleep(60)


if __name__ == "__main__":
    main()
