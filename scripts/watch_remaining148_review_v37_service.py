#!/usr/bin/env python3
"""Keep the public remaining148 v37 review service healthy on port 8775."""

from __future__ import annotations

import json
import os
import shlex
import signal
import sqlite3
import ssl
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
SESSION = "football_remaining148_review_v37_server_20260908"
PROXY_ENV_NAMES = (
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
)


def tmux_alive() -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", SESSION],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def listener_pid() -> int | None:
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-t"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return int(values[0]) if values else None


def command_for_pid(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    return path.read_bytes().replace(b"\0", b" ").decode(errors="replace") if path.exists() else ""


def stop_expected_previous_server() -> None:
    pid = listener_pid()
    if pid is None:
        return
    command = command_for_pid(pid)
    expected_db = "review_platform_remaining148/reviews.sqlite3"
    if "server_multiuser_v37.py" in command and expected_db in command:
        return
    previous_versions = ("server_multiuser_v36.py", "server_multiuser_v35.py")
    if expected_db not in command or not any(version in command for version in previous_versions):
        raise RuntimeError(f"refusing to stop unexpected 8775 listener pid={pid}: {command}")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while listener_pid() is not None and time.monotonic() < deadline:
        time.sleep(0.25)
    if listener_pid() is not None:
        raise RuntimeError(f"previous 8775 server did not stop: pid={pid}")


def direct_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in PROXY_ENV_NAMES:
        environment.pop(name, None)
    return environment


def start_server() -> None:
    command = [
        "/home/new_users/qiuqi/miniconda3/bin/python",
        "tools/football_event_review/server_multiuser_v37.py",
        "--manifest", str(PLATFORM / "review_manifest.json"),
        "--db", str(PLATFORM / "reviews.sqlite3"),
        "--access-config", str(PLATFORM / "access_config.json"),
        "--proxy-root", str(PLATFORM / "media_original_faststart"),
        "--host", "119.147.202.180",
        "--port", str(PORT),
        "--tls-cert", str(TLS_ROOT / "server.crt"),
        "--tls-key", str(TLS_ROOT / "server.key"),
    ]
    log = PLATFORM / "server.log"
    unset_proxy = "env " + " ".join(f"-u {name}" for name in PROXY_ENV_NAMES)
    shell = f"{unset_proxy} {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION, shell],
        cwd=ROOT,
        check=False,
        env=direct_environment(),
    )
    if result.returncode:
        raise RuntimeError("failed to start v37 server tmux")


def wait_health() -> None:
    context = ssl._create_unverified_context()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
    )
    deadline = time.monotonic() + 600
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                f"https://119.147.202.180:{PORT}/healthz",
                headers={"Connection": "close"},
            )
            data = opener.open(request, timeout=3).read()
            if json.loads(data).get("version") == "v37-direct-range-streaming":
                return
        except Exception as error:  # retained for startup diagnostics
            last_error = error
        time.sleep(1)
    raise RuntimeError(f"v37 direct health check failed: {last_error}")


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
        target_db.close()
        source_db.close()
    return destination


def main() -> None:
    if not (PLATFORM / "PREPARED.json").is_file():
        raise RuntimeError("remaining148 platform is not prepared")
    readonly = ROOT / "outputs/football_event_review/test18_completed_readonly_snapshot_20260907"
    if not (readonly / "reviews.sqlite3").is_file():
        raise RuntimeError("completed test18 read-only snapshot is missing")

    backup_database(PLATFORM / "reviews.sqlite3")
    stop_expected_previous_server()
    if not tmux_alive():
        start_server()
    wait_health()

    last_revision = -1
    last_backup = 0.0
    while True:
        if not tmux_alive():
            start_server()
            wait_health()
        database = PLATFORM / "reviews.sqlite3"
        if database.is_file():
            with sqlite3.connect(database) as connection:
                revision = int(connection.execute("SELECT COALESCE(SUM(revision),0) FROM reviews").fetchone()[0])
                revision += int(connection.execute("SELECT COALESCE(SUM(revision),0) FROM video_team_profiles").fetchone()[0])
            if revision != last_revision and time.monotonic() - last_backup >= 600:
                backup_database(database)
                last_revision, last_backup = revision, time.monotonic()
        ready = {
            "status": "ready_public_direct_http11_range",
            "version": "v37-direct-range-streaming",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "base_url": f"https://119.147.202.180:{PORT}",
            "credentials": str((PLATFORM / "credentials_private.json").resolve()),
            "database_revision_sum": last_revision,
            "proxy_environment_removed": True,
        }
        temporary = PLATFORM / "READY.json.tmp"
        temporary.write_text(json.dumps(ready, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(PLATFORM / "READY.json")
        time.sleep(60)


if __name__ == "__main__":
    main()
