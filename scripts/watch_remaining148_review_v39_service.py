#!/usr/bin/env python3
"""Keep the public remaining148 v39 review service healthy on port 8775."""

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

import watch_remaining148_review_v38_service as base


SESSION = "football_remaining148_review_v39_server_20260908"
VERSION = "v39-clean-header-colour-names"


def stop_expected_previous_server() -> None:
    pid = base.listener_pid()
    if pid is None:
        return
    command = base.command_for_pid(pid)
    expected_db = "review_platform_remaining148/reviews.sqlite3"
    if "server_multiuser_v39.py" in command and expected_db in command:
        return
    previous_versions = (
        "server_multiuser_v38.py", "server_multiuser_v37.py",
        "server_multiuser_v36.py", "server_multiuser_v35.py",
    )
    if expected_db not in command or not any(version in command for version in previous_versions):
        raise RuntimeError(f"refusing to stop unexpected 8775 listener pid={pid}: {command}")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while base.listener_pid() is not None and time.monotonic() < deadline:
        time.sleep(0.25)
    if base.listener_pid() is not None:
        raise RuntimeError(f"previous 8775 server did not stop: pid={pid}")


def tmux_alive() -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", SESSION],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def start_server() -> None:
    command = [
        "/home/new_users/qiuqi/miniconda3/bin/python",
        "tools/football_event_review/server_multiuser_v39.py",
        "--manifest", str(base.PLATFORM / "review_manifest.json"),
        "--db", str(base.PLATFORM / "reviews.sqlite3"),
        "--access-config", str(base.PLATFORM / "access_config.json"),
        "--proxy-root", str(base.PLATFORM / "media_original_faststart"),
        "--host", "119.147.202.180",
        "--port", str(base.PORT),
        "--tls-cert", str(base.TLS_ROOT / "server.crt"),
        "--tls-key", str(base.TLS_ROOT / "server.key"),
    ]
    log = base.PLATFORM / "server.log"
    unset_proxy = "env " + " ".join(f"-u {name}" for name in base.PROXY_ENV_NAMES)
    shell = f"{unset_proxy} {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION, shell],
        cwd=base.ROOT,
        check=False,
        env=base.direct_environment(),
    )
    if result.returncode:
        raise RuntimeError("failed to start v39 server tmux")


def wait_health() -> None:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl._create_unverified_context()),
    )
    deadline = time.monotonic() + 600
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                f"https://119.147.202.180:{base.PORT}/healthz",
                headers={"Connection": "close"},
            )
            if json.loads(opener.open(request, timeout=3).read()).get("version") == VERSION:
                return
        except Exception as error:
            last_error = error
        time.sleep(1)
    raise RuntimeError(f"v39 direct health check failed: {last_error}")


def main() -> None:
    if not (base.PLATFORM / "PREPARED.json").is_file():
        raise RuntimeError("remaining148 platform is not prepared")
    readonly = base.ROOT / "outputs/football_event_review/test18_completed_readonly_snapshot_20260907"
    if not (readonly / "reviews.sqlite3").is_file():
        raise RuntimeError("completed test18 read-only snapshot is missing")

    base.backup_database(base.PLATFORM / "reviews.sqlite3")
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
        database = base.PLATFORM / "reviews.sqlite3"
        if database.is_file():
            with sqlite3.connect(database) as connection:
                revision = int(connection.execute("SELECT COALESCE(SUM(revision),0) FROM reviews").fetchone()[0])
                revision += int(connection.execute("SELECT COALESCE(SUM(revision),0) FROM video_team_profiles").fetchone()[0])
            if revision != last_revision and time.monotonic() - last_backup >= 600:
                base.backup_database(database)
                last_revision, last_backup = revision, time.monotonic()
        ready = {
            "status": "ready_public_clean_header_colour_names",
            "version": VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "base_url": f"https://119.147.202.180:{base.PORT}",
            "credentials": str((base.PLATFORM / "credentials_private.json").resolve()),
            "database_revision_sum": last_revision,
            "proxy_environment_removed": True,
            "visible_fixed_seek_control": False,
            "semantic_team_colour_names": True,
        }
        temporary = base.PLATFORM / "READY.json.tmp"
        temporary.write_text(json.dumps(ready, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(base.PLATFORM / "READY.json")
        time.sleep(60)


if __name__ == "__main__":
    main()
