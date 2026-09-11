#!/usr/bin/env python3
"""Finish queue construction and keep the four-user external review service alive."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sqlite3
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cloudflared", type=Path, default=None)
    parser.add_argument("--expose-public", action="store_true")
    parser.add_argument("--port", type=int, default=8773)
    parser.add_argument("--poll-sec", type=float, default=60.0)
    return parser.parse_args()


def run_checked(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now(timezone.utc).isoformat()}] {shlex.join(command)}\n")
        handle.flush()
        result = subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}: {shlex.join(command)}")


def tmux_alive(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def start_tmux(session: str, command: list[str]) -> None:
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, shlex.join(command)],
        cwd=ROOT, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"failed to start tmux session {session}")


def wait_health(port: int, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            payload = json.loads(
                urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3).read()
            )
            if payload.get("ok"):
                return
        except Exception as error:  # noqa: BLE001 - retained for final diagnostic
            last_error = error
        time.sleep(1)
    raise RuntimeError(f"review service health check failed: {last_error}")


def parse_tunnel_url(log_path: Path, timeout: float = 120.0) -> str:
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_path.is_file():
            matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="replace"))
            if matches:
                return matches[-1]
        time.sleep(1)
    raise RuntimeError(f"Cloudflare tunnel URL not found in {log_path}")


def update_credential_urls(credentials_path: Path, access_path: Path, base_url: str) -> None:
    credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
    access = json.loads(access_path.read_text(encoding="utf-8"))
    slug_by_user = {str(user["user_id"]): str(user["slug"]) for user in access["users"]}
    for user in credentials["users"]:
        user["url"] = f"{base_url.rstrip('/')}/u/{slug_by_user[str(user['user_id'])]}"
    credentials["active_base_url"] = base_url.rstrip("/")
    credentials["url_updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = credentials_path.with_suffix(credentials_path.suffix + ".tmp")
    temporary.write_text(json.dumps(credentials, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, credentials_path)


def sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_db = sqlite3.connect(source)
    target_db = sqlite3.connect(destination)
    try:
        source_db.backup(target_db)
        result = target_db.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"backup integrity check failed: {result}")
    finally:
        target_db.close()
        source_db.close()


def materialize_cached_dense_sources(run_dir: Path) -> None:
    """Expose strictly validated cached videos through the unified run tree."""
    inventory = load_json(run_dir / "full_review_inventory.json")
    for row in inventory["videos"]:
        cache_source = row.get("cache_source")
        if not cache_source:
            continue
        destination = run_dir / str(row["video_id"])
        source = Path(cache_source).resolve()
        if destination.exists() or destination.is_symlink():
            continue
        if not source.is_dir():
            raise RuntimeError(f"cached dense source disappeared: {source}")
        destination.symlink_to(source, target_is_directory=True)


def main() -> None:
    args = parse_args()
    if args.expose_public and args.cloudflared is None:
        raise RuntimeError("--cloudflared is required with --expose-public")
    run_dir = args.run_dir.resolve()
    full_output = run_dir / "review_platform"
    full_output.mkdir(parents=True, exist_ok=True)
    pipeline_log = full_output / "pipeline.log"
    while not (run_dir / "DENSE_COMPLETE.json").is_file():
        time.sleep(max(10.0, args.poll_sec))

    queue = full_output / "review_queue_gt_union_model.jsonl"
    queue_report = full_output / "review_queue_report.json"
    # Production review profile: preserve the agreed high-recall operating point.
    # On test18 this corresponds to about 41% reviewed source duration.
    adaptive_report = full_output / "adaptive_thresholds_loov_90_85_85.json"
    manifest = full_output / "review_manifest.json"
    validation = full_output / "review_manifest_validation.json"
    access = full_output / "access_config.json"
    credentials = full_output / "credentials_private.json"
    database = full_output / "reviews.sqlite3"
    materialize_cached_dense_sources(run_dir)
    if not adaptive_report.is_file():
        run_checked([
            "python", "scripts/analyze_video_adaptive_dense_thresholds.py",
            "--run-dir", str(run_dir),
            "--video-id-file", str(run_dir / "all_video_ids.txt"),
            "--labels", "shot,save,set_piece", "--match-tolerance-sec", "3",
            "--recall-floors", "shot=0.90,save=0.85,set_piece=0.85",
            "--output", str(adaptive_report),
        ], pipeline_log)
    if not queue.is_file() or not queue_report.is_file():
        run_checked([
            "python", "tools/football_event_review/build_full_dense_review_queue.py",
            "--run-dir", str(run_dir),
            "--video-id-file", str(run_dir / "all_video_ids.txt"),
            "--inventory", str(run_dir / "full_review_inventory.json"),
            "--adaptive-report", str(adaptive_report),
            "--output", str(queue), "--report", str(queue_report),
            "--max-segment-sec", "45", "--gt-context-sec", "5",
            "--match-tolerance-sec", "3",
            "--recall-floors", "shot=0.90,save=0.85,set_piece=0.85",
        ], pipeline_log)
    if not manifest.is_file():
        run_checked([
            "python", "tools/football_event_review/build_annotation_repair_manifest.py",
            "--queue-jsonl", str(queue), "--run-dir", str(run_dir),
            "--video-root", str(args.video_root.resolve()),
            "--inventory", str(run_dir / "full_review_inventory.json"),
            "--output", str(manifest),
        ], pipeline_log)
    run_checked([
        "python", "tools/football_event_review/validate_full_review_manifest.py",
        "--manifest", str(manifest), "--run-dir", str(run_dir),
        "--queue", str(queue),
        "--queue-report", str(queue_report),
        "--output", str(validation),
    ], pipeline_log)
    if not access.is_file() or not credentials.is_file():
        run_checked([
            "python", "tools/football_event_review/create_multiuser_access.py",
            "--manifest", str(manifest), "--output-config", str(access),
            "--output-credentials", str(credentials),
            "--base-url", f"http://127.0.0.1:{args.port}",
        ], pipeline_log)

    server_session = "football_full_review_v31_server"
    tunnel_session = "football_full_review_v31_tunnel"
    server_command = [
        "/home/new_users/qiuqi/miniconda3/bin/python",
        "tools/football_event_review/server_multiuser_v31.py",
        "--manifest", str(manifest), "--db", str(database),
        "--access-config", str(access), "--host", "127.0.0.1",
        "--port", str(args.port),
    ]
    tunnel_log = full_output / "cloudflared.log"
    tunnel_command = (
        [
            str(args.cloudflared.resolve()), "tunnel", "--no-autoupdate",
            "--protocol", "http2", "--logfile", str(tunnel_log),
            "--url", f"http://127.0.0.1:{args.port}",
        ]
        if args.expose_public else []
    )
    current_url = "" if args.expose_public else f"http://127.0.0.1:{args.port}"
    last_backup_revision = -1
    last_backup_at = 0.0
    while True:
        if not tmux_alive(server_session):
            start_tmux(server_session, server_command)
        wait_health(args.port)
        if args.expose_public:
            if not tmux_alive(tunnel_session):
                start_tmux(tunnel_session, tunnel_command)
            tunnel_url = parse_tunnel_url(tunnel_log)
            if tunnel_url != current_url:
                update_credential_urls(credentials, access, tunnel_url)
                current_url = tunnel_url
        if database.is_file():
            connection = sqlite3.connect(database)
            try:
                revision = int(connection.execute("SELECT COALESCE(SUM(revision),0) FROM reviews").fetchone()[0])
            finally:
                connection.close()
            if revision != last_backup_revision and time.monotonic() - last_backup_at >= 10 * 60:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                sqlite_backup(database, full_output / "backups" / f"reviews_{stamp}.sqlite3")
                last_backup_revision = revision
                last_backup_at = time.monotonic()
        ready = {
            "status": "ready_public" if args.expose_public else "ready_local",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "url": current_url,
            "credentials": str(credentials),
            "manifest_validation": load_json(validation),
            "database": str(database),
            "database_revision_sum": last_backup_revision,
        }
        temporary = (full_output / "READY.json.tmp")
        temporary.write_text(json.dumps(ready, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, full_output / "READY.json")
        time.sleep(max(60.0, args.poll_sec))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
