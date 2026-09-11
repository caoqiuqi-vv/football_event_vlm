#!/usr/bin/env python3
"""Reproduce v39 concurrency/HTTP issues using disposable data and loopback only.

This is a diagnostic, not a production load test or a passing regression suite.
It never opens a deployment database or sends requests to port 8775.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import secrets
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
from pathlib import Path
import platform
import socket
import sqlite3
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/football_event_review"))
import server_multiuser_v39 as v39
import server_multiuser_v30 as v30
import server_hierarchical as hierarchical


def summary(values: list[float]) -> dict:
    values = sorted(values)
    return {"count": len(values), **{
        name: round(values[max(0, math.ceil(len(values) * fraction) - 1)], 3)
        for name, fraction in (("p50_ms", .5), ("p95_ms", .95), ("max_ms", 1))
    }}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events-per-video", type=int, default=200)
    parser.add_argument("--version", choices=("v39", "v40"), default="v39")
    args = parser.parse_args()
    if args.version == "v40":
        import server_multiuser_v40 as release
    else:
        release = v39
    if args.events_per_video < 32:
        parser.error("--events-per-video must be at least 32")
    report = {"version": args.version, "python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
              "transport": "loopback HTTP; synthetic bytes, no decoding, no WAN/TLS",
              "users": 4, "events_per_video": args.events_per_video}
    with tempfile.TemporaryDirectory(prefix="review-ui-audit-") as directory:
        root = Path(directory)
        media = root / "video.mp4"
        media.write_bytes(bytes(range(256)) * 8192)
        users, videos = [], []
        salt = "01" * 16
        digest = v30.MultiUserReviewStore.password_digest("test-only", salt)
        for index in range(4):
            vid = f"video-{index}"
            users.append(dict(user_id=f"user-{index}", slug=f"user-{index}",
                              display_name=f"Test {index}", password_salt=salt,
                              password_hash=digest, video_ids=[vid]))
            events = [dict(id=f"{vid}-event-{j}", video_id=vid, segment_id=f"{vid}-segment-{j}",
                           label="shot", time_sec=j * 5 + 10, start_sec=j * 5 + 5,
                           end_sec=j * 5 + 15, score=.9)
                      for j in range(args.events_per_video)]
            videos.append(dict(video_id=vid, video_path=str(media),
                               duration_sec=args.events_per_video * 5 + 30, events=events))
        manifest, access = root / "manifest.json", root / "access.json"
        manifest.write_text(json.dumps(dict(labels=["shot", "save", "set_piece"], videos=videos)))
        access.write_text(json.dumps(dict(users=users)))
        store_class = release.ReliableReviewStore if args.version == "v40" else v39.v36.TeamCalibrationStore
        store = store_class(manifest, root / "reviews.sqlite3", access, root / "proxy")

        def payload(event_id: str) -> dict:
            return dict(operation_id=secrets.token_hex(16), selected_labels=["shot"], active_label="shot", compact_response=True,
                        attribution_by_label={"shot": {"event_team": "teamA"}},
                        expected_revisions={event_id: store._event_by_id(event_id)["review"]["revision"]})

        # Same revision in two tabs: exactly one should commit in this single process.
        event_id = "video-0-event-0"
        shared = payload(event_id)
        barrier = threading.Barrier(2)

        def conflict_worker(_: int) -> str:
            barrier.wait()
            try:
                store.update_segment_for_user(users[0], event_id, {**shared, "operation_id": secrets.token_hex(16)})
                return "committed"
            except v30.RevisionConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            report["same_revision_two_tabs"] = sorted(pool.map(conflict_worker, range(2)))

        # Invalid late-stage validation should not leave a committed review.
        event_id = "video-0-event-1"
        bad = payload(event_id)
        bad["selected_labels"] = ["shot", "back_pass"]
        bad["attribution_by_label"] = {"shot": {"event_team": "teamA"}, "back_pass": {"field_side": "INVALID"}}
        before = store._event_by_id(event_id)["review"]
        try:
            store.update_segment_for_user(users[0], event_id, bad)
            error = None
        except ValueError as exc:
            error = str(exc)
        after = store._event_by_id(event_id)["review"]
        report["late_validation_atomicity"] = dict(error=error, before_status=before["status"],
            after_status=after["status"], before_revision=before["revision"], after_revision=after["revision"])

        original_connect = store.connect
        statements = []

        @contextmanager
        def traced_connect():
            with original_connect() as connection:
                connection.set_trace_callback(statements.append)
                yield connection

        token = store.create_session(users[0]["user_id"])
        store.connect = traced_connect
        for _ in range(10):
            store.session_user(token)
        report["session_lookup"] = dict(lookups=10, writes=sum(
            sql.lstrip().upper().startswith("UPDATE REVIEW_SESSIONS") for sql in statements))
        statements.clear()
        event_id = "video-0-event-2"
        store.update_segment_for_user(users[0], event_id, payload(event_id))
        report["single_save_sql"] = dict(statements=len(statements), commits=sum(
            sql.strip().upper() == "COMMIT" for sql in statements))
        store.connect = original_connect

        # Measure application saves only, without network or media.
        report["store_save_benchmark"] = []
        for concurrency in (1, 4, 8, 16):
            def save_worker(index: int) -> float:
                user_index = index % 4
                eid = f"video-{user_index}-event-{10 + index // 4}"
                data = payload(eid)
                started = time.perf_counter()
                store.update_segment_for_user(users[user_index], eid, data)
                return (time.perf_counter() - started) * 1000

            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                durations = list(pool.map(save_worker, range(32)))
            report["store_save_benchmark"].append(dict(concurrency=concurrency,
                wall_sec=round(time.perf_counter() - start, 3), **summary(durations)))

        handler_class = release.ReliableReviewHandler if args.version == "v40" else v39.CleanReviewHandler
        class QuietHandler(handler_class):
            def log_message(self, *args):
                pass

        server_class = release.ReliableHTTPServer if args.version == "v40" else v39.v32.ReviewThreadingHTTPServer
        server = server_class(("127.0.0.1", 0), QuietHandler)
        server.store = store
        server.is_tls = False
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cookie = f"{v30.SESSION_COOKIE}={token}"

        def request(path: str, headers: dict | None = None, method="GET", body=None):
            connection = http.client.HTTPConnection(*server.server_address, timeout=.5)
            try:
                connection.request(method, path, body=body, headers=headers or {})
                response = connection.getresponse()
                result = dict(status=response.status, content_length=response.getheader("Content-Length"),
                              content_range=response.getheader("Content-Range"),
                              connection=response.getheader("Connection"))
                try:
                    data = response.read()
                    result.update(body_bytes=len(data), body_read_timeout=False)
                except (TimeoutError, socket.timeout):
                    result["body_read_timeout"] = True
                return result
            finally:
                connection.close()

        try:
            report["http"] = {
                "health": request("/healthz"),
                "authenticated_redirect": request("/u/user-0", {"Cookie": cookie}),
                "login_redirect": request("/auth/login", {"Content-Type": "application/x-www-form-urlencoded"},
                                          "POST", "slug=user-0&password=test-only"),
                "range": request("/media/video-0", {"Cookie": cookie, "Range": "bytes=100-199"}),
                "invalid_range": request("/media/video-0", {"Cookie": cookie, "Range": "bytes=99999999-"}),
                "zero_suffix_range": request("/media/video-0", {"Cookie": cookie, "Range": "bytes=-0"}),
                "cross_user_media": request("/media/video-1", {"Cookie": cookie}),
            }
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        # Materialize the effective JS after all historical patches, for inspection.
        generated = hierarchical.patch_hierarchical_js(
            (ROOT / "tools/football_event_review/static/app.js").read_text())
        report["effective_js"] = dict(bytes=len(generated.encode()),
            abort_controller="AbortController" in generated, beforeunload="beforeunload" in generated,
            indexed_db="indexedDB" in generated, durable_local_storage="football-review-v40:" in generated)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".effective.js").write_text(generated)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
