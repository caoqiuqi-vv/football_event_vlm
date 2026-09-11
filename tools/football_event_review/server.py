#!/usr/bin/env python3
"""Local HTTP server for reviewing DINO football event proposals."""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import sqlite3
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


# Chrome commonly asks for an open-ended byte range. Sending the whole
# remainder of a multi-GB match monopolizes an SSH port-forward and makes
# VSCode Remote unresponsive. Eight MiB covers roughly one review clip while
# keeping seeks cheap; pacing leaves interactive SSH traffic headroom.
MAX_OPEN_ENDED_RANGE_BYTES = 8 * 1024 * 1024
MEDIA_RATE_LIMIT_BYTES_PER_SEC = 4 * 1024 * 1024


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
VALID_STATUSES = {"unreviewed", "accepted", "deleted", "modified"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    def __init__(self, manifest_path: Path, db_path: Path):
        self.manifest_path = manifest_path.resolve()
        self.db_path = db_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.videos = {item["video_id"]: item for item in self.manifest["videos"]}
        self.model_labels = list(self.manifest.get("labels", ["shot", "save", "set_piece"]))
        self.review_labels = list(
            self.manifest.get(
                "review_labels",
                ["shot", "save", "free_kick", "penalty", "corner", "shot_on_target"],
            )
        )
        self.lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    source_time_sec REAL NOT NULL,
                    source_score REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_video_idx ON events(video_id, source_time_sec);
                CREATE TABLE IF NOT EXISTS reviews (
                    event_id TEXT PRIMARY KEY REFERENCES events(id),
                    status TEXT NOT NULL DEFAULT 'unreviewed',
                    corrected_label TEXT,
                    corrected_time_sec REAL,
                    note TEXT NOT NULL DEFAULT '',
                    reviewer TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS review_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL REFERENCES events(id),
                    revision INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            for video in self.manifest["videos"]:
                for event in video["events"]:
                    connection.execute(
                        """INSERT INTO events
                           (id, video_id, source_label, source_time_sec, source_score, payload_json)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(id) DO UPDATE SET
                             video_id=excluded.video_id,
                             source_label=excluded.source_label,
                             source_time_sec=excluded.source_time_sec,
                             source_score=excluded.source_score,
                             payload_json=excluded.payload_json""",
                        (
                            event["id"],
                            video["video_id"],
                            event["label"],
                            event["time_sec"],
                            event["score"],
                            json.dumps(event, ensure_ascii=False),
                        ),
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO reviews(event_id) VALUES (?)", (event["id"],)
                    )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict:
        payload = json.loads(row["payload_json"])
        payload["review"] = {
            "status": row["status"],
            "corrected_label": row["corrected_label"],
            "corrected_time_sec": row["corrected_time_sec"],
            "note": row["note"],
            "reviewer": row["reviewer"],
            "revision": row["revision"],
            "updated_at": row["updated_at"],
        }
        return payload

    def events_for_video(self, video_id: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT e.*, r.status, r.corrected_label, r.corrected_time_sec,
                          r.note, r.reviewer, r.revision, r.updated_at
                   FROM events e JOIN reviews r ON r.event_id=e.id
                   WHERE e.video_id=? ORDER BY e.source_time_sec, e.source_label""",
                (video_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def bootstrap(self) -> dict:
        with self.connect() as connection:
            status_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM reviews GROUP BY status"
            ).fetchall()
            per_video = {
                row["video_id"]: {
                    "total": row["total"],
                    "reviewed": row["reviewed"],
                }
                for row in connection.execute(
                    """SELECT e.video_id, COUNT(*) AS total,
                              SUM(CASE WHEN r.status != 'unreviewed' THEN 1 ELSE 0 END) AS reviewed
                       FROM events e JOIN reviews r ON r.event_id=e.id GROUP BY e.video_id"""
                )
            }
        videos = []
        for video_id, video in self.videos.items():
            counts = per_video.get(video_id, {"total": 0, "reviewed": 0})
            videos.append(
                {
                    "video_id": video_id,
                    "duration_sec": video.get("duration_sec", 0),
                    "total": counts["total"],
                    "reviewed": counts["reviewed"],
                }
            )
        status_counts = {status: 0 for status in VALID_STATUSES}
        status_counts.update({row["status"]: row["count"] for row in status_rows})
        return {
            "labels": self.model_labels,
            "model_labels": self.model_labels,
            "review_labels": self.review_labels,
            "videos": videos,
            "status_counts": status_counts,
            "source": self.manifest.get("source", {}),
        }

    def video_payload(self, video_id: str) -> dict:
        if video_id not in self.videos:
            raise KeyError(video_id)
        video = self.videos[video_id]
        return {
            "video_id": video_id,
            "duration_sec": video.get("duration_sec", 0),
            "timeline": video.get("timeline", []),
            "events": self.events_for_video(video_id),
            "media_url": f"/media/{video_id}",
        }

    def update_review(self, event_id: str, data: dict) -> dict:
        status = str(data.get("status", "")).strip()
        if status not in VALID_STATUSES - {"unreviewed"}:
            raise ValueError(f"Invalid review status: {status}")
        corrected_label = data.get("corrected_label")
        corrected_time = data.get("corrected_time_sec")
        if status == "modified":
            if corrected_label is not None and corrected_label not in self.review_labels:
                raise ValueError(f"Invalid corrected label: {corrected_label}")
            if corrected_label is None and corrected_time is None:
                raise ValueError("modified requires corrected_label or corrected_time_sec")
        else:
            corrected_label = None
            corrected_time = None
        if corrected_time is not None:
            corrected_time = max(0.0, float(corrected_time))
        note = str(data.get("note", ""))[:2000]
        reviewer = str(data.get("reviewer", ""))[:120]

        with self.lock, self.connect() as connection:
            current = connection.execute(
                "SELECT * FROM reviews WHERE event_id=?", (event_id,)
            ).fetchone()
            if current is None:
                raise KeyError(event_id)
            connection.execute(
                "INSERT INTO review_history(event_id, revision, state_json, created_at) VALUES (?, ?, ?, ?)",
                (event_id, current["revision"], json.dumps(dict(current)), utc_now()),
            )
            connection.execute(
                """UPDATE reviews SET status=?, corrected_label=?, corrected_time_sec=?,
                          note=?, reviewer=?, revision=revision+1, updated_at=?
                   WHERE event_id=?""",
                (status, corrected_label, corrected_time, note, reviewer, utc_now(), event_id),
            )
            row = connection.execute(
                """SELECT e.*, r.status, r.corrected_label, r.corrected_time_sec,
                          r.note, r.reviewer, r.revision, r.updated_at
                   FROM events e JOIN reviews r ON r.event_id=e.id WHERE e.id=?""",
                (event_id,),
            ).fetchone()
        return self._event_from_row(row)

    def undo(self, event_id: str) -> dict:
        with self.lock, self.connect() as connection:
            previous = connection.execute(
                "SELECT * FROM review_history WHERE event_id=? ORDER BY id DESC LIMIT 1",
                (event_id,),
            ).fetchone()
            if previous is None:
                raise ValueError("No previous review state")
            state = json.loads(previous["state_json"])
            connection.execute(
                """UPDATE reviews SET status=?, corrected_label=?, corrected_time_sec=?, note=?,
                          reviewer=?, revision=?, updated_at=? WHERE event_id=?""",
                (
                    state["status"], state["corrected_label"], state["corrected_time_sec"],
                    state["note"], state["reviewer"], state["revision"], state["updated_at"], event_id,
                ),
            )
            connection.execute("DELETE FROM review_history WHERE id=?", (previous["id"],))
            row = connection.execute(
                """SELECT e.*, r.status, r.corrected_label, r.corrected_time_sec,
                          r.note, r.reviewer, r.revision, r.updated_at
                   FROM events e JOIN reviews r ON r.event_id=e.id WHERE e.id=?""",
                (event_id,),
            ).fetchone()
        return self._event_from_row(row)

    def export_rows(self, include_unreviewed: bool = False) -> list[dict]:
        rows: list[dict] = []
        for video_id in self.videos:
            for event in self.events_for_video(video_id):
                review = event["review"]
                status = review["status"]
                if status == "deleted" or (status == "unreviewed" and not include_unreviewed):
                    continue
                output_label = review["corrected_label"] or event["label"]
                output_time = (
                    review["corrected_time_sec"]
                    if review["corrected_time_sec"] is not None
                    else event["time_sec"]
                )
                rows.append(
                    {
                        "video_id": video_id,
                        "label": output_label,
                        "time_sec": output_time,
                        "score": event["score"],
                        "review_status": status,
                        "source_label": event["label"],
                        "source_time_sec": event["time_sec"],
                        "frame_detection_score": event.get("frame_detection_scores", {}).get(event["label"], 0),
                        "evaluation_status": event.get("evaluation_status", ""),
                        "match_tolerance_sec": event.get("match_tolerance_sec", ""),
                        "matching_gt_times": json.dumps(event.get("matching_gt_times", [])),
                        "reviewer": review["reviewer"],
                        "note": review["note"],
                        "event_id": event["id"],
                    }
                )
        return sorted(rows, key=lambda row: (row["video_id"], row["time_sec"], row["label"]))


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "FootballEventReview/1.0"

    @property
    def store(self) -> ReviewStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def send_bytes(
        self, content: bytes, content_type: str, status: int = 200, headers: dict | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, data: object, status: int = 200) -> None:
        self.send_bytes(
            json.dumps(data, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("Request body too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/bootstrap":
                return self.send_json(self.store.bootstrap())
            if path.startswith("/api/videos/"):
                video_id = path.rsplit("/", 1)[-1]
                return self.send_json(self.store.video_payload(video_id))
            if path == "/api/export":
                query = parse_qs(parsed.query)
                export_format = query.get("format", ["json"])[0]
                include_unreviewed = query.get("include_unreviewed", ["0"])[0] == "1"
                rows = self.store.export_rows(include_unreviewed)
                if export_format == "csv":
                    buffer = io.StringIO()
                    fields = list(rows[0]) if rows else ["video_id", "label", "time_sec"]
                    writer = csv.DictWriter(buffer, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                    return self.send_bytes(
                        buffer.getvalue().encode("utf-8-sig"),
                        "text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=reviewed_events.csv"},
                    )
                return self.send_bytes(
                    json.dumps({"events": rows}, ensure_ascii=False, indent=2).encode("utf-8"),
                    "application/json; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=reviewed_events.json"},
                )
            if path.startswith("/media/"):
                return self.send_media(path.rsplit("/", 1)[-1])
            return self.send_static(path)
        except KeyError as error:
            self.send_json({"error": f"Not found: {error}"}, HTTPStatus.NOT_FOUND)
        except Exception as error:  # local diagnostic tool: make failures visible in UI
            self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            if path.startswith("/api/events/") and path.endswith("/decision"):
                event_id = path.split("/")[3]
                return self.send_json(self.store.update_review(event_id, self.read_json()))
            if path.startswith("/api/events/") and path.endswith("/undo"):
                event_id = path.split("/")[3]
                return self.send_json(self.store.undo(event_id))
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except KeyError as error:
            self.send_json({"error": f"Not found: {error}"}, HTTPStatus.NOT_FOUND)
        except Exception as error:
            self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT not in target.parents and target != STATIC_ROOT:
            return self.send_json({"error": "Invalid path"}, HTTPStatus.BAD_REQUEST)
        if not target.exists() or not target.is_file():
            return self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_bytes(target.read_bytes(), content_type)

    def send_media(self, video_id: str) -> None:
        video = self.store.videos.get(video_id)
        if video is None:
            return self.send_json({"error": "Unknown video"}, HTTPStatus.NOT_FOUND)
        path = Path(video["video_path"])
        if not path.exists():
            return self.send_json({"error": f"Video missing: {path}"}, HTTPStatus.NOT_FOUND)
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, file_size - 1
        status = HTTPStatus.OK
        headers = {"Accept-Ranges": "bytes"}
        if range_header and range_header.startswith("bytes="):
            raw_start, raw_end = range_header[6:].split("-", 1)
            if raw_start:
                start = min(int(raw_start), file_size - 1)
            if raw_end:
                end = min(
                    int(raw_end),
                    start + MAX_OPEN_ENDED_RANGE_BYTES - 1,
                    file_size - 1,
                )
            else:
                end = min(start + MAX_OPEN_ENDED_RANGE_BYTES - 1, file_size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Content-Length", str(length))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            sent = 0
            started_at = time.monotonic()
            while remaining > 0:
                chunk = handle.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Seeking or selecting the next review clip cancels the old
                    # browser request. Stop disk/network work immediately.
                    break
                remaining -= len(chunk)
                sent += len(chunk)
                target_elapsed = sent / MEDIA_RATE_LIMIT_BYTES_PER_SEC
                delay = target_elapsed - (time.monotonic() - started_at)
                if delay > 0:
                    time.sleep(min(delay, 0.25))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("outputs/football_event_review/reviews.sqlite3"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = ReviewStore(args.manifest, args.db)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    server.store = store  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}"
    print(f"Football event review UI: {url}")
    print(f"Review database: {store.db_path}")
    if args.open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
