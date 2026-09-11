#!/usr/bin/env python3
"""Password-protected, video-assigned multi-user football review service."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import html
import json
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import server_hierarchical as hierarchical
import server_hierarchical_v30  # noqa: F401  # installs evidence-safe complete chain


SESSION_COOKIE = "football_review_session_v30"
SESSION_SECONDS = 12 * 60 * 60
MAX_LOGIN_ATTEMPTS = 8
LOGIN_WINDOW_SECONDS = 15 * 60

_BaseStore = hierarchical.HierarchicalReviewStore
_BaseHandler = hierarchical.HierarchicalReviewHandler
_html_v29 = hierarchical.patch_hierarchical_html
_js_v29 = hierarchical.patch_hierarchical_js


class RevisionConflict(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MultiUserReviewStore(_BaseStore):
    def __init__(self, manifest_path: Path, db_path: Path, access_config_path: Path):
        self.access_config_path = access_config_path.resolve()
        self.access_config = json.loads(self.access_config_path.read_text(encoding="utf-8"))
        self.users_by_id: dict[str, dict] = {}
        self.users_by_slug: dict[str, dict] = {}
        super().__init__(manifest_path, db_path)
        self._initialize_access()

    def _initialize_access(self) -> None:
        users = self.access_config.get("users", [])
        if not users:
            raise RuntimeError("access config has no users")
        assigned: dict[str, str] = {}
        for user in users:
            required = {"user_id", "slug", "display_name", "password_salt", "password_hash", "video_ids"}
            missing = required - set(user)
            if missing:
                raise RuntimeError(f"access user missing fields: {sorted(missing)}")
            if "password" in user:
                raise RuntimeError("plaintext password must not be stored in access config")
            user_id = str(user["user_id"])
            slug = str(user["slug"])
            if user_id in self.users_by_id or slug in self.users_by_slug:
                raise RuntimeError(f"duplicate user id or slug: {user_id}/{slug}")
            self.users_by_id[user_id] = user
            self.users_by_slug[slug] = user
            for video_id in user["video_ids"]:
                video_id = str(video_id)
                if video_id in assigned:
                    raise RuntimeError(f"video assigned twice: {video_id}")
                assigned[video_id] = user_id
        missing_videos = sorted(set(self.videos) - set(assigned))
        extra_videos = sorted(set(assigned) - set(self.videos))
        if missing_videos or extra_videos:
            raise RuntimeError(
                f"assignment/manifest mismatch missing={missing_videos} extra={extra_videos}"
            )

        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_users (
                    user_id TEXT PRIMARY KEY,
                    slug TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES review_users(user_id),
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS review_sessions_expiry_idx
                    ON review_sessions(expires_at);
                CREATE TABLE IF NOT EXISTS video_assignments (
                    video_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES review_users(user_id),
                    assigned_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS video_assignments_user_idx
                    ON video_assignments(user_id, video_id);
                CREATE TABLE IF NOT EXISTS submission_audit (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS submission_audit_video_idx
                    ON submission_audit(video_id, created_at);
                """
            )
            for user in users:
                connection.execute(
                    """INSERT INTO review_users
                       (user_id, slug, display_name, password_salt, password_hash, active, updated_at)
                       VALUES (?, ?, ?, ?, ?, 1, ?)
                       ON CONFLICT(user_id) DO UPDATE SET
                         slug=excluded.slug, display_name=excluded.display_name,
                         password_salt=excluded.password_salt,
                         password_hash=excluded.password_hash, active=1,
                         updated_at=excluded.updated_at""",
                    (
                        str(user["user_id"]), str(user["slug"]), str(user["display_name"]),
                        str(user["password_salt"]), str(user["password_hash"]), utc_now(),
                    ),
                )
            connection.execute(
                "UPDATE review_users SET active=0 WHERE user_id NOT IN (%s)"
                % ",".join("?" for _ in users),
                tuple(str(user["user_id"]) for user in users),
            )
            for video_id, user_id in assigned.items():
                connection.execute(
                    """INSERT INTO video_assignments(video_id, user_id, assigned_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(video_id) DO UPDATE SET
                         user_id=excluded.user_id, assigned_at=excluded.assigned_at""",
                    (video_id, user_id, utc_now()),
                )
            connection.execute(
                "DELETE FROM video_assignments WHERE video_id NOT IN (%s)"
                % ",".join("?" for _ in assigned),
                tuple(assigned),
            )

    @staticmethod
    def password_digest(password: str, salt_hex: str) -> str:
        return hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=2**14, r=8, p=1, dklen=32,
        ).hex()

    def authenticate(self, slug: str, password: str) -> dict | None:
        user = self.users_by_slug.get(slug)
        if user is None:
            # Keep unknown-user timing close to the normal password path.
            self.password_digest(password, "00" * 16)
            return None
        actual = self.password_digest(password, str(user["password_salt"]))
        if not hmac.compare_digest(actual, str(user["password_hash"])):
            return None
        return user

    def create_session(self, user_id: str) -> str:
        raw = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw.encode()).hexdigest()
        current = time.time()
        with self.connect() as connection:
            connection.execute("DELETE FROM review_sessions WHERE expires_at < ?", (current,))
            connection.execute(
                """INSERT INTO review_sessions
                   (token_hash, user_id, created_at, expires_at, last_seen_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (token_hash, user_id, current, current + SESSION_SECONDS, current),
            )
        return raw

    def session_user(self, raw_token: str) -> dict | None:
        if not raw_token:
            return None
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        current = time.time()
        with self.connect() as connection:
            row = connection.execute(
                """SELECT u.user_id, u.slug, u.display_name, s.expires_at
                   FROM review_sessions s JOIN review_users u ON u.user_id=s.user_id
                   WHERE s.token_hash=? AND s.expires_at>? AND u.active=1""",
                (token_hash, current),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE review_sessions SET last_seen_at=? WHERE token_hash=?",
                (current, token_hash),
            )
        return dict(row)

    def revoke_session(self, raw_token: str) -> None:
        if not raw_token:
            return
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM review_sessions WHERE token_hash=?",
                (hashlib.sha256(raw_token.encode()).hexdigest(),),
            )

    def allowed_video_ids(self, user_id: str) -> set[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT video_id FROM video_assignments WHERE user_id=?", (user_id,)
            ).fetchall()
        return {str(row["video_id"]) for row in rows}

    def owns_video(self, user_id: str, video_id: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM video_assignments WHERE user_id=? AND video_id=?",
                (user_id, video_id),
            ).fetchone()
        return row is not None

    def event_video_id(self, event_id: str) -> str:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT video_id FROM events WHERE id=?", (event_id,)
            ).fetchone()
        if row is None:
            raise KeyError(event_id)
        return str(row["video_id"])

    def bootstrap_for_user(self, user: dict) -> dict:
        result = super().bootstrap()
        allowed = self.allowed_video_ids(str(user["user_id"]))
        result["videos"] = [video for video in result["videos"] if video["video_id"] in allowed]
        placeholders = ",".join("?" for _ in allowed)
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT r.status, COUNT(*) AS count
                    FROM reviews r JOIN events e ON e.id=r.event_id
                    WHERE e.video_id IN ({placeholders}) GROUP BY r.status""",
                tuple(sorted(allowed)),
            ).fetchall()
        result["status_counts"] = {
            status: next((int(row["count"]) for row in rows if row["status"] == status), 0)
            for status in ("unreviewed", "accepted", "modified", "deleted")
        }
        result["current_user"] = {
            "user_id": user["user_id"],
            "display_name": user["display_name"],
            "assigned_videos": len(allowed),
        }
        return result

    def video_payload_for_user(self, user_id: str, video_id: str) -> dict:
        if not self.owns_video(user_id, video_id):
            raise PermissionError(video_id)
        payload = super().video_payload(video_id)
        # Evaluation/GT matching is administrator-only. Reviewers see evidence,
        # not whether an old annotation happened to match the model.
        for event in payload.get("events", []):
            event.pop("matching_gt_times", None)
            event.pop("evaluation_status", None)
            event.pop("suggested_secondary_labels", None)
        return payload

    def _write_audit(
        self,
        *,
        request_id: str,
        user_id: str,
        video_id: str,
        segment_id: str,
        action: str,
        payload: dict,
        result: dict,
        status: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO submission_audit
                   (request_id, user_id, video_id, segment_id, action,
                    payload_json, result_json, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id, user_id, video_id, segment_id, action,
                    json.dumps(payload, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False), status, utc_now(),
                ),
            )

    def update_segment_for_user(self, user: dict, event_id: str, payload: dict) -> dict:
        user_id = str(user["user_id"])
        video_id = self.event_video_id(event_id)
        if not self.owns_video(user_id, video_id):
            raise PermissionError(video_id)
        anchor = self._event_by_id(event_id)
        segment_id = str(anchor.get("segment_id") or event_id)
        expected = payload.get("expected_revisions")
        if not isinstance(expected, dict):
            raise ValueError("missing expected_revisions")
        request_id = secrets.token_hex(16)
        clean_payload = dict(payload)
        clean_payload["reviewer"] = str(user["display_name"])
        with self.lock:
            group = [
                event for event in self.events_for_video(video_id)
                if str(event.get("segment_id") or event["id"]) == segment_id
            ]
            stale = {
                event["id"]: {
                    "expected": expected.get(event["id"]),
                    "actual": int(event["review"].get("revision") or 0),
                }
                for event in group
                if expected.get(event["id"]) is None
                or int(expected[event["id"]]) != int(event["review"].get("revision") or 0)
            }
            if stale:
                self._write_audit(
                    request_id=request_id, user_id=user_id, video_id=video_id,
                    segment_id=segment_id, action="segment-decision",
                    payload=clean_payload, result={"stale": stale}, status="conflict",
                )
                raise RevisionConflict(f"stale revision: {stale}")
            try:
                # Dispatch through self so later schema versions can add review
                # outcomes without bypassing the v30 authorization/revision guard.
                result = self.update_segment_labels(event_id, clean_payload)
            except Exception as error:
                self._write_audit(
                    request_id=request_id, user_id=user_id, video_id=video_id,
                    segment_id=segment_id, action="segment-decision",
                    payload=clean_payload, result={"error": str(error)}, status="failed",
                )
                raise
            self._write_audit(
                request_id=request_id, user_id=user_id, video_id=video_id,
                segment_id=segment_id, action="segment-decision",
                payload=clean_payload, result=result, status="committed",
            )
        result["request_id"] = request_id
        return result


def patch_html_v30(text: str) -> str:
    text = _html_v29(text).replace(
        "hierarchical-v30-evidence-safe-review-20260902",
        "multiuser-v30-video-assigned-20260902",
        1,
    )
    old = '<label class="reviewer">质检员 <input id="reviewerInput" placeholder="姓名/工号" /></label>'
    new = '<span class="reviewer account-badge" id="userIdentity">质检员</span><input id="reviewerInput" type="hidden" /><a class="ghost-btn" href="/logout">退出</a>'
    if old not in text:
        raise RuntimeError("v30 reviewer anchor missing")
    text = text.replace(old, new, 1)
    text = text.replace('<a class="ghost-btn" href="/api/export?format=csv">导出 CSV</a>', "", 1)
    text = text.replace('<a class="ghost-btn" href="/api/export?format=json">导出 JSON</a>', "", 1)
    return text


def patch_js_v30(text: str) -> str:
    text = _js_v29(text)
    old = '''  $("#reviewerInput").value = localStorage.getItem("football-reviewer") || "";
  $("#reviewerInput").onchange = (event) => localStorage.setItem("football-reviewer", event.target.value);'''
    new = '''  const currentUser = state.bootstrap.current_user || {};
  $("#reviewerInput").value = currentUser.display_name || "";
  $("#userIdentity").textContent = `${currentUser.display_name || "质检员"} · ${currentUser.assigned_videos || 0} 个视频`;'''
    if old not in text:
        raise RuntimeError("v30 reviewer initialization anchor missing")
    text = text.replace(old, new, 1)
    revision_anchor = "    compact_response: true,\n"
    if revision_anchor not in text:
        raise RuntimeError("v30 revision payload anchor missing")
    text = text.replace(
        revision_anchor,
        '''    compact_response: true,
    expected_revisions: Object.fromEntries(segmentEvents(event).map((item) => [item.id, Number(item.review.revision || 0)])),
''',
        1,
    )
    return text


hierarchical.patch_hierarchical_html = patch_html_v30
hierarchical.patch_hierarchical_js = patch_js_v30
hierarchical.HIERARCHICAL_CSS += """
.account-badge { padding:7px 10px; border:1px solid #405062; border-radius:7px; background:#17222d; }
"""


LOGIN_HTML = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>足球事件质检登录</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0b1118;color:#e7edf4;font:15px system-ui}form{width:min(360px,calc(100vw - 48px));padding:28px;border:1px solid #354454;border-radius:14px;background:#131d27;box-shadow:0 20px 60px #0008}h1{font-size:21px;margin:0 0 8px}p{color:#91a0af;margin:0 0 20px}input{box-sizing:border-box;width:100%;padding:12px;margin:8px 0 14px;color:#fff;border:1px solid #46576a;border-radius:8px;background:#0d151e}button{width:100%;padding:12px;border:0;border-radius:8px;background:#d7ff38;color:#111;font-weight:800;cursor:pointer}.error{color:#ff8999}</style></head>
<body><form method="post" action="/auth/login"><h1>足球事件质检台</h1><p>{display_name}</p>{error}<input type="hidden" name="slug" value="{slug}"><label>访问密码<input type="password" name="password" required autofocus autocomplete="current-password"></label><button type="submit">安全登录</button></form></body></html>"""


class MultiUserReviewHandler(_BaseHandler):
    server_version = "FootballEventReview/3.0"
    login_attempts: dict[str, deque[float]] = defaultdict(deque)
    login_lock = threading.Lock()

    @property
    def store(self) -> MultiUserReviewStore:
        return self.server.store  # type: ignore[attr-defined]

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; media-src 'self' blob:; img-src 'self' data:; "
            "script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
        )
        if urlparse(self.path).path.startswith("/media/"):
            self.send_header("Cache-Control", "private, max-age=86400")
        super().end_headers()

    def session_token(self) -> str:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        item = cookie.get(SESSION_COOKIE)
        return item.value if item else ""

    def current_user(self) -> dict | None:
        return self.store.session_user(self.session_token())

    def send_login(self, slug: str, error: str = "") -> None:
        user = self.store.users_by_slug.get(slug)
        if user is None:
            return self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        body = (
            LOGIN_HTML
            .replace("{slug}", html.escape(slug, quote=True))
            .replace("{display_name}", html.escape(str(user["display_name"])))
            .replace(
                "{error}",
                f'<p class="error">{html.escape(error)}</p>' if error else "",
            )
        ).encode("utf-8")
        self.send_bytes(body, "text/html; charset=utf-8", headers={"Cache-Control": "no-store"})

    def client_key(self, slug: str) -> str:
        address = self.headers.get("CF-Connecting-IP") or self.client_address[0]
        return f"{address}|{slug}"

    def login_allowed(self, slug: str) -> bool:
        key = self.client_key(slug)
        current = time.monotonic()
        with self.login_lock:
            attempts = self.login_attempts[key]
            while attempts and current - attempts[0] > LOGIN_WINDOW_SECONDS:
                attempts.popleft()
            return len(attempts) < MAX_LOGIN_ATTEMPTS

    def record_login_failure(self, slug: str) -> None:
        with self.login_lock:
            self.login_attempts[self.client_key(slug)].append(time.monotonic())

    def require_user(self) -> dict | None:
        user = self.current_user()
        if user is None:
            self.send_json({"error": "Authentication required"}, HTTPStatus.UNAUTHORIZED)
        return user

    def same_origin_post(self) -> bool:
        fetch_site = self.headers.get("Sec-Fetch-Site", "")
        return fetch_site in {"", "same-origin", "none"}

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/healthz":
            return self.send_json({"ok": True, "version": "v30"})
        if path.startswith("/u/"):
            slug = path.split("/", 2)[-1]
            user = self.current_user()
            if user and user.get("slug") == slug:
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", "/")
                self.end_headers()
                return
            return self.send_login(slug)
        if path == "/logout":
            self.store.revoke_session(self.session_token())
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/")
            self.send_header(
                "Set-Cookie", f"{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"
            )
            self.end_headers()
            return
        user = self.require_user()
        if user is None:
            return
        if path == "/api/bootstrap":
            return self.send_json(self.store.bootstrap_for_user(user))
        if path.startswith("/api/videos/"):
            video_id = path.rsplit("/", 1)[-1]
            try:
                return self.send_json(self.store.video_payload_for_user(str(user["user_id"]), video_id))
            except PermissionError:
                return self.send_json({"error": "Forbidden"}, HTTPStatus.FORBIDDEN)
        if path == "/api/export":
            return self.send_json({"error": "Administrator export only"}, HTTPStatus.FORBIDDEN)
        if path.startswith("/media/"):
            video_id = path.rsplit("/", 1)[-1]
            if not self.store.owns_video(str(user["user_id"]), video_id):
                return self.send_json({"error": "Forbidden"}, HTTPStatus.FORBIDDEN)
            return self.send_media(video_id)
        return self.send_static(path)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/auth/login":
            length = min(int(self.headers.get("Content-Length", "0")), 8192)
            form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            slug = form.get("slug", [""])[0]
            password = form.get("password", [""])[0]
            if not self.login_allowed(slug):
                return self.send_login(slug, "尝试次数过多，请 15 分钟后再试")
            user = self.store.authenticate(slug, password)
            if user is None:
                self.record_login_failure(slug)
                time.sleep(0.25)
                return self.send_login(slug, "密码错误")
            raw = self.store.create_session(str(user["user_id"]))
            secure = (
                self.headers.get("X-Forwarded-Proto", "").lower() == "https"
                or bool(getattr(self.server, "is_tls", False))
            )
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/")
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={raw}; Path=/; HttpOnly; SameSite=Strict; "
                f"Max-Age={SESSION_SECONDS}" + ("; Secure" if secure else ""),
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if not self.same_origin_post():
            return self.send_json({"error": "Cross-site request rejected"}, HTTPStatus.FORBIDDEN)
        user = self.require_user()
        if user is None:
            return
        try:
            if path.startswith("/api/events/") and path.endswith("/segment-decision"):
                event_id = path.split("/")[3]
                payload = self.read_json()
                result = self.store.update_segment_for_user(user, event_id, payload)
                return self.send_json(result)
            if path.startswith("/api/events/") and path.endswith("/undo"):
                event_id = path.split("/")[3]
                video_id = self.store.event_video_id(event_id)
                if not self.store.owns_video(str(user["user_id"]), video_id):
                    raise PermissionError(video_id)
                return self.send_json(self.store.undo(event_id))
            return self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except RevisionConflict as error:
            return self.send_json({"error": str(error), "code": "revision_conflict"}, HTTPStatus.CONFLICT)
        except PermissionError:
            return self.send_json({"error": "Forbidden"}, HTTPStatus.FORBIDDEN)
        except (ValueError, json.JSONDecodeError) as error:
            return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except KeyError as error:
            return self.send_json({"error": f"Not found: {error}"}, HTTPStatus.NOT_FOUND)
        except Exception as error:
            return self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--access-config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = MultiUserReviewStore(args.manifest, args.db, args.access_config)
    server = ThreadingHTTPServer((args.host, args.port), MultiUserReviewHandler)
    server.store = store  # type: ignore[attr-defined]
    print(f"Multi-user football review UI: http://{args.host}:{args.port}", flush=True)
    print(f"Review database: {store.db_path}", flush=True)
    print(f"Assigned users: {len(store.users_by_id)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
