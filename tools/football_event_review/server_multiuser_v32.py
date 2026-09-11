#!/usr/bin/env python3
"""Multi-user v31 service with atomic low-bandwidth review-proxy fallback."""

from __future__ import annotations

import argparse
import html
import mimetypes
import ssl
import time
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import server as base
import server_multiuser_v31 as v31


class ReviewThreadingHTTPServer(ThreadingHTTPServer):
    """Keep slow TLS/video clients from starving the accept loop."""

    daemon_threads = True
    request_queue_size = 128

    def get_request(self):
        request, client_address = super().get_request()
        # Bounds abandoned TLS handshakes and clients that stop reading a
        # video range. Active transfers are unaffected while bytes flow.
        request.settimeout(15)
        return request, client_address


class ProxyReviewStore(v31.SecondConfirmationStore):
    def __init__(self, manifest_path: Path, db_path: Path, access_config_path: Path, proxy_root: Path):
        self.proxy_root = proxy_root.resolve()
        self.proxy_root.mkdir(parents=True, exist_ok=True)
        super().__init__(manifest_path, db_path, access_config_path)

    def media_path(self, video_id: str, *, original: bool = False) -> tuple[Path, str]:
        video = self.videos.get(video_id)
        if video is None:
            raise KeyError(video_id)
        proxy = self.proxy_root / f"{video_id}.mp4"
        if not original and proxy.is_file() and proxy.stat().st_size > 1024 * 1024:
            return proxy, "proxy"
        return Path(video["video_path"]), "original"


class ProxyReviewHandler(v31.SecondConfirmationHandler):
    @property
    def store(self) -> ProxyReviewStore:
        return self.server.store  # type: ignore[attr-defined]

    def send_portal(self) -> None:
        links = "".join(
            f'<li><a href="/u/{html.escape(str(user["slug"]), quote=True)}">'
            f'{html.escape(str(user["display_name"]))}</a></li>'
            for user in self.store.users_by_slug.values()
        )
        body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>足球事件质检登录</title><style>
body{{font-family:system-ui,sans-serif;max-width:640px;margin:10vh auto;padding:24px;background:#f5f7fa}}
main{{background:white;padding:28px;border-radius:14px;box-shadow:0 8px 30px #0001}}
li{{margin:14px 0}}a{{font-size:18px;color:#0759c7}}
</style></head><body><main><h1>足球事件质检平台</h1>
<p>请选择分配给您的质检账号，然后输入密码。</p><ul>{links}</ul></main></body></html>""".encode("utf-8")
        self.send_bytes(body, "text/html; charset=utf-8", headers={"Cache-Control": "no-store"})

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/" and self.current_user() is None:
            return self.send_portal()
        return super().do_GET()

    def send_media(self, video_id: str) -> None:
        query = parse_qs(urlparse(self.path).query)
        try:
            path, variant = self.store.media_path(
                video_id, original=query.get("quality", [""])[0] == "original"
            )
        except KeyError:
            return self.send_json({"error": "Unknown video"}, HTTPStatus.NOT_FOUND)
        if not path.is_file():
            return self.send_json({"error": f"Video missing: {path}"}, HTTPStatus.NOT_FOUND)

        file_size = path.stat().st_size
        start, end, status = 0, file_size - 1, HTTPStatus.OK
        range_header = self.headers.get("Range", "")
        headers = {"Accept-Ranges": "bytes", "X-Review-Media-Variant": variant}
        if range_header.startswith("bytes="):
            raw_start, raw_end = range_header[6:].split("-", 1)
            if raw_start:
                start = min(int(raw_start), file_size - 1)
            if raw_end:
                end = min(int(raw_end), start + base.MAX_OPEN_ENDED_RANGE_BYTES - 1, file_size - 1)
            else:
                end = min(start + base.MAX_OPEN_ENDED_RANGE_BYTES - 1, file_size - 1)
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
            remaining, sent, started_at = length, 0, time.monotonic()
            while remaining > 0:
                chunk = handle.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)
                sent += len(chunk)
                delay = sent / base.MEDIA_RATE_LIMIT_BYTES_PER_SEC - (time.monotonic() - started_at)
                if delay > 0:
                    time.sleep(min(delay, 0.25))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--access-config", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = ProxyReviewStore(args.manifest, args.db, args.access_config, args.proxy_root)
    server = ReviewThreadingHTTPServer((args.host, args.port), ProxyReviewHandler)
    server.store = store  # type: ignore[attr-defined]
    if bool(args.tls_cert) != bool(args.tls_key):
        raise ValueError("--tls-cert and --tls-key must be provided together")
    scheme = "http"
    if args.tls_cert and args.tls_key:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.tls_cert, args.tls_key)
        # A listening SSLSocket normally performs the TLS handshake inside
        # accept(), before ThreadingHTTPServer can create a worker. One slow
        # or abandoned connection then blocks every reviewer. Lazy handshake
        # moves that work into the request thread.
        server.socket = context.wrap_socket(
            server.socket,
            server_side=True,
            do_handshake_on_connect=False,
        )
        server.is_tls = True  # type: ignore[attr-defined]
        scheme = "https"
    else:
        server.is_tls = False  # type: ignore[attr-defined]
    print(f"Multi-user football review UI v32: {scheme}://{args.host}:{args.port}", flush=True)
    print(f"Review proxy root: {store.proxy_root}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
