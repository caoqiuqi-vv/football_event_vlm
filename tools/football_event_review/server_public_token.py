#!/usr/bin/env python3
"""Capability-token protected launcher for externally reachable review UI."""

from __future__ import annotations

import argparse
import hmac
import os
import threading
import webbrowser
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import server as base
import server_hierarchical as hierarchical
import server_hierarchical_v4  # noqa: F401  # installs compact hierarchical UI patches


COOKIE_NAME = "football_review_access"


class TokenProtectedHandler(hierarchical.HierarchicalReviewHandler):
    server_version = "FootballEventReview/2.1"

    def has_access(self) -> bool:
        expected = self.server.access_token  # type: ignore[attr-defined]
        parsed = urlparse(self.path)
        supplied = parse_qs(parsed.query).get("token", [""])[0]
        if supplied and hmac.compare_digest(supplied, expected):
            clean_query = [
                (key, value)
                for key, values in parse_qs(parsed.query, keep_blank_values=True).items()
                if key != "token"
                for value in values
            ]
            location = parsed.path + (("?" + urlencode(clean_query)) if clean_query else "")
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", location or "/")
            self.send_header(
                "Set-Cookie",
                f"{COOKIE_NAME}={expected}; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800",
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return True
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        value = cookie.get(COOKIE_NAME)
        return bool(value and hmac.compare_digest(value.value, expected))

    def deny(self) -> None:
        body = "访问令牌无效或已过期。请使用完整的授权链接。".encode("utf-8")
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self.has_access():
            return self.deny()
        # A valid query token just emitted its cookie/redirect response.
        if "token=" in urlparse(self.path).query:
            return
        return super().do_GET()

    def do_POST(self) -> None:
        if not self.has_access():
            return self.deny()
        return super().do_POST()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--access-token", default=os.environ.get("FOOTBALL_REVIEW_TOKEN", ""))
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if len(args.access_token) < 24:
        raise SystemExit("--access-token must contain at least 24 characters")
    store = hierarchical.HierarchicalReviewStore(base.Path(args.manifest), base.Path(args.db))
    server = ThreadingHTTPServer((args.host, args.port), TokenProtectedHandler)
    server.store = store  # type: ignore[attr-defined]
    server.access_token = args.access_token  # type: ignore[attr-defined]
    print(f"Protected football review UI listening on {args.host}:{args.port}", flush=True)
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}/?token={args.access_token}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
