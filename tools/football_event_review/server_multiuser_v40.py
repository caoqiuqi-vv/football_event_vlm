#!/usr/bin/env python3
"""Reliable multi-user review: atomic saves, durable client queue and faststart media."""
from __future__ import annotations

import argparse
import email.utils
from http import HTTPStatus
import json
import gzip
import mimetypes
from pathlib import Path
import re
import sqlite3
import ssl
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse

import server as base
import server_hierarchical as hierarchical
import server_multiuser_v30 as v30
import server_multiuser_v32 as v32
import server_multiuser_v39 as v39
from reliable_store import ReliableReviewStore
from attribution_rules import RULES_VERSION

VERSION = 'v40-reliable-review'
MEDIA_REVISION = 'adaptive-forward-buffer-20260910'
MEDIA_WRITE_TIMEOUT = 45
MEDIA_CHUNK_BYTES = 32 * 1024
_html_v39 = hierarchical.patch_hierarchical_html
_js_v39 = hierarchical.patch_hierarchical_js


def patch_html_v40(text):
    text = _html_v39(text).replace('multiuser-v39-clean-header-colour-names-20260908', VERSION + '-' + RULES_VERSION + '-' + MEDIA_REVISION)
    panel = '''<aside id="saveQueuePanel" class="save-queue-panel">
      <button id="saveQueueToggle" type="button"><span id="saveQueueStatus" role="status">正在恢复保存记录…</span></button>
      <button id="retryVideo" type="button" hidden>重新加载视频</button>
      <section id="saveQueueDetails" hidden><p>待保存操作保留在本浏览器。离线后可重新打开页面继续。</p><div id="saveQueueItems"></div></section>
    </aside>'''
    return text.replace('</header>', panel + '</header>', 1)


def patch_js_v40(text):
    text = _js_v39(text)
    anchor = 'init().catch((error) => { console.error(error); toast(`载入失败：${error.message}`); });'
    if text.count(anchor) != 1:
        raise RuntimeError('v40 initialization anchor changed')
    runtime = (base.STATIC_ROOT / 'review_outbox.js').read_text() + '\n' + (base.STATIC_ROOT / 'forward_buffer.js').read_text() + '\n' + (base.STATIC_ROOT / 'reliable_review.js').read_text()
    return text.replace(anchor, runtime + '\n' + anchor)


hierarchical.patch_hierarchical_html = patch_html_v40
hierarchical.patch_hierarchical_js = patch_js_v40
hierarchical.HIERARCHICAL_CSS += '''
.topbar{grid-template-columns:minmax(220px,300px) minmax(100px,1fr) auto auto;gap:16px}
@media(max-width:1000px){.topbar{grid-template-columns:minmax(0,1fr) auto}}
.save-queue-panel{position:relative;flex-shrink:0;z-index:1100;max-width:min(230px,32vw);padding:5px;border:1px solid #567184;border-radius:8px;background:#14212ef5;color:#e4edf6;font:12px system-ui}
.save-queue-panel button{color:inherit;background:#243748;border:1px solid #617b90;border-radius:5px;padding:6px;cursor:pointer}
#saveQueueDetails{position:absolute;right:0;top:100%;width:min(430px,90vw);max-height:55vh;overflow:auto;padding:8px;background:#14212e;border:1px solid #567184;border-radius:8px}#saveQueueItems>div{border-top:1px solid #425767;padding:8px 0;overflow-wrap:anywhere}
#saveQueueItems button{margin:5px 5px 0 0}.save-queue-panel[data-failed="true"]{border-color:#e8a957}
'''


class ReliableHTTPServer(v32.ReviewThreadingHTTPServer):
    def __init__(self, address, handler, max_connections=128, max_media=16):
        self.connection_slots = threading.BoundedSemaphore(max_connections)
        self.media_slots = threading.BoundedSemaphore(max_media)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self.connection_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connection_slots.release()


class ReliableReviewHandler(v39.CleanReviewHandler):
    server_version = 'FootballEventReview/4.0'

    def handle_one_request(self):
        self._user_checked = False
        self._request_user = None
        try:
            return super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, ssl.SSLError):
            self.close_connection = True

    def current_user(self):
        if not getattr(self, '_user_checked', False):
            self._request_user = super().current_user()
            self._user_checked = True
        return self._request_user

    def send_response_only(self, code, message=None):
        self._response_status = int(code)
        self._response_headers = set()
        return super().send_response_only(code, message)

    def send_header(self, keyword, value):
        self._response_headers.add(keyword.lower())
        return super().send_header(keyword, value)

    def end_headers(self):
        if self._response_status in (301, 302, 303, 307, 308) and 'content-length' not in self._response_headers:
            self.send_header('Content-Length', '0')
        return super().end_headers()

    def send_bytes(self, content, content_type, status=200, headers=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(content)

    def send_json(self, payload, status=200):
        content = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        headers = {'Vary': 'Accept-Encoding'}
        accepts_gzip = False
        for entry in self.headers.get('Accept-Encoding', '').lower().split(','):
            parts = [part.strip() for part in entry.split(';')]
            if parts[0] != 'gzip':
                continue
            try:
                quality = next((float(p[2:]) for p in parts[1:] if p.startswith('q=')), 1.0)
                accepts_gzip = quality > 0
            except ValueError:
                pass
        if accepts_gzip and len(content) >= 2048:
            content = gzip.compress(content, compresslevel=3)
            headers['Content-Encoding'] = 'gzip'
        return self.send_bytes(content, 'application/json; charset=utf-8', status, headers)

    def do_HEAD(self):
        if urlparse(self.path).path.startswith('/media/'):
            return self.do_GET()
        self.send_response(405)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/healthz':
            return self.send_json({'ok': True, 'version': VERSION, 'attribution_rules': RULES_VERSION, 'media_revision': MEDIA_REVISION})
        if path == '/readyz':
            media = self.store.media_readiness()
            with self.store.connect() as connection:
                connection.execute("SELECT value FROM review_metadata WHERE key='namespace'").fetchone()
            return self.send_json({'ok': media['ready'], 'version': VERSION, 'media': media},
                                  200 if media['ready'] else 503)
        try:
            return super().do_GET()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except sqlite3.OperationalError:
            return self.send_json({'error': '服务繁忙，请稍后重试'}, 503)

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        try:
            limit = 8192 if path == '/auth/login' else 1024**2
            length = int(self.headers.get('Content-Length', '-1'))
            if self.headers.get('Transfer-Encoding') or not 0 <= length <= limit:
                self.close_connection = True
                return self.send_json({'error': 'Invalid request body length'}, 413)
            if path == '/auth/login':
                return super().do_POST()
            if not self.same_origin_post():
                self.close_connection = True
                return self.send_json({'error': 'Cross-site request rejected'}, 403)
            user = self.require_user()
            if user is None:
                self.close_connection = True
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Incomplete request body')
            payload = json.loads(raw or b'{}', parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Non-finite number')))
            if not isinstance(payload, dict):
                raise ValueError('Expected a JSON object')
            parts = path.split('/')
            if len(parts) == 5 and parts[1:3] == ['api', 'events']:
                if parts[4] == 'segment-decision':
                    result = self.store.update_segment_for_user(user, parts[3], payload)
                elif parts[4] == 'undo':
                    result = self.store.undo_for_user(user, parts[3], payload)
                else:
                    return self.send_json({'error': 'Not found'}, 404)
            elif len(parts) == 5 and parts[1:3] == ['api', 'videos'] and parts[4] == 'team-profile':
                result = self.store.update_team_profile_for_user(user, parts[3], payload)
            else:
                return self.send_json({'error': 'Not found'}, 404)
            return self.send_json(result)
        except v30.RevisionConflict as error:
            return self.send_json({'error': str(error), 'code': 'revision_conflict'}, 409)
        except PermissionError:
            return self.send_json({'error': 'Forbidden'}, 403)
        except KeyError:
            return self.send_json({'error': 'Not found'}, 404)
        except (ValueError, TypeError) as error:
            return self.send_json({'error': str(error)}, 400)
        except sqlite3.OperationalError:
            return self.send_json({'error': '服务繁忙，请稍后重试'}, 503)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:
            return self.send_json({'error': '保存未完成，请重试相同操作'}, 500)

    def send_media(self, video_id):
        query = parse_qs(urlparse(self.path).query)
        path, variant = self.store.media_path(video_id, original=query.get('quality') == ['original'])
        try:
            handle = path.open('rb')
        except FileNotFoundError:
            return self.send_json({'error': 'Video unavailable'}, 404)
        with handle:
            import os
            st = os.fstat(handle.fileno())
            size = st.st_size
            etag = f'"{st.st_mtime_ns:x}-{size:x}"'
            version = self.store.media_version(path, st)
            if query.get('v', [version])[0] != version:
                return self.send_json({'error': 'Video changed; reload this video'}, 409)
            modified = email.utils.formatdate(st.st_mtime, usegmt=True)
            if self.headers.get('If-None-Match') == etag:
                self.send_response(304)
                self.send_header('ETag', etag)
                self.end_headers()
                return
            start, end, status = 0, size - 1, 200
            requested = self.headers.get('Range', '').strip()
            if_range = self.headers.get('If-Range')
            if if_range and if_range not in (etag, modified):
                requested = ''
            if requested:
                try:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
                    if not match or not size:
                        raise ValueError()
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = min(int(last) if last else size - 1, size - 1)
                    else:
                        suffix = int(last)
                        if suffix <= 0:
                            raise ValueError()
                        start = max(0, size - suffix)
                    if not 0 <= start <= end < size:
                        raise ValueError()
                    status = 206
                except ValueError:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
            if not self.server.media_slots.acquire(blocking=False):
                self.send_response(503)
                self.send_header('Retry-After', '1')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            transfer_start = time.monotonic()
            sent = 0
            outcome = "complete"
            previous_timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(MEDIA_WRITE_TIMEOUT)
                self.send_response(status)
                self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'video/mp4')
                self.send_header('Content-Length', str(end - start + 1))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('ETag', etag)
                self.send_header('Cache-Control', 'private, max-age=0, must-revalidate')
                self.send_header('Vary', 'Cookie')
                self.send_header('Last-Modified', modified)
                self.send_header('X-Review-Media-Variant', variant)
                if status == 206:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                if self.command == 'HEAD':
                    return
                handle.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = handle.read(min(MEDIA_CHUNK_BYTES, remaining))
                    if not chunk:
                        self.close_connection = True
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
                    sent += len(chunk)
            except (BrokenPipeError, ConnectionResetError, TimeoutError, ssl.SSLError) as error:
                outcome = type(error).__name__
                self.close_connection = True
            finally:
                try:
                    self.connection.settimeout(previous_timeout)
                except OSError:
                    pass
                self.server.media_slots.release()
                self.log_message("MEDIA video=%s range=%s-%s bytes=%s seconds=%.3f outcome=%s",
                    video_id, start, end, sent, time.monotonic()-transfer_start, outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest', 'db', 'access-config', 'proxy-root'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8775)
    parser.add_argument('--tls-cert', type=Path)
    parser.add_argument('--tls-key', type=Path)
    parser.add_argument('--require-faststart', action='store_true')
    args = parser.parse_args()
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error('TLS certificate and key must be provided together')
    store = ReliableReviewStore(args.manifest, args.db, args.access_config, args.proxy_root)
    readiness = store.media_readiness()
    if args.require_faststart and not readiness['ready']:
        raise RuntimeError(f'Media not ready: {readiness}')
    server = ReliableHTTPServer((args.host, args.port), ReliableReviewHandler)
    server.store = store
    server.is_tls = bool(args.tls_cert)
    if server.is_tls:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
    stop = threading.Event()
    def activity_worker():
        while not stop.wait(60):
            try:
                store.flush_session_activity()
            except sqlite3.Error:
                pass
    worker = threading.Thread(target=activity_worker, daemon=True)
    worker.start()
    print(json.dumps({'version': VERSION, 'port': args.port, 'media': readiness}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


if __name__ == '__main__':
    main()
