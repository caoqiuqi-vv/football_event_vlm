#!/usr/bin/env python3
"""v36 review UI with direct HTTP/1.1 range streaming and robust absolute seek."""

from __future__ import annotations

import email.utils
import html
import mimetypes
import ssl
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import server as base
import server_hierarchical as hierarchical
import server_multiuser_v32 as v32
import server_multiuser_v36 as v36


# A browser asking for ``bytes=N-`` receives a bounded response. 64 MiB avoids
# repeated TLS requests during ordinary playback while still bounding abandoned
# transfers after a seek. There is intentionally no application-level rate
# limiter: the kernel provides backpressure and each authenticated client reads
# directly from the server NIC.
MAX_RANGE_BYTES = 64 * 1024 * 1024
STREAM_CHUNK_BYTES = 512 * 1024

_html_v36 = hierarchical.patch_hierarchical_html
_js_v36 = hierarchical.patch_hierarchical_js


def patch_html_v37(text: str) -> str:
    text = _html_v36(text).replace(
        "multiuser-v36-team-calibration-20260907",
        "multiuser-v37-direct-range-seek-20260908",
        1,
    )
    anchor = '<button id="soundBtn" class="sound-btn">🔊 开启声音</button>'
    if anchor not in text:
        raise RuntimeError("v37 absolute-seek HTML anchor missing")
    controls = r'''<div id="absoluteSeekBar" class="absolute-seek-bar">
            <input id="absoluteSeekInput" type="text" inputmode="decimal" autocomplete="off" placeholder="00:00:00 或秒数" aria-label="跳转到固定时间">
            <button id="absoluteSeekButton" type="button">跳转</button>
            <small>按需 Range 流</small>
          </div>'''
    return text.replace(anchor, anchor + "\n          " + controls, 1)


def patch_js_v37(text: str) -> str:
    text = _js_v36(text)
    state_anchor = "teamSetupOpen: true, activeTeamColour: 'teamA', teamSampleEnd: null,"
    if state_anchor not in text:
        raise RuntimeError("v37 state anchor missing")
    text = text.replace(
        state_anchor,
        state_anchor + "\n  mediaBaseUrl: '', pendingAbsoluteSeek: null,",
        1,
    )

    helper_anchor = "function playEventContext(event) {"
    if helper_anchor not in text:
        raise RuntimeError("v37 seek helper anchor missing")
    helpers = r'''function parseAbsoluteSeek(raw) {
  const value = String(raw ?? "").trim();
  if (!value) return NaN;
  if (!value.includes(":")) return Number(value);
  const parts = value.split(":").map(Number);
  if (parts.some((part) => !Number.isFinite(part) || part < 0) || parts.length > 3) return NaN;
  return parts.reduce((total, part) => total * 60 + part, 0);
}

function mediaUrlAt(timeSec) {
  const baseUrl = state.mediaBaseUrl || state.video?.media_url || "";
  return `${baseUrl.split("#", 1)[0]}#t=${Math.max(0, Number(timeSec) || 0).toFixed(3)}`;
}

function seekAbsolute(timeSec, autoPlay = false) {
  if (!state.video) return;
  const target = Math.max(0, Math.min(Number(state.video.duration_sec || Infinity), Number(timeSec)));
  if (!Number.isFinite(target)) { toast("请输入秒数或 HH:MM:SS"); return; }
  const player = $("#player");
  const videoId = state.currentVideoId;
  state.pendingAbsoluteSeek = { videoId, target, autoPlay };
  const apply = () => {
    const pending = state.pendingAbsoluteSeek;
    if (!pending || pending.videoId !== state.currentVideoId) return;
    state.pendingAbsoluteSeek = null;
    try { player.currentTime = pending.target; } catch (_) { return; }
    $("#absoluteSeekInput").value = formatTime(pending.target);
    if (pending.autoPlay) player.play().catch(() => toast("请点击画面开始播放"));
  };
  if (!player.getAttribute("src")) {
    // The media fragment tells the browser the first useful timestamp before
    // it opens the file, avoiding an unnecessary fetch from t=0.
    player.src = mediaUrlAt(target);
    player.load();
  }
  if (player.readyState >= 1) apply();
  else player.addEventListener("loadedmetadata", apply, { once: true });
}

'''
    text = text.replace(helper_anchor, helpers + helper_anchor, 1)

    old_context = '''function playEventContext(event) {
  const player = $("#player");
  const bounds = segmentEvidenceBounds(event);
  player.currentTime = bounds.start;
  state.contextEnd = bounds.end;
  const playback = player.play();
  if (playback) playback.catch(() => toast("浏览器阻止了自动播放，请点击画面或按 Space"));
}'''
    new_context = '''function playEventContext(event) {
  const bounds = segmentEvidenceBounds(event);
  state.contextEnd = bounds.end;
  seekAbsolute(bounds.start, true);
}'''
    if old_context not in text:
        raise RuntimeError("v37 play context anchor missing")
    text = text.replace(old_context, new_context, 1)

    old_select = '    else $("#player").currentTime = segmentEvidenceBounds(event).start;'
    if old_select not in text:
        raise RuntimeError("v37 select seek anchor missing")
    text = text.replace(
        old_select,
        '    else seekAbsolute(segmentEvidenceBounds(event).start, false);',
        1,
    )

    old_load = '''  const player = $("#player");
  if (player.getAttribute("src") !== state.video.media_url) player.src = state.video.media_url;
  $("#videoBadge").textContent = videoId;'''
    new_load = '''  const player = $("#player");
  player.pause();
  player.removeAttribute("src");
  player.load();
  state.mediaBaseUrl = state.video.media_url;
  state.pendingAbsoluteSeek = null;
  $("#videoBadge").textContent = videoId;'''
    if old_load not in text:
        raise RuntimeError("v37 load-video source anchor missing")
    text = text.replace(old_load, new_load, 1)

    old_timeline = '    $("#player").currentTime = Math.max(0, Math.min(state.video.duration_sec, time));'
    if old_timeline not in text:
        raise RuntimeError("v37 timeline seek anchor missing")
    text = text.replace(old_timeline, '    seekAbsolute(time, false);', 1)

    bind_anchor = '  $("#undoBtn").onclick = undoDecision;'
    if bind_anchor not in text:
        raise RuntimeError("v37 seek binding anchor missing")
    binding = r'''
  const submitAbsoluteSeek = () => {
    const target = parseAbsoluteSeek($("#absoluteSeekInput").value);
    seekAbsolute(target, false);
  };
  $("#absoluteSeekButton").onclick = submitAbsoluteSeek;
  $("#absoluteSeekInput").onkeydown = (event) => {
    if (event.key === "Enter") { event.preventDefault(); submitAbsoluteSeek(); }
  };'''
    text = text.replace(bind_anchor, bind_anchor + binding, 1)
    return text


hierarchical.patch_hierarchical_html = patch_html_v37
hierarchical.patch_hierarchical_js = patch_js_v37
hierarchical.HIERARCHICAL_CSS += r'''
.absolute-seek-bar{position:absolute;left:50%;top:9px;bottom:auto;z-index:7;display:flex;align-items:center;gap:4px;padding:5px;border:1px solid #4b5c6e;border-radius:8px;background:#0b121bd9;backdrop-filter:blur(5px);transform:translateX(-50%)}
.absolute-seek-bar input{box-sizing:border-box;width:126px;height:29px;padding:0 7px;color:#edf4fb;border:1px solid #506276;border-radius:5px;background:#101b26;font:11px ui-monospace,monospace}
.absolute-seek-bar button{height:29px;padding:0 9px;color:#101710;border:0;border-radius:5px;background:#d7ff38;font-weight:800;cursor:pointer}.absolute-seek-bar small{padding:0 4px;color:#91a2b2;font-size:8px;white-space:nowrap}
@media(max-width:760px){.absolute-seek-bar small{display:none}.absolute-seek-bar input{width:112px}}
'''


class DirectRangeStreamingHandler(v36.TeamCalibrationHandler):
    """Authenticated media delivery with persistent HTTP/1.1 byte ranges."""

    server_version = "FootballEventReview/3.7"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/healthz":
            return self.send_json({"ok": True, "version": "v37-direct-range-streaming"})
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

        stat = path.stat()
        file_size = int(stat.st_size)
        start, end, status = 0, file_size - 1, HTTPStatus.OK
        range_header = self.headers.get("Range", "").strip()
        if range_header:
            try:
                if not range_header.startswith("bytes=") or "," in range_header:
                    raise ValueError("unsupported range")
                raw_start, raw_end = range_header[6:].split("-", 1)
                if not raw_start and not raw_end:
                    raise ValueError("empty range")
                if not raw_start:
                    suffix = max(1, int(raw_end))
                    start = max(0, file_size - min(suffix, MAX_RANGE_BYTES))
                    end = file_size - 1
                else:
                    start = int(raw_start)
                    if start < 0 or start >= file_size:
                        raise ValueError("range start outside file")
                    requested_end = int(raw_end) if raw_end else file_size - 1
                    end = min(requested_end, start + MAX_RANGE_BYTES - 1, file_size - 1)
                    if end < start:
                        raise ValueError("range end before start")
                status = HTTPStatus.PARTIAL_CONTENT
            except (TypeError, ValueError):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", f'"{stat.st_mtime_ns:x}-{file_size:x}"')
        self.send_header("Last-Modified", email.utils.formatdate(stat.st_mtime, usegmt=True))
        self.send_header("X-Review-Media-Variant", variant)
        self.send_header("X-Review-Streaming", "direct-http11-range-v2")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(STREAM_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, TimeoutError, ssl.SSLError):
                    break
                remaining -= len(chunk)


def main() -> None:
    args = v32.parse_args()
    store = v36.TeamCalibrationStore(args.manifest, args.db, args.access_config, args.proxy_root)
    server = v32.ReviewThreadingHTTPServer((args.host, args.port), DirectRangeStreamingHandler)
    server.store = store  # type: ignore[attr-defined]
    if bool(args.tls_cert) != bool(args.tls_key):
        raise ValueError("--tls-cert and --tls-key must be provided together")
    scheme = "http"
    if args.tls_cert and args.tls_key:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(
            server.socket, server_side=True, do_handshake_on_connect=False
        )
        server.is_tls = True  # type: ignore[attr-defined]
        scheme = "https"
    else:
        server.is_tls = False  # type: ignore[attr-defined]
    print(f"Multi-user football review UI v37: {scheme}://{args.host}:{args.port}", flush=True)
    print("Media transport: direct HTTP/1.1 keep-alive + bounded byte ranges", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
