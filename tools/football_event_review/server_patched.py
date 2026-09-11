#!/usr/bin/env python3
"""Review UI server with lightweight runtime patches for navigation smoothness."""

from __future__ import annotations

import mimetypes
import threading
import webbrowser
from http.server import ThreadingHTTPServer

import server as base


def replace_required(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"UI patch anchor not found: {old[:80]!r}")
    return text.replace(old, new, 1)


def patch_javascript(text: str) -> str:
    text = replace_required(
        text,
        'const EVALUATION_NAMES = { fp: "模型 FP · 待人工核验", matched: "已匹配 GT", unlabeled: "该类别未标注", whistle_rescue: "哨声补漏 · 待人工确认" };',
        'const EVALUATION_NAMES = { fp: "模型 FP · 待人工核验", fn: "模型 FN · GT 漏检", matched: "已匹配 GT", unlabeled: "该类别未标注", whistle_rescue: "哨声补漏 · 待人工确认" };',
    )
    text = replace_required(
        text,
        "  hitboxes: [], contextEnd: null,\n",
        "  hitboxes: [], contextEnd: null, timelineDrawPending: false,\n",
    )
    text = replace_required(
        text,
        '  requestAnimationFrame(() => list.querySelector(".current")?.scrollIntoView({ block: "nearest" }));',
        '''  requestAnimationFrame(() => {
    const current = list.querySelector(".current");
    if (!current) return;
    const top = current.offsetTop;
    const bottom = top + current.offsetHeight;
    if (top < list.scrollTop) list.scrollTop = top;
    else if (bottom > list.scrollTop + list.clientHeight) list.scrollTop = bottom - list.clientHeight;
  });''',
    )
    text = replace_required(
        text,
        "function selectEvent(eventId, seek = true, autoPlay = false) {\n  state.currentEventId = eventId;",
        "function selectEvent(eventId, seek = true, autoPlay = false) {\n  const changed = state.currentEventId !== eventId;\n  state.currentEventId = eventId;",
    )
    text = replace_required(
        text,
        "  renderQueue(); renderEvent(); drawTimeline();\n}\n\nfunction nextEvent",
        '''  renderQueue(); renderEvent(); drawTimeline();
  if (changed) requestAnimationFrame(() => { $("#eventCard").scrollTop = 0; });
}

function scheduleTimelineDraw() {
  if (state.timelineDrawPending) return;
  state.timelineDrawPending = true;
  requestAnimationFrame(() => {
    state.timelineDrawPending = false;
    drawTimeline();
  });
}

function nextEvent''',
    )
    text = replace_required(
        text,
        '''  canvas.width = Math.round(canvas.clientWidth * ratio);
  canvas.height = Math.round(canvas.clientHeight * ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);''',
        '''  const pixelWidth = Math.round(canvas.clientWidth * ratio);
  const pixelHeight = Math.round(canvas.clientHeight * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);''',
    )
    text = replace_required(
        text,
        'if (state.contextEnd && player.currentTime >= state.contextEnd) { player.pause(); state.contextEnd = null; } drawTimeline(); };',
        'if (state.contextEnd && player.currentTime >= state.contextEnd) { player.pause(); state.contextEnd = null; } scheduleTimelineDraw(); };',
    )
    return text


def patch_html(text: str) -> str:
    return replace_required(
        text,
        '            <option value="fp">仅看模型 FP</option>',
        '            <option value="fp">仅看模型 FP</option>\n            <option value="fn">仅看模型 FN</option>',
    )


CSS_PATCH = """
.review-panel, .event-card, .queue-list { overscroll-behavior: contain; }
.event-card, .queue-list { scrollbar-gutter: stable; }
.event-card { min-height: 0; }
.evaluation-badge.fn { color: #f0d377; border-color: #756127; background: #3c3218; }
"""


class PatchedReviewHandler(base.ReviewHandler):
    server_version = "FootballEventReview/1.1"

    def send_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (base.STATIC_ROOT / relative).resolve()
        if base.STATIC_ROOT not in target.parents and target != base.STATIC_ROOT:
            return self.send_json({"error": "Invalid path"}, 400)
        if not target.exists() or not target.is_file():
            return self.send_json({"error": "Not found"}, 404)
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.name in {"app.js", "index.html", "styles.css"}:
            text = target.read_text(encoding="utf-8")
            if target.name == "app.js":
                text = patch_javascript(text)
            elif target.name == "index.html":
                text = patch_html(text)
            else:
                text += CSS_PATCH
            return self.send_bytes(text.encode("utf-8"), f"{content_type}; charset=utf-8")
        self.send_bytes(target.read_bytes(), content_type)


def main() -> None:
    args = base.parse_args()
    store = base.ReviewStore(args.manifest, args.db)
    server = ThreadingHTTPServer((args.host, args.port), PatchedReviewHandler)
    server.store = store  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}"
    print(f"Football event review UI (patched): {url}", flush=True)
    print(f"Review database: {store.db_path}", flush=True)
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
