#!/usr/bin/env python3
"""Faster clip navigation, scrollable timeline, and compact whistle icon."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v16  # noqa: F401


_html_v16 = hierarchical.patch_hierarchical_html
_js_v16 = hierarchical.patch_hierarchical_js


def patch_html_v17(text: str) -> str:
    text = _html_v16(text).replace(
        "hierarchical-v16-header-whistle-20260818",
        "hierarchical-v17-prefetch-scroll-icon-20260818",
    )
    old_badge = (
        '<span id="headerWhistleBadge" class="header-whistle-badge no-whistle" '
        'title="当前片段附近的哨声检测结果">未检测到哨声</span>'
    )
    new_badge = '''<span id="headerWhistleBadge" class="header-whistle-badge no-whistle" role="img" aria-label="未检测到哨声" title="当前片段附近未检测到哨声"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/><path class="whistle-slash" d="M4 4l16 16"/></svg></span>'''
    if old_badge not in text:
        raise RuntimeError("v17 whistle badge anchor missing")
    text = text.replace(old_badge, new_badge, 1)

    old_player = '<video id="player" controls preload="metadata"></video>'
    new_player = (
        '<video id="player" controls preload="metadata" playsinline></video>'
        '<video id="clipPreloader" class="clip-preloader" preload="metadata" muted playsinline aria-hidden="true"></video>'
    )
    if old_player not in text:
        raise RuntimeError("v17 player anchor missing")
    text = text.replace(old_player, new_player, 1)

    old_timeline = '<canvas id="timeline" height="210" aria-label="事件时间轴"></canvas>'
    new_timeline = '<div id="timelineScroll" class="timeline-scroll"><canvas id="timeline" height="330" aria-label="事件时间轴"></canvas></div>'
    if old_timeline not in text:
        raise RuntimeError("v17 timeline anchor missing")
    text = text.replace(old_timeline, new_timeline, 1)
    return text.replace(
        '滚轮缩放 · 拖动平移 · 点击跳转 · 彩色菱形为模型候选',
        '上下滚动查看三类事件 · Ctrl/⌘ + 滚轮缩放时间 · 拖动平移 · 点击跳转',
        1,
    )


def patch_js_v17(text: str) -> str:
    text = _js_v16(text)
    old_badge = '''  headerWhistleBadge.className = `header-whistle-badge ${whistleScore > 0 ? "has-whistle" : "no-whistle"}`;
  headerWhistleBadge.textContent = whistleScore > 0 ? `🔔 有哨声 ${whistleScore.toFixed(2)}` : "未检测到哨声";
  headerWhistleBadge.title = whistleScore > 0
    ? `当前片段附近检测到哨声，置信度 ${whistleScore.toFixed(3)}`
    : "当前片段附近未检测到哨声";'''
    new_badge = '''  headerWhistleBadge.className = `header-whistle-badge ${whistleScore > 0 ? "has-whistle" : "no-whistle"}`;
  headerWhistleBadge.setAttribute("aria-label", whistleScore > 0 ? `检测到哨声，置信度 ${whistleScore.toFixed(3)}` : "未检测到哨声");
  headerWhistleBadge.title = whistleScore > 0
    ? `检测到哨声 · ${whistleScore.toFixed(3)}`
    : "未检测到哨声";'''
    if old_badge not in text:
        raise RuntimeError("v17 whistle JS anchor missing")
    text = text.replace(old_badge, new_badge, 1)

    select_anchor = "function selectEvent(eventId, seek = true, autoPlay = false) {"
    prefetch = r'''let clipPrefetchTimer = null;

function prefetchUpcomingSegment() {
  window.clearTimeout(clipPrefetchTimer);
  clipPrefetchTimer = window.setTimeout(() => {
    if (!state.video) return;
    const groups = segmentGroups();
    const currentKey = segmentKey(currentEvent());
    const index = groups.findIndex((group) => segmentKey(group[0]) === currentKey);
    const nextGroup = groups[index + 1];
    if (!nextGroup) return;
    const target = Math.max(0, Number(segmentRepresentative(nextGroup).time_sec) - 3);
    const preloader = $("#clipPreloader");
    const warm = () => {
      try {
        if (Math.abs(Number(preloader.currentTime || 0) - target) > 0.25) preloader.currentTime = target;
      } catch (_) {}
    };
    if (preloader.readyState >= 1) warm();
    else preloader.addEventListener("loadedmetadata", warm, { once: true });
  }, 180);
}

'''
    if select_anchor not in text:
        raise RuntimeError("v17 selectEvent anchor missing")
    text = text.replace(select_anchor, prefetch + select_anchor, 1)
    old_render = "  renderQueue(); renderEvent(); drawTimeline();\n  if (changed) requestAnimationFrame(() => { $(\"#eventCard\").scrollTop = 0; });"
    new_render = "  renderQueue(); renderEvent(); drawTimeline();\n  prefetchUpcomingSegment();\n  if (changed) requestAnimationFrame(() => { $(\"#eventCard\").scrollTop = 0; });"
    if old_render not in text:
        raise RuntimeError("v17 selectEvent render anchor missing")
    text = text.replace(old_render, new_render, 1)

    old_src = '  $("#player").src = state.video.media_url;'
    new_src = '''  const player = $("#player");
  const preloader = $("#clipPreloader");
  if (player.getAttribute("src") !== state.video.media_url) player.src = state.video.media_url;
  if (preloader.getAttribute("src") !== state.video.media_url) preloader.src = state.video.media_url;'''
    if old_src not in text:
        raise RuntimeError("v17 video source anchor missing")
    text = text.replace(old_src, new_src, 1)

    old_wheel = '''  canvas.addEventListener("wheel", (event) => {
    if (!state.video) return;
    event.preventDefault();'''
    new_wheel = '''  canvas.addEventListener("wheel", (event) => {
    if (!state.video || !(event.ctrlKey || event.metaKey)) return;
    event.preventDefault();'''
    if old_wheel not in text:
        raise RuntimeError("v17 timeline wheel anchor missing")
    text = text.replace(old_wheel, new_wheel, 1)

    old_rows = "    const top = 28 + rowIndex * 57, bottom = top + 46;"
    new_rows = "    const top = 28 + rowIndex * 92, bottom = top + 76;"
    if old_rows not in text:
        raise RuntimeError("v17 timeline row anchor missing")
    return text.replace(old_rows, new_rows, 1)


hierarchical.patch_hierarchical_html = patch_html_v17
hierarchical.patch_hierarchical_js = patch_js_v17
hierarchical.HIERARCHICAL_CSS += r'''
/* v17: silent look-ahead video element warms the browser's byte-range cache. */
.clip-preloader { position: absolute; width: 1px !important; height: 1px !important; opacity: 0; pointer-events: none; }

/* Compact signal lamp: recognizable at a glance, with details available on hover. */
.header-whistle-badge {
  width: 27px;
  min-width: 27px;
  height: 27px;
  min-height: 27px;
  margin: 0 7px 0 1px;
  padding: 5px;
  border-radius: 7px;
}
.header-whistle-badge svg { width: 100%; height: 100%; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.header-whistle-badge.has-whistle { color: #dfff45; border-color: #bad82f; background: #8aa02042; box-shadow: 0 0 9px #d7ff3840; }
.header-whistle-badge.has-whistle .whistle-slash { display: none; }
.header-whistle-badge.no-whistle { color: #596675; border-color: #35404c; background: #151d26; opacity: .72; }
.header-whistle-badge.no-whistle .whistle-slash { display: block; stroke: #8c5660; }

/* The plot owns a taller drawing surface; its panel scrolls instead of clipping a class row. */
.timeline-panel { display: flex; flex-direction: column; overflow: hidden; }
.timeline-scroll { flex: 1 1 auto; min-height: 0; overflow-x: hidden; overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable; border-radius: 7px; }
#timeline { width: 100%; height: 330px; min-height: 330px; }
.timeline-hint { flex: 0 0 auto; }
'''


if __name__ == "__main__":
    hierarchical.main()
