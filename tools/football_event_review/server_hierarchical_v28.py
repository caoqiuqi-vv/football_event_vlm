#!/usr/bin/env python3
"""Remote-friendly review UI without the second hidden video preloader."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v27  # noqa: F401


_html_v27 = hierarchical.patch_hierarchical_html
_js_v27 = hierarchical.patch_hierarchical_js


def patch_html_v28(text: str) -> str:
    text = _html_v27(text).replace(
        "hierarchical-v27-gt-subtype-defaults-20260901",
        "hierarchical-v28-ssh-friendly-streaming-20260901",
        1,
    )
    return text.replace(
        '<video id="clipPreloader" class="clip-preloader" preload="metadata" muted playsinline aria-hidden="true"></video>',
        "",
        1,
    )


def patch_js_v28(text: str) -> str:
    text = _js_v27(text)
    start = text.find("function prefetchUpcomingSegment() {")
    end = text.find("function selectEvent(", start)
    if start < 0 or end < 0:
        raise RuntimeError("v28 prefetch function anchors missing")
    text = text[:start] + "function prefetchUpcomingSegment() {}\n\n" + text[end:]
    old = '''  const player = $("#player");
  const preloader = $("#clipPreloader");
  if (player.getAttribute("src") !== state.video.media_url) player.src = state.video.media_url;
  if (preloader.getAttribute("src") !== state.video.media_url) preloader.src = state.video.media_url;'''
    new = '''  const player = $("#player");
  if (player.getAttribute("src") !== state.video.media_url) player.src = state.video.media_url;'''
    if old not in text:
        raise RuntimeError("v28 dual-player source anchor missing")
    return text.replace(old, new, 1)


hierarchical.patch_hierarchical_html = patch_html_v28
hierarchical.patch_hierarchical_js = patch_js_v28


if __name__ == "__main__":
    hierarchical.main()
