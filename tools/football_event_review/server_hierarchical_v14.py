#!/usr/bin/env python3
"""Review UI with a slim vertical playback rail on the video right edge."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v13  # noqa: F401


_html_v13 = hierarchical.patch_hierarchical_html


def patch_html_v14(text: str) -> str:
    text = _html_v13(text).replace(
        "hierarchical-v13-video-speed-controls-20260818",
        "hierarchical-v14-vertical-speed-rail-20260818",
    )
    old = '<button id="pauseEventBtn" class="video-pause" type="button" title="暂停在当前画面（P）">⏸ 暂停</button>'
    new = '<button id="pauseEventBtn" class="video-pause" type="button" title="暂停在当前画面（P）"><b aria-hidden="true">⏸</b><small>暂停</small></button>'
    if old not in text:
        raise RuntimeError("v14 pause button anchor missing")
    return text.replace(old, new, 1)


hierarchical.patch_hierarchical_html = patch_html_v14
hierarchical.HIERARCHICAL_CSS += r'''
/* v14: a narrow, low-obstruction playback rail on the right edge. */
.playback-rate-bar {
  left: auto;
  right: 9px;
  top: 50%;
  bottom: auto;
  width: 48px;
  max-width: 48px;
  flex-direction: column;
  gap: 4px;
  padding: 6px 5px;
  border-color: #53617099;
  border-radius: 12px;
  background: #0a1119c7;
  box-shadow: 0 5px 20px #0009;
  transform: translateY(-50%);
  opacity: .52;
  overflow: visible;
}
.playback-rate-bar:hover, .playback-rate-bar:focus-within { opacity: .96; }
.playback-rate-bar > span {
  width: 100%;
  padding: 3px 0 1px;
  color: #9aa8b6;
  border-top: 1px solid #3a4653;
  text-align: center;
  font-size: 9px;
}
.playback-rate-bar button {
  width: 38px;
  min-width: 38px;
  height: 29px;
  padding: 2px;
  border-radius: 6px;
  font-size: 10px;
}
.playback-rate-bar .video-pause {
  width: 38px;
  min-width: 38px;
  height: 42px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1px;
}
.playback-rate-bar .video-pause b { font-size: 14px; line-height: 1; }
.playback-rate-bar .video-pause small { font-size: 9px; font-weight: 750; }
@media (max-width: 700px) {
  .playback-rate-bar {
    left: auto;
    right: 5px;
    top: 50%;
    bottom: auto;
    max-width: 48px;
    overflow: visible;
    transform: translateY(-50%);
  }
}
'''


if __name__ == "__main__":
    hierarchical.main()
