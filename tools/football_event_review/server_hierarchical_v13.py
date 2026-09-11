#!/usr/bin/env python3
"""Review UI with video-local pause and persistent playback-rate controls."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v12  # noqa: F401


_html_v12 = hierarchical.patch_hierarchical_html
_js_v12 = hierarchical.patch_hierarchical_js


def replace_required(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"v13 patch anchor missing: {old[:100]!r}")
    return text.replace(old, new, 1)


def patch_html_v13(text: str) -> str:
    text = _html_v12(text).replace(
        "hierarchical-v12-conditional-details-20260818",
        "hierarchical-v13-video-speed-controls-20260818",
    )
    pause_action = '            <button id="pauseEventBtn" class="rapid-pause"><span>⏸ 暂停</span><kbd>P</kbd></button>\n'
    text = replace_required(text, pause_action, "")
    sound = '          <button id="soundBtn" class="sound-btn">🔊 开启声音</button>'
    controls = '''          <div id="playbackRateBar" class="playback-rate-bar" aria-label="播放速度控制">
            <button id="pauseEventBtn" class="video-pause" type="button" title="暂停在当前画面（P）">⏸ 暂停</button>
            <span>倍速</span>
            <button type="button" data-playback-rate="0.5">0.5×</button>
            <button type="button" data-playback-rate="0.75">0.75×</button>
            <button type="button" data-playback-rate="1" class="selected">1×</button>
            <button type="button" data-playback-rate="1.25">1.25×</button>
            <button type="button" data-playback-rate="1.5">1.5×</button>
            <button type="button" data-playback-rate="2">2×</button>
          </div>'''
    return replace_required(text, sound, sound + "\n" + controls)


def patch_js_v13(text: str) -> str:
    text = _js_v12(text)
    anchor = '''  const player = $("#player");
  $("#soundBtn").onclick = () => {'''
    replacement = '''  const player = $("#player");
  const applyPlaybackRate = (rawRate) => {
    const rate = Number(rawRate) || 1;
    player.playbackRate = rate;
    localStorage.setItem("football-review-playback-rate", String(rate));
    document.querySelectorAll("[data-playback-rate]").forEach((button) =>
      button.classList.toggle("selected", Number(button.dataset.playbackRate) === rate)
    );
  };
  document.querySelectorAll("[data-playback-rate]").forEach((button) => {
    button.onclick = () => { applyPlaybackRate(button.dataset.playbackRate); toast(`播放速度 ${button.dataset.playbackRate}×`); };
  });
  player.addEventListener("loadedmetadata", () => applyPlaybackRate(localStorage.getItem("football-review-playback-rate") || 1));
  applyPlaybackRate(localStorage.getItem("football-review-playback-rate") || 1);
  $("#soundBtn").onclick = () => {'''
    return replace_required(text, anchor, replacement)


hierarchical.patch_hierarchical_html = patch_html_v13
hierarchical.patch_hierarchical_js = patch_js_v13
hierarchical.HIERARCHICAL_CSS += r'''
/* v13: playback controls belong to the video, not the event-decision rail. */
.playback-rate-bar {
  position: absolute;
  left: 50%;
  bottom: 48px;
  z-index: 8;
  display: flex;
  align-items: center;
  gap: 3px;
  padding: 5px;
  color: #c9d3dc;
  background: #0c1219df;
  border: 1px solid #3a4653;
  border-radius: 8px;
  box-shadow: 0 5px 18px #0008;
  transform: translateX(-50%);
  opacity: .86;
  transition: opacity .15s ease;
}
.playback-rate-bar:hover { opacity: 1; }
.playback-rate-bar > span { padding: 0 4px; color: #82909e; font-size: 10px; white-space: nowrap; }
.playback-rate-bar button { min-width: 42px; height: 30px; padding: 3px 7px; color: #cbd5df; background: #1a2430; border: 1px solid #3b4857; border-radius: 5px; cursor: pointer; font-size: 11px; font-weight: 750; white-space: nowrap; }
.playback-rate-bar button:hover { color: #fff; border-color: #718297; }
.playback-rate-bar button.selected { color: #111; background: var(--accent); border-color: var(--accent); }
.playback-rate-bar .video-pause { min-width: 66px; color: #f1d16e; background: #2b2516; border-color: #6b5925; }

.rapid-actions {
  grid-template-columns: 86px 86px minmax(166px, 1.2fr) minmax(158px, 1.1fr) 86px;
}
@media (max-width: 700px) {
  .rapid-actions { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .playback-rate-bar { bottom: 42px; max-width: calc(100% - 16px); overflow-x: auto; }
}
'''


if __name__ == "__main__":
    hierarchical.main()
