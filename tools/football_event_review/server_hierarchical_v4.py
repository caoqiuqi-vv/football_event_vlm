#!/usr/bin/env python3
"""Compact hierarchical UI with collapsible advanced fields."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v3  # noqa: F401


_html_v3 = hierarchical.patch_hierarchical_html
_js_v3 = hierarchical.patch_hierarchical_js


def patch_html_v4(text: str) -> str:
    text = _html_v3(text).replace("hierarchical-v3-compact-20260812", "hierarchical-v4-compact-20260812")
    return text.replace(
        '            <div class="time-adjust">',
        '            <button id="toggleAdvanced" class="advanced-toggle" type="button">调整事件时间 / 添加备注 ▾</button>\n            <div class="time-adjust">',
        1,
    )


def patch_js_v4(text: str) -> str:
    text = _js_v3(text)
    anchor = '  $("#usePlayerTime").onclick = () => { $("#correctedTime").value = $("#player").currentTime.toFixed(3); };'
    addition = '''
  $("#toggleAdvanced").onclick = () => {
    const block = document.querySelector(".modify-block");
    const open = block.classList.toggle("advanced-open");
    $("#toggleAdvanced").textContent = open ? "收起时间与备注 ▴" : "调整事件时间 / 添加备注 ▾";
  };'''
    if anchor not in text:
        raise RuntimeError("advanced options anchor missing")
    return text.replace(anchor, anchor + addition, 1)


hierarchical.patch_hierarchical_html = patch_html_v4
hierarchical.patch_hierarchical_js = patch_js_v4
hierarchical.HIERARCHICAL_CSS += """
.advanced-toggle { width: 100%; margin-top: 3px; padding: 5px; color: #8e9aa9; background: transparent; border: 0; cursor: pointer; font-size: 10px; }
.advanced-toggle:hover { color: #d7ff38; }
"""


if __name__ == "__main__":
    hierarchical.main()
