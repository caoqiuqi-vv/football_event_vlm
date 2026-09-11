#!/usr/bin/env python3
"""Compact, low-pointer-travel launcher for hierarchical football review."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v2  # noqa: F401  # installs cache-busted v2 HTML patch


_patch_html_v2 = hierarchical.patch_hierarchical_html


def patch_html_v3(text: str) -> str:
    text = _patch_html_v2(text)
    text = text.replace("hierarchical-v2-20260812", "hierarchical-v3-compact-20260812")
    text = text.replace("分层标签 v2", "分层标签 v3 · 紧凑操作")

    # Move the complete label correction block directly below the rapid actions.
    block_start = text.index('          <div class="modify-block">')
    block_end = text.index('\n\n          <footer class="event-footer">', block_start)
    modify_block = text[block_start:block_end]
    text = text[:block_start] + text[block_end:]
    rapid_end_anchor = '          </div>\n          <div class="score-grid">'
    if rapid_end_anchor not in text:
        raise RuntimeError("compact UI rapid action anchor not found")
    text = text.replace(
        rapid_end_anchor,
        '          </div>\n' + modify_block + '\n          <div class="score-grid evidence-details">',
        1,
    )
    text = text.replace(
        '<button id="saveModify" class="wide-secondary emphasized">保存修改并下一个</button>',
        '<button id="saveModify" class="wide-secondary emphasized compact-confirm">确认当前标签并下一个</button>',
        1,
    )
    text = text.replace(
        '<span>主事件（可修正）</span>',
        '<div class="label-flow-heading"><span>① 主事件</span><small>选择后点确认；定位球需选择类型</small></div>',
        1,
    )
    return text


hierarchical.patch_hierarchical_html = patch_html_v3


COMPACT_CSS = """
.event-card { padding-top: 10px; }
.event-heading { margin-bottom: 6px; }
.rapid-actions { margin-top: 6px; margin-bottom: 6px; grid-template-columns: .62fr .82fr 1.12fr 1.12fr .62fr; }
.rapid-actions button { height: 45px; }
.modify-block { margin: 0 0 8px; padding: 8px; border-color: #3a4655; background: #0e151d; }
.label-flow-heading { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.label-flow-heading span { color: #f0f4f8; font-size: 12px; font-weight: 800; }
.label-flow-heading small { color: #7f8b99; font-size: 9px; }
.class-buttons { margin: 3px 0 6px; }
.primary-label-buttons button { height: 36px; }
.conditional-label-panel { margin: 6px 0; padding: 7px; }
.conditional-label-panel strong { font-size: 11px; }
.detail-buttons { margin-top: 5px; }
.detail-buttons button { min-height: 34px; }
.compact-confirm { height: 42px; margin-top: 7px; color: #101710; background: var(--accent); border-color: var(--accent); font-weight: 850; font-size: 14px; }
.compact-confirm:hover { background: #e2ff72; }
.time-adjust, #usePlayerTime, #note { display: none; }
.modify-block.advanced-open .time-adjust { display: grid; }
.modify-block.advanced-open #usePlayerTime, .modify-block.advanced-open #note { display: block; }
.score-grid { margin-top: 8px; }
.evidence-details { opacity: .82; }
.playback-actions { margin-bottom: 7px; }
"""


_send_static = hierarchical.HierarchicalReviewHandler.send_static


def send_static_v3(self, request_path: str) -> None:
    # Reuse the hierarchical handler, but append compact CSS to its generated sheet.
    if request_path.split("?", 1)[0].endswith("styles.css"):
        original = hierarchical.HIERARCHICAL_CSS
        hierarchical.HIERARCHICAL_CSS = original + COMPACT_CSS
        try:
            return _send_static(self, request_path)
        finally:
            hierarchical.HIERARCHICAL_CSS = original
    return _send_static(self, request_path)


hierarchical.HierarchicalReviewHandler.send_static = send_static_v3


if __name__ == "__main__":
    hierarchical.main()
