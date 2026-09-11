#!/usr/bin/env python3
"""Add a bottom-of-details confirmation action for low-travel review."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v23  # noqa: F401

_html_v23 = hierarchical.patch_hierarchical_html
_js_v23 = hierarchical.patch_hierarchical_js


def patch_html_v24(text: str) -> str:
    text = _html_v23(text).replace(
        "hierarchical-v23-goal-side-presets-20260818",
        "hierarchical-v24-bottom-confirm-20260818",
    )
    anchor = '            <button id="saveModify" class="wide-secondary emphasized compact-confirm">确认当前标签并下一个</button>'
    replacement = anchor + '\n            <button id="bottomConfirmBtn" class="bottom-confirm" type="button"><span>✓ 确认当前结果并下一个</span><kbd>Enter</kbd></button>'
    if anchor not in text:
        raise RuntimeError("v24 bottom confirmation anchor missing")
    return text.replace(anchor, replacement, 1)


def patch_js_v24(text: str) -> str:
    text = _js_v23(text)
    bind_anchor = '  $("#acceptBtn").onclick = saveMultiLabelSegment;'
    if bind_anchor not in text:
        raise RuntimeError("v24 accept binding anchor missing")
    text = text.replace(bind_anchor, bind_anchor + '\n  $("#bottomConfirmBtn").onclick = saveMultiLabelSegment;', 1)
    hint_anchor = '  $("#saveSegmentLabelsBtn").textContent = selectedNames.length ? "确认所选标签并播放下一片段" : "删除本片段全部预测并播放下一片段";'
    if hint_anchor not in text:
        raise RuntimeError("v24 selected-label hint anchor missing")
    update = hint_anchor + '\n  $("#bottomConfirmBtn").querySelector("span").textContent = selectedNames.length ? `✓ 确认 ${selectedNames.join(" + ")} 并下一个` : "× 删除本片段并下一个";\n  $("#bottomConfirmBtn").classList.toggle("delete-all", selectedNames.length === 0);'
    return text.replace(hint_anchor, update, 1)


hierarchical.patch_hierarchical_html = patch_html_v24
hierarchical.patch_hierarchical_js = patch_js_v24
hierarchical.HIERARCHICAL_CSS += r'''
.bottom-confirm { position:sticky; bottom:0; z-index:8; width:100%; min-height:45px; margin-top:8px; color:#102118; border:1px solid #72e0ad; border-radius:8px; background:#5dd39e; box-shadow:0 -6px 18px #0d1218cc; cursor:pointer; font-size:13px; font-weight:850; }
.bottom-confirm kbd { margin-left:10px; font:9px ui-monospace,monospace; opacity:.65; }

.bottom-confirm.delete-all { color:#fff; border-color:#b9485a; background:#913447; }

/* v24 theme polish: quieter palette for long review sessions. */
:root {
  --bg: #0d1117;
  --panel: #121922;
  --panel-2: #17212b;
  --line: #263442;
  --text: #eef3f8;
  --muted: #8b9aaa;
  --accent: #7dd3c7;
  --accent-2: #4fb6ab;
  --shot: #ef6f6c;
  --save: #67a9e8;
  --set-piece: #c99af0;
  --green: #73d3a3;
  --danger: #d95f73;
}
body { background:#0d1117; color:var(--text); }
.topbar,
.review-panel,
.timeline-panel,
.event-card,
.rapid-actions { background:var(--panel); }
.topbar { border-bottom-color:#243140; }
.brand-mark {
  color:#071817;
  background:linear-gradient(135deg,#91e0d5,#64bdb5);
  box-shadow:0 0 0 1px #b7fff11f,0 8px 22px #0008;
}
.progress { background:#202a35; }
.progress i { background:linear-gradient(90deg,#69c8bf,#a1ddd6); }
input, select, textarea { color:#e7edf4; background:#0f151d; border-color:#314151; }
input:focus, select:focus, textarea:focus { border-color:#69c8bf; box-shadow:0 0 0 2px #7dd3c724; }
.ghost-btn,
.rapid-nav,
.playback-actions button,
.time-adjust button,
.wide-secondary,
.compact-team-row button,
.compact-goal-row button,
.quick-time-actions button,
.detail-buttons button,
.class-buttons button {
  color:#d8e2eb;
  background:#18222d;
  border-color:#344555;
}
.ghost-btn:hover,
.rapid-nav:hover,
.playback-actions button:hover,
.wide-secondary:hover {
  color:#f2fbfb;
  border-color:#6fbfb8;
  background:#1d2b36;
}
.event-class,
.prediction-result .event-label { color:#111820; box-shadow:none; }
.event-class.shot,
.event-label.label-shot,
.label-shot { background:var(--shot); }
.event-class.save,
.event-label.label-save,
.label-save { background:var(--save); }
.event-class.set_piece,
.event-label.label-set_piece,
.label-set_piece { background:var(--set-piece); }
.evaluation-badge.fp { color:#f6b5bf; border-color:#68404a; background:#301923; }
.evaluation-badge.matched { color:#a1e7c5; border-color:#32644e; background:#152b24; }
.evaluation-badge.unlabeled { color:#efd99a; border-color:#6a5830; background:#2c2518; }
.evaluation-badge.whistle_rescue { color:#a8e4df; border-color:#39706b; background:#132b2b; }
.rapid-actions { border-color:#243140; box-shadow:0 6px 18px #07101866; }
.rapid-replay { color:#082223; border-color:#72c9d1; background:#85d5dc; }
.rapid-replay:hover { background:#9fe1e6; }
.rapid-decision.accept,
.bottom-confirm { color:#082015; border-color:#80d7aa; background:#7bd6a8; }
.rapid-decision.accept:hover,
.bottom-confirm:hover { background:#8be0b5; }
.rapid-decision.delete,
.bottom-confirm.delete-all { color:#fff2f4; border-color:#b95063; background:#9b3a4d; }
.rapid-decision.delete:hover,
.bottom-confirm.delete-all:hover { background:#b1485b; }
.primary-small,
.class-buttons button.selected,
.detail-buttons button.selected,
.compact-team-row button.selected,
.compact-goal-row button.selected,
.playback-rate-bar button.selected,
#useCurrentFrame { color:#082015; border-color:#7dd3c7; background:#7dd3c7; }
.modify-block,
.segment-label-editor,
.conditional-label-panel,
.team-attribution-panel,
.attribution-card,
.quick-time-editor,
.score-grid > div,
.segment-batch-actions { background:#101821; border-color:#293948; }
.segment-label-card { color:#111820; box-shadow:inset 0 -1px 0 #0002; }
.segment-label-card.label-shot { background:#ef6f6c; }
.segment-label-card.label-save { background:#67a9e8; }
.segment-label-card.label-set_piece { background:#c99af0; }
.whistle-evidence { border-color:#4c8983; background:#13302f; }
.whistle-evidence strong,
.whistle-evidence b { color:#9ce4dc; }
.header-whistle-badge.has-whistle { color:#9ce4dc; border-color:#4c8983; background:#143b38; box-shadow:0 0 10px #7dd3c733; }
.header-whistle-badge.no-whistle { color:#657483; border-color:#354555; background:#121b25; }
.header-whistle-badge.no-whistle .whistle-slash { stroke:#8f5d67; }
.queue-item:hover,
.queue-item.current { background:#192431; }
.queue-item.current { box-shadow:inset 2px 0 #7dd3c7; }
.queue-item em { color:#9ce4dc; border-color:#4c8983; background:#143b38; }
#timeline { background:#0b1118; border-color:#263442; }
.video-overlay { background:#071018c9; border-color:#ffffff17; }
.buffer-badge,
.switch-badge { color:#9ce4dc; border-color:#4c8983; background:#102629e8; }
.switch-badge { color:#082015; background:#9ce4dce8; }
.sound-btn { color:#082015; border-color:#7dd3c7; background:#7dd3c7; }
'''

if __name__ == "__main__":
    hierarchical.main()
