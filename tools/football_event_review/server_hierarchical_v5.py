#!/usr/bin/env python3
"""Unified label styling and viewport-safe scrolling for the review UI."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v4  # noqa: F401


_html_v4 = hierarchical.patch_hierarchical_html
_js_v4 = hierarchical.patch_hierarchical_js


def patch_html_v5(text: str) -> str:
    text = _html_v4(text).replace("hierarchical-v4-compact-20260812", "hierarchical-v5-scroll-20260813")
    text = text.replace("分层标签 v3 · 紧凑操作", "高召回复核")
    text = text.replace(
        '<div><span id="eventClass" class="event-class">SHOT</span><strong id="eventTime">00:00.000</strong></div>',
        '<div><small class="prediction-caption">模型预测</small><span id="eventClass" class="event-class">射门</span><strong id="eventTime">00:00.000</strong></div>',
        1,
    )
    text = text.replace(
        '<div class="label-flow-heading"><span>① 主事件</span><small>选择后点确认；定位球需选择类型</small></div>',
        '<div class="label-flow-heading"><span>人工确认主事件</span><small id="correctionHint">默认与模型预测一致</small></div>',
        1,
    )
    return text


def patch_js_v5(text: str) -> str:
    text = _js_v4(text)
    text = text.replace(
        '  classElement.textContent = sourceLabel.toUpperCase().replace("_", " ");',
        '  classElement.textContent = LABEL_NAMES[sourceLabel] || sourceLabel;',
        1,
    )
    text = text.replace(
        '''  $("#classButtons").innerHTML = primaryLabels.map((label, index) =>
    `<button data-label="${label}" class="${state.selectedLabel === label ? "selected" : ""}">${index + 1} ${LABEL_NAMES[label]}</button>`
  ).join("");''',
        '''  $("#classButtons").innerHTML = primaryLabels.map((label, index) =>
    `<button data-label="${label}" class="label-${label} ${state.selectedLabel === label ? "selected" : ""}">${index + 1} ${LABEL_NAMES[label]}</button>`
  ).join("");''',
        1,
    )
    anchor = '  const shotPanel = $("#shotDetailPanel");'
    replacement = '''  const current = currentEvent();
  const sourcePrimary = current ? (["free_kick", "penalty", "corner", "kickoff", "other_set_piece"].includes(current.label) ? "set_piece" : (["shot_on_target", "goal", "own_goal"].includes(current.label) ? "shot" : current.label)) : null;
  $("#correctionHint").textContent = state.selectedLabel === sourcePrimary ? "与模型预测一致" : `人工修正为：${LABEL_NAMES[state.selectedLabel] || state.selectedLabel}`;
  $("#correctionHint").classList.toggle("changed", state.selectedLabel !== sourcePrimary);
  const shotPanel = $("#shotDetailPanel");'''
    if anchor not in text:
        raise RuntimeError("label hint anchor missing")
    return text.replace(anchor, replacement, 1)


hierarchical.patch_hierarchical_html = patch_html_v5
hierarchical.patch_hierarchical_js = patch_js_v5
hierarchical.HIERARCHICAL_CSS += """
.prediction-caption { color: #7f8b99; font-size: 9px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.event-heading > div:first-child { justify-content: flex-start; }
.primary-label-buttons button.label-shot.selected { color: #111; background: var(--shot); border-color: var(--shot); }
.primary-label-buttons button.label-save.selected { color: #111; background: var(--save); border-color: var(--save); }
.primary-label-buttons button.label-set_piece.selected { color: #111; background: var(--set-piece); border-color: var(--set-piece); }
#correctionHint.changed { color: #f1d470; font-weight: 700; }

/* Do not trap short screens: the page itself scrolls when the fixed workspace is taller than the viewport. */
html { min-height: 100%; overflow-y: auto; }
body { min-height: 100%; overflow-x: hidden; overflow-y: auto; }
.workspace { min-height: 680px; }
.left-column, .review-panel { min-height: 0; }
.event-card { min-height: 180px; overflow-y: auto; overscroll-behavior-y: contain; }
.queue-list { min-height: 90px; }

@media (max-height: 850px) and (min-width: 1001px) {
  .workspace { height: 680px; }
  .left-column { grid-template-rows: 390px 290px; }
  .review-panel { height: 680px; }
}
@media (max-height: 700px) and (min-width: 1001px) {
  .workspace { height: 620px; min-height: 620px; }
  .left-column { grid-template-rows: 340px 280px; }
  .review-panel { height: 620px; }
}
"""


if __name__ == "__main__":
    hierarchical.main()
