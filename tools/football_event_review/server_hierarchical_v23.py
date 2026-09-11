#!/usr/bin/env python3
"""Confidence-gated left/right goal presets for shot and save."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v22  # noqa: F401

_html_v22 = hierarchical.patch_hierarchical_html
_js_v22 = hierarchical.patch_hierarchical_js


def patch_html_v23(text: str) -> str:
    return _html_v22(text).replace(
        "hierarchical-v22-label-capacity-header-20260818",
        "hierarchical-v23-goal-side-presets-20260818",
    )


def patch_js_v23(text: str) -> str:
    text = _js_v22(text)
    old_init = '''    state.attributionByLabel[label] ||= {
      event_team: review.event_team || evidence.suggested_event_team || "unknown",
      goal_side: review.goal_side || evidence.suggested_goal_side || (label === "set_piece" ? "not_applicable" : "unknown"),
    };'''
    new_init = '''    if (!state.attributionByLabel[label]) {
      const reviewed = review.status && review.status !== "unreviewed";
      const savedGoal = review.goal_side;
      const suggestedGoal = ["left", "right"].includes(evidence.suggested_goal_side) && Number(evidence.goal_side_confidence || 0) >= 0.60 ? evidence.suggested_goal_side : null;
      const goalSide = label === "set_piece" ? "not_applicable" : (reviewed ? (savedGoal || "unknown") : (["left", "right"].includes(savedGoal) ? savedGoal : (suggestedGoal || "unknown")));
      state.attributionByLabel[label] = {
        event_team: review.event_team || evidence.suggested_event_team || "unknown",
        goal_side: goalSide,
        goal_side_source: !reviewed && suggestedGoal && goalSide === suggestedGoal ? "auto" : (reviewed ? "reviewed" : "none"),
        goal_side_confidence: Number(evidence.goal_side_confidence || 0),
      };
    }'''
    if old_init not in text:
        raise RuntimeError("v23 attribution initialization anchor missing")
    text = text.replace(old_init, new_init, 1)

    old_goal = '''    const goal = ["shot", "save"].includes(label) ? `<div class="compact-goal-row"><span>目标球门</span>${[["left", "← 左侧"], ["right", "右侧 →"], ["unknown", "不明确"]].map(([side, name]) => `<button data-attr-label="${label}" data-side="${side}" class="${value.goal_side === side ? 'selected' : ''}">${name}</button>`).join("")}</div>` : "";'''
    new_goal = '''    const presetHint = value.goal_side_source === "auto" ? `<small>预设 ${Math.round(value.goal_side_confidence * 100)}%</small>` : "";
    const goal = ["shot", "save"].includes(label) ? `<div class="compact-goal-row"><span>目标球门${presetHint}</span>${[["left", "← 左侧"], ["right", "右侧 →"], ["unknown", "不明确"]].map(([side, name]) => `<button data-attr-label="${label}" data-side="${side}" class="${value.goal_side === side ? 'selected' : ''} ${value.goal_side === side && value.goal_side_source === 'auto' ? 'auto-preset' : ''}">${name}</button>`).join("")}</div>` : "";'''
    if old_goal not in text:
        raise RuntimeError("v23 goal row anchor missing")
    text = text.replace(old_goal, new_goal, 1)

    old_click = 'state.attributionByLabel[button.dataset.attrLabel].goal_side = button.dataset.side; renderTeamAttribution();'
    new_click = 'state.attributionByLabel[button.dataset.attrLabel].goal_side = button.dataset.side; state.attributionByLabel[button.dataset.attrLabel].goal_side_source = "manual"; renderTeamAttribution();'
    if old_click not in text:
        raise RuntimeError("v23 goal click anchor missing")
    return text.replace(old_click, new_click, 1)


hierarchical.patch_hierarchical_html = patch_html_v23
hierarchical.patch_hierarchical_js = patch_js_v23
hierarchical.HIERARCHICAL_CSS += r'''
.compact-goal-row > span small { display:block; margin-top:2px; color:#b7cf4b; font-size:7px; white-space:nowrap; }
.compact-goal-row button.auto-preset { box-shadow:inset 0 0 0 1px #17220d,0 0 0 1px #bed84966; }
'''

if __name__ == "__main__":
    hierarchical.main()
