#!/usr/bin/env python3
"""Compact per-label attribution and quick event-time correction."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v15 as v15
import server_hierarchical_v20  # noqa: F401

_html_v20 = hierarchical.patch_hierarchical_html
_js_v20 = hierarchical.patch_hierarchical_js


def patch_html_v21(text: str) -> str:
    text = _html_v20(text).replace(
        "hierarchical-v20-instant-actions-20260818",
        "hierarchical-v21-compact-complete-editing-20260818",
    )
    old_time = '<strong id="eventTime">00:00.000</strong>'
    new_time = '<button id="timeEditButton" class="time-edit-button" type="button" title="校正事件时间"><strong id="eventTime">00:00.000</strong><i>✎</i><small id="timeDeltaBadge"></small></button>'
    if old_time not in text:
        raise RuntimeError("v21 time anchor missing")
    text = text.replace(old_time, new_time, 1)

    start = text.index('            <section id="teamAttributionPanel"')
    end = text.index('            </section>', start) + len('            </section>')
    panel = '''            <section class="attribution-by-label-panel">
              <div class="attribution-heading"><strong>③ 阵营与方向</strong><small>按事件分别确认</small></div>
              <div id="attributionCards" class="attribution-cards"></div>
            </section>'''
    text = text[:start] + panel + text[end:]

    anchor = '            <button id="toggleAdvanced" class="advanced-toggle" type="button">调整事件时间 / 添加备注 ▾</button>'
    quick = '''            <section id="quickTimeEditor" class="quick-time-editor hidden">
              <div class="quick-time-summary"><span>修正时间</span><strong id="quickTimeValue">00:00.000</strong><small id="quickTimeDelta">原始时间</small></div>
              <div class="quick-time-actions"><button data-quick-adjust="-0.5">−0.5s</button><button id="useCurrentFrame">设为当前画面 <kbd>T</kbd></button><button data-quick-adjust="0.5">+0.5s</button><button id="resetEventTime">重置</button></div>
              <small class="quick-time-hint">[ / ] 微调 0.5 秒；暂停到目标画面后按 T</small>
            </section>
            <button id="toggleAdvanced" class="advanced-toggle" type="button">精确时间 / 备注 ▾</button>'''
    if anchor not in text:
        raise RuntimeError("v21 advanced anchor missing")
    return text.replace(anchor, quick, 1)


def patch_js_v21(text: str) -> str:
    text = _js_v20(text)
    old = "selectedTeam: 'unknown', selectedGoalSide: 'unknown', correctedTime: null,"
    new = "selectedTeam: 'unknown', selectedGoalSide: 'unknown', attributionByLabel: {}, correctedTime: null,"
    if old not in text:
        raise RuntimeError("v21 state anchor missing")
    text = text.replace(old, new, 1)

    renderer = r'''function renderTeamAttribution() {
  const event = currentEvent();
  if (!event) return;
  const group = segmentEvents(event);
  const labels = (state.bootstrap.primary_labels || ["shot", "save", "set_piece"]).filter((label) => state.selectedSegmentLabels.has(label));
  const roles = { shot: "射门方", save: "扑救方", set_piece: "定位球执行方" };
  $("#attributionCards").innerHTML = labels.map((label) => {
    const item = group.find((x) => x.label === label) || event;
    const review = item.review || {}, evidence = item.team_evidence || {};
    state.attributionByLabel[label] ||= {
      event_team: review.event_team || evidence.suggested_event_team || "unknown",
      goal_side: review.goal_side || evidence.suggested_goal_side || (label === "set_piece" ? "not_applicable" : "unknown"),
    };
    const value = state.attributionByLabel[label];
    const teams = ["teamA", "teamB", "unknown"].map((team) => {
      const palette = teamPalette(item, team);
      const name = team === "unknown" ? "无法判断" : (palette.display_name || (team === "teamA" ? "队伍 A" : "队伍 B"));
      const dot = team === "unknown" ? "" : `<i style="background:${palette.hex || '#777'}"></i>`;
      return `<button data-attr-label="${label}" data-team="${team}" class="${value.event_team === team ? 'selected' : ''}">${dot}<span>${name}</span></button>`;
    }).join("");
    const goal = ["shot", "save"].includes(label) ? `<div class="compact-goal-row"><span>目标球门</span>${[["left", "← 左侧"], ["right", "右侧 →"], ["unknown", "不明确"]].map(([side, name]) => `<button data-attr-label="${label}" data-side="${side}" class="${value.goal_side === side ? 'selected' : ''}">${name}</button>`).join("")}</div>` : "";
    return `<article class="attribution-card label-${label}"><header><strong>${LABEL_NAMES[label]} · ${roles[label]}</strong><small>${evidence.suggested_event_team && evidence.suggested_event_team !== 'unknown' ? `自动建议 ${Math.round(Number(evidence.event_team_confidence || 0) * 100)}%` : '人工确认'}</small></header><div class="compact-team-row">${teams}</div>${goal}</article>`;
  }).join("");
  $("#attributionCards").querySelectorAll("[data-team]").forEach((button) => button.onclick = () => { state.attributionByLabel[button.dataset.attrLabel].event_team = button.dataset.team; renderTeamAttribution(); });
  $("#attributionCards").querySelectorAll("[data-side]").forEach((button) => button.onclick = () => { state.attributionByLabel[button.dataset.attrLabel].goal_side = button.dataset.side; renderTeamAttribution(); });
}'''
    text = v15._replace_function(text, "renderTeamAttribution", "renderSegmentLabelEditor", renderer)

    anchor = "  state.secondaryLabels = new Set(group.flatMap((item) => item.review.secondary_labels || []));"
    if anchor not in text:
        raise RuntimeError("v21 group anchor missing")
    text = text.replace(anchor, anchor + "\n  state.attributionByLabel = {};", 1)
    old = "      renderSegmentLabelEditor();\n      renderClassButtons();"
    if old not in text:
        raise RuntimeError("v21 selector anchor missing")
    text = text.replace(old, old + "\n      renderTeamAttribution();", 1)

    helpers = r'''function setCorrectedEventTime(value, notify = false) {
  const event = currentEvent(); if (!event) return;
  const corrected = Math.max(0, Number(value));
  $("#correctedTime").value = corrected.toFixed(3); state.correctedTime = corrected;
  renderTimeCorrection();
  if (notify) toast(`事件时间已设为 ${formatTime(corrected)}`);
}
function renderTimeCorrection() {
  const event = currentEvent(); if (!event) return;
  const corrected = Number($("#correctedTime").value || event.time_sec), delta = corrected - Number(event.time_sec);
  const changed = Math.abs(delta) >= .001, deltaText = changed ? `${delta > 0 ? "+" : ""}${delta.toFixed(1)}s` : "原始时间";
  $("#eventTime").textContent = formatTime(corrected); $("#quickTimeValue").textContent = formatTime(corrected);
  $("#timeDeltaBadge").textContent = changed ? deltaText : ""; $("#quickTimeDelta").textContent = deltaText;
  $("#timeEditButton").classList.toggle("changed", changed);
}

'''
    text = text.replace("let segmentSaveQueue = Promise.resolve();", helpers + "let segmentSaveQueue = Promise.resolve();", 1)
    time_anchor = '  $("#correctedTime").value = Number(state.correctedTime).toFixed(3);'
    text = text.replace(time_anchor, time_anchor + "\n  renderTimeCorrection();", 1)

    payload = '''    event_team: state.selectedTeam || "unknown",
    goal_side: state.selectedGoalSide || "unknown",
    compact_response: true,'''
    replacement = '''    event_team: "unknown",
    goal_side: "unknown",
    attribution_by_label: state.attributionByLabel,
    apply_time_to_all: true,
    compact_response: true,'''
    if payload not in text:
        raise RuntimeError("v21 payload anchor missing")
    text = text.replace(payload, replacement, 1)

    old_bind = '''  document.querySelectorAll("[data-adjust]").forEach((button) => button.onclick = () => { const input = $("#correctedTime"); input.value = Math.max(0, Number(input.value) + Number(button.dataset.adjust)).toFixed(3); });
  $("#usePlayerTime").onclick = () => { $("#correctedTime").value = $("#player").currentTime.toFixed(3); };'''
    new_bind = '''  document.querySelectorAll("[data-adjust]").forEach((button) => button.onclick = () => setCorrectedEventTime(Number($("#correctedTime").value) + Number(button.dataset.adjust)));
  $("#usePlayerTime").onclick = () => setCorrectedEventTime($("#player").currentTime, true);
  $("#correctedTime").oninput = renderTimeCorrection;
  $("#timeEditButton").onclick = () => $("#quickTimeEditor").classList.toggle("hidden");
  document.querySelectorAll("[data-quick-adjust]").forEach((button) => button.onclick = () => setCorrectedEventTime(Number($("#correctedTime").value) + Number(button.dataset.quickAdjust)));
  $("#useCurrentFrame").onclick = () => setCorrectedEventTime($("#player").currentTime, true);
  $("#resetEventTime").onclick = () => setCorrectedEventTime(currentEvent()?.time_sec || 0);'''
    if old_bind not in text:
        raise RuntimeError("v21 binding anchor missing")
    text = text.replace(old_bind, new_bind, 1)
    text = text.replace('$("#toggleAdvanced").textContent = open ? "收起时间与备注 ▴" : "调整事件时间 / 添加备注 ▾";', '$("#toggleAdvanced").textContent = open ? "收起精确时间与备注 ▴" : "精确时间 / 备注 ▾";', 1)

    old_keys = '''    else if (event.key === "Enter") saveDecision("accepted");
    else if (["x", "X", "Delete", "Backspace"].includes(event.key)) saveDecision("deleted");'''
    new_keys = '''    else if (event.key === "Enter") saveMultiLabelSegment();
    else if (["x", "X", "Delete", "Backspace"].includes(event.key)) { state.selectedSegmentLabels.clear(); renderSegmentLabelEditor(); renderTeamAttribution(); saveMultiLabelSegment(); }
    else if (["t", "T"].includes(event.key)) setCorrectedEventTime(player.currentTime, true);
    else if (event.key === "[") setCorrectedEventTime(Number($("#correctedTime").value) - .5);
    else if (event.key === "]") setCorrectedEventTime(Number($("#correctedTime").value) + .5);'''
    if old_keys not in text:
        raise RuntimeError("v21 keyboard anchor missing")
    return text.replace(old_keys, new_keys, 1)


hierarchical.patch_hierarchical_html = patch_html_v21
hierarchical.patch_hierarchical_js = patch_js_v21
hierarchical.HIERARCHICAL_CSS += r"""
.attribution-heading{display:flex;justify-content:space-between;align-items:center;margin:7px 0 5px}.attribution-heading strong{font-size:11px}.attribution-heading small{color:#748292;font-size:9px}.attribution-cards{display:grid;gap:5px}.attribution-card{padding:6px;border:1px solid #304050;border-left-width:3px;border-radius:7px;background:#101923}.attribution-card.label-shot{border-left-color:var(--shot)}.attribution-card.label-save{border-left-color:var(--save)}.attribution-card.label-set_piece{border-left-color:var(--set-piece)}.attribution-card header{display:flex;justify-content:space-between;margin-bottom:5px}.attribution-card header strong{font-size:10px}.attribution-card header small{color:#748292;font-size:8px}.compact-team-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px}.compact-team-row button,.compact-goal-row button{min-width:0;min-height:31px;color:#c8d1db;border:1px solid #394858;border-radius:6px;background:#19232e;cursor:pointer;font-size:10px;white-space:nowrap}.compact-team-row button{display:flex;justify-content:center;align-items:center;gap:5px}.compact-team-row button i{width:11px;height:11px;border:2px solid #fff9;border-radius:50%}.compact-team-row button.selected,.compact-goal-row button.selected{color:#101710;border-color:var(--accent);background:var(--accent);font-weight:800}.compact-goal-row{display:grid;grid-template-columns:66px repeat(3,minmax(0,1fr));align-items:center;gap:4px;margin-top:4px}.compact-goal-row>span{color:#8290a0;font-size:9px;text-align:center}
.time-edit-button{display:inline-flex;align-items:center;gap:6px;padding:3px 6px;color:inherit;border:1px solid transparent;border-radius:7px;background:transparent;cursor:pointer;white-space:nowrap}.time-edit-button:hover,.time-edit-button.changed{border-color:#66788b;background:#1a2430}.time-edit-button i{color:#718092;font-size:11px;font-style:normal}.time-edit-button small{color:#f1d470;font:800 9px ui-monospace,monospace}.quick-time-editor{margin:6px 0;padding:7px;border:1px solid #3b4a59;border-radius:7px;background:#101923}.quick-time-summary{display:flex;align-items:baseline;gap:8px;margin-bottom:6px}.quick-time-summary span,.quick-time-summary small{color:#8390a0;font-size:9px}.quick-time-summary strong{font:800 15px ui-monospace,monospace}.quick-time-actions{display:grid;grid-template-columns:.75fr 1.7fr .75fr .75fr;gap:4px}.quick-time-actions button{min-height:32px;color:#cdd6df;border:1px solid #3a4856;border-radius:6px;background:#1c2732;cursor:pointer;font-size:10px}#useCurrentFrame{color:#102118;border-color:#5dd39e;background:#5dd39e;font-weight:800}.quick-time-hint{display:block;margin-top:5px;color:#667585;font-size:8px;text-align:center}
"""

if __name__ == "__main__":
    hierarchical.main()
