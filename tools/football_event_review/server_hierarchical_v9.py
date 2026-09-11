#!/usr/bin/env python3
"""Segment-level multi-label launcher for the team-aware review UI.

The database remains label-level for backward-compatible exports, while the
browser queue and decisions are grouped by ``segment_id`` so overlapping
shot/save predictions are watched and reviewed only once.
"""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v8  # noqa: F401  # install all v8/team-aware patches


_html_v8 = hierarchical.patch_hierarchical_html
_js_v8 = hierarchical.patch_hierarchical_js


def _replace_function(text: str, name: str, next_name: str, replacement: str) -> str:
    start = text.index(f"function {name}(")
    end = text.index(f"function {next_name}(", start)
    if text[max(0, end - 6) : end] == "async ":
        end -= 6
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def patch_html_v9(text: str) -> str:
    text = _html_v8(text).replace(
        "hierarchical-v8-team-aware-20260818",
        "hierarchical-v9-segment-multilabel-20260818",
    )
    text = text.replace("高召回复核", "高召回 · 片段多标签复核", 1)
    anchor = '          <div class="rapid-actions" aria-label="快速审核操作">'
    panel = '''          <section id="segmentLabelEditor" class="segment-label-editor">
            <div class="segment-label-heading">
              <strong>本片段事件标签（可多选）</strong>
              <span id="segmentLabelHint">一次播放、一次提交</span>
            </div>
            <div id="segmentLabelButtons" class="segment-label-buttons"></div>
            <button id="saveSegmentLabelsBtn" class="save-segment-labels" type="button">确认所选标签并播放下一片段</button>
          </section>
'''
    if anchor not in text:
        raise RuntimeError("segment editor HTML anchor missing")
    return text.replace(anchor, panel + anchor, 1)


def patch_js_v9(text: str) -> str:
    text = _js_v8(text)
    old_state = "  filtered: [], selectedLabel: null, secondaryLabels: new Set(), selectedTeam: 'unknown', selectedGoalSide: 'unknown', correctedTime: null,"
    new_state = "  filtered: [], selectedLabel: null, secondaryLabels: new Set(), selectedSegmentLabels: new Set(), currentSegmentKey: null, selectedTeam: 'unknown', selectedGoalSide: 'unknown', correctedTime: null,"
    if old_state not in text:
        raise RuntimeError("v9 state anchor missing")
    text = text.replace(old_state, new_state, 1)

    apply_filters = r'''function segmentKey(event) {
  return event?.segment_id || event?.id || "";
}

function segmentEvents(event = currentEvent()) {
  if (!event || !state.video) return [];
  const key = segmentKey(event);
  return state.video.events.filter((item) => segmentKey(item) === key);
}

function segmentGroups(events = state.video?.events || []) {
  const groups = new Map();
  events.forEach((event) => {
    const key = segmentKey(event);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(event);
  });
  return [...groups.values()].sort((a, b) =>
    Math.min(...a.map((x) => x.support_start_sec ?? x.start_sec ?? x.time_sec)) -
    Math.min(...b.map((x) => x.support_start_sec ?? x.start_sec ?? x.time_sec))
  );
}

function segmentRepresentative(group) {
  return group.find((item) => item.id === state.currentEventId) ||
    [...group].sort((a, b) => Number(b.score || 0) - Number(a.score || 0))[0];
}

function segmentStatus(group) {
  if (group.some((item) => item.review.status === "unreviewed")) return "unreviewed";
  if (group.every((item) => item.review.status === "deleted")) return "deleted";
  if (group.some((item) => item.review.status === "modified" || item.review.status === "deleted")) return "modified";
  return "accepted";
}

function applyFilters() {
  if (!state.video) return;
  const label = $("#labelFilter").value;
  const status = $("#statusFilter").value;
  const evaluation = $("#evaluationFilter").value;
  const conflictOnly = $("#conflictOnly").checked;
  const source = $("#sourceFilter")?.value || "all";
  const groups = segmentGroups().filter((group) =>
    (label === "all" || group.some((event) => event.label === label)) &&
    (status === "all" || (status === "unreviewed" ? group.some((event) => event.review.status === status) : group.some((event) => event.review.status === status))) &&
    (evaluation === "all" || group.some((event) => event.evaluation_status === evaluation)) &&
    (!conflictOnly || group.some(isConflict)) &&
    (source === "all" || group.some((event) =>
      (source === "model" && event.review_source !== "whistle_rescue") ||
      (source === "whistle_flagged" && event.review_source === "dino_frame_whistle") ||
      (source === "whistle_rescue" && event.review_source === "whistle_rescue")
    ))
  );
  state.filtered = groups.map(segmentRepresentative);
  renderQueue();
  drawTimeline();
}'''
    text = _replace_function(text, "applyFilters", "renderProgress", apply_filters)

    render_queue = r'''function renderQueue() {
  const list = $("#queueList");
  list.innerHTML = "";
  const labelCount = state.filtered.reduce((sum, event) => sum + segmentEvents(event).length, 0);
  $("#queueCount").textContent = `${state.filtered.length} 个片段 · ${labelCount} 个标签`;
  state.filtered.forEach((event) => {
    const group = segmentEvents(event);
    const labels = [...new Set(group.map((item) => item.label))];
    const status = segmentStatus(group);
    const score = Math.max(...group.map((item) => Number(item.score || 0)));
    const whistle = Math.max(...group.map((item) => Number(item.whistle_score || 0)));
    const button = document.createElement("button");
    button.className = `queue-item segment-queue ${segmentKey(event) === segmentKey(currentEvent()) ? "current" : ""} ${status !== "unreviewed" ? "reviewed" : ""} ${status}`;
    const badges = labels.map((label) => `<em class="queue-label label-${label}">${LABEL_NAMES[label] || label}</em>`).join("");
    button.innerHTML = `<i class="color" style="background:${COLORS[labels[0]] || '#aaa'}"></i><time>${formatTime(Math.min(...group.map((x) => x.time_sec)))}</time><span class="label multi-labels">${badges}${whistle ? ' <em class="whistle-mini">哨</em>' : ''}</span><span class="score">${score.toFixed(3)}</span>`;
    button.onclick = () => selectEvent(event.id, true);
    list.appendChild(button);
  });
  requestAnimationFrame(() => list.querySelector(".current")?.scrollIntoView({ block: "nearest" }));
}'''
    text = _replace_function(text, "renderQueue", "renderEvent", render_queue)

    # Initialize the multi-label choice once per segment and render a unified heading.
    source_anchor = "  const sourceLabel = event.label;"
    source_replacement = '''  const group = segmentEvents(event);
  const groupKey = segmentKey(event);
  if (state.currentSegmentKey !== groupKey) {
    state.currentSegmentKey = groupKey;
    state.selectedSegmentLabels = new Set(group.filter((item) => item.review.status !== "deleted").map((item) => item.label));
  }
  const sourceLabel = event.label;'''
    if source_anchor not in text:
        raise RuntimeError("renderEvent source anchor missing")
    text = text.replace(source_anchor, source_replacement, 1)
    text = text.replace(
        "  classElement.className = `event-class ${sourceLabel}`;\n  classElement.textContent = LABEL_NAMES[sourceLabel] || sourceLabel;",
        "  classElement.className = `event-class multi`;\n  classElement.innerHTML = [...new Set(group.map((item) => item.label))].map((label) => `<span class=\"event-label label-${label}\">${LABEL_NAMES[label] || label}</span>`).join('<b>＋</b>');",
        1,
    )
    text = text.replace(
        '  $("#supportRange").textContent = `片段 ${formatTime(event.support_start_sec)} – ${formatTime(event.support_end_sec)}`;',
        '  $("#supportRange").textContent = `片段 ${formatTime(Math.min(...group.map((item) => item.support_start_sec ?? item.start_sec)))} – ${formatTime(Math.max(...group.map((item) => item.support_end_sec ?? item.end_sec)))}`;',
        1,
    )
    text = text.replace(
        "  const siblings = state.video.events.filter((item) => item.segment_id === event.segment_id && item.review.status === \"unreviewed\");",
        "  const siblings = group.filter((item) => item.review.status === \"unreviewed\");",
        1,
    )
    text = text.replace(
        "  renderClassButtons();\n  renderTeamAttribution();",
        "  renderClassButtons();\n  renderTeamAttribution();\n  renderSegmentLabelEditor();",
        1,
    )

    play_anchor = "function playEventContext(event) {"
    editor_functions = r'''function renderSegmentLabelEditor() {
  const group = segmentEvents();
  const labels = [...new Set(group.map((item) => item.label))];
  $("#segmentLabelButtons").innerHTML = labels.map((label) => {
    const item = group.find((candidate) => candidate.label === label);
    const selected = state.selectedSegmentLabels.has(label);
    return `<button type="button" data-segment-label="${label}" class="label-${label} ${selected ? 'selected' : ''}"><i>${selected ? '✓' : '×'}</i><span>${LABEL_NAMES[label] || label}</span><small>D ${Number(item?.score || 0).toFixed(2)} · F ${Number(item?.frame_detection_scores?.[label] || 0).toFixed(2)}</small></button>`;
  }).join("");
  $("#segmentLabelButtons").querySelectorAll("button").forEach((button) => {
    button.onclick = () => {
      const label = button.dataset.segmentLabel;
      if (state.selectedSegmentLabels.has(label)) state.selectedSegmentLabels.delete(label);
      else state.selectedSegmentLabels.add(label);
      renderSegmentLabelEditor();
    };
  });
  const selectedNames = labels.filter((label) => state.selectedSegmentLabels.has(label)).map((label) => LABEL_NAMES[label]);
  $("#segmentLabelHint").textContent = selectedNames.length ? `保留：${selectedNames.join(" + ")}` : "全部删除为误报";
  $("#saveSegmentLabelsBtn").classList.toggle("delete-all", selectedNames.length === 0);
  $("#saveSegmentLabelsBtn").textContent = selectedNames.length ? "确认所选标签并播放下一片段" : "删除本片段全部预测并播放下一片段";
}

async function saveMultiLabelSegment() {
  const event = currentEvent();
  if (!event) return;
  const group = segmentEvents(event);
  const selected = state.selectedSegmentLabels;
  const missingSetPieceType = group.some((item) => item.label === "set_piece" && selected.has("set_piece") && !(item.review.secondary_labels || []).some((label) => (state.bootstrap.set_piece_type_labels || []).includes(label)) && item.id !== event.id);
  if (missingSetPieceType) {
    toast("该片段包含定位球：请先点定位球标签单独选择具体类型");
    return;
  }
  if (event.label === "set_piece" && selected.has("set_piece") && !(state.bootstrap.set_piece_type_labels || []).some((label) => state.secondaryLabels.has(label))) {
    toast("请先选择定位球类型");
    return;
  }
  const reviewer = $("#reviewerInput").value.trim();
  $("#player").play().catch(() => {});
  try {
    for (const item of group) {
      const keep = selected.has(item.label);
      const isCurrent = item.id === event.id;
      const primary = isCurrent ? state.selectedLabel : (item.review.corrected_label || item.label);
      const details = isCurrent ? [...state.secondaryLabels] : (item.review.secondary_labels || []);
      const evidence = item.team_evidence || {};
      const payload = keep ? {
        status: item.review.status === "modified" || primary !== item.label ? "modified" : "accepted",
        reviewer,
        note: isCurrent ? $("#note").value.trim() : (item.review.note || "同片段多标签审核"),
        corrected_label: primary,
        secondary_labels: details,
        corrected_time_sec: isCurrent ? Number($("#correctedTime").value) : (item.review.corrected_time_sec ?? item.time_sec),
        event_team: isCurrent ? (state.selectedTeam || "unknown") : (item.review.event_team || evidence.suggested_event_team || "unknown"),
        goal_side: isCurrent ? (state.selectedGoalSide || "unknown") : (item.review.goal_side || evidence.suggested_goal_side || (item.label === "set_piece" ? "not_applicable" : "unknown")),
      } : { status: "deleted", reviewer, note: "同片段多标签审核：未选中" };
      const updated = await api(`/api/events/${item.id}/decision`, { method: "POST", body: JSON.stringify(payload) });
      state.video.events[state.video.events.findIndex((candidate) => candidate.id === item.id)] = updated;
    }
    const kept = [...selected].map((label) => LABEL_NAMES[label] || label);
    await refreshBootstrap();
    applyFilters();
    toast(kept.length ? `已保存多标签：${kept.join(" + ")}` : "已删除本片段全部预测");
    nextEvent(1, true, true);
  } catch (error) { toast(error.message); }
}

'''
    if play_anchor not in text:
        raise RuntimeError("segment editor JS anchor missing")
    text = text.replace(play_anchor, editor_functions + play_anchor, 1)

    next_event = r'''function nextEvent(direction = 1, onlyUnreviewed = false, autoPlay = true) {
  if (!state.video) return;
  const allGroups = segmentGroups();
  const currentKey = segmentKey(currentEvent());
  const currentIndex = allGroups.findIndex((group) => segmentKey(group[0]) === currentKey);
  let targetGroup = null;
  if (onlyUnreviewed) {
    const candidates = direction > 0 ? allGroups.slice(currentIndex + 1) : allGroups.slice(0, Math.max(0, currentIndex)).reverse();
    targetGroup = candidates.find((group) => group.some((event) => event.review.status === "unreviewed")) || null;
  } else {
    targetGroup = allGroups[currentIndex < 0 ? 0 : currentIndex + direction] || null;
  }
  selectEvent(segmentRepresentative(targetGroup).id, true, autoPlay);
}'''
    text = _replace_function(text, "nextEvent", "navigateAdjacent", next_event)

    navigate = r'''function navigateAdjacent(direction) {
  if (!state.video) return;
  const groups = segmentGroups();
  const currentKey = segmentKey(currentEvent());
  const index = groups.findIndex((group) => segmentKey(group[0]) === currentKey);
  const target = groups[index + direction];
  if (!target) { toast(direction > 0 ? "已经是最后一个片段" : "已经是第一个片段"); return; }
  selectEvent(segmentRepresentative(target).id, true, true);
}'''
    text = _replace_function(text, "navigateAdjacent", "saveDecision", navigate)

    # Main fast actions now operate on the complete segment, not one label row.
    text = text.replace(
        '  $("#acceptBtn").onclick = () => saveDecision("accepted");\n  $("#deleteBtn").onclick = () => saveDecision("deleted");',
        '  $("#acceptBtn").onclick = saveMultiLabelSegment;\n  $("#deleteBtn").onclick = () => { state.selectedSegmentLabels.clear(); renderSegmentLabelEditor(); saveMultiLabelSegment(); };\n  $("#saveSegmentLabelsBtn").onclick = saveMultiLabelSegment;',
        1,
    )
    return text


hierarchical.patch_hierarchical_html = patch_html_v9
hierarchical.patch_hierarchical_js = patch_js_v9
hierarchical.HIERARCHICAL_CSS += r'''
.segment-label-editor { margin: 7px 0 6px; padding: 8px; border: 1px solid #405365; border-radius: 9px; background: #111c27; }
.segment-label-heading { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.segment-label-heading strong { color: #eff5fa; font-size: 12px; }
.segment-label-heading span { color: #9eadba; font-size: 10px; }
.segment-label-buttons { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
.segment-label-buttons button { min-height: 46px; display: grid; grid-template-columns: 18px 1fr; grid-template-rows: 1fr 1fr; align-items: center; padding: 5px 7px; color: #97a5b3; background: #18232e; border: 1px solid #3d4a58; border-radius: 7px; cursor: pointer; text-align: left; }
.segment-label-buttons button i { grid-row: 1 / 3; font-style: normal; font-size: 16px; }
.segment-label-buttons button span { font-weight: 850; }
.segment-label-buttons button small { color: #7e8a98; font-size: 9px; }
.segment-label-buttons button.selected.label-shot { color: #111; background: var(--shot); border-color: var(--shot); }
.segment-label-buttons button.selected.label-save { color: #111; background: var(--save); border-color: var(--save); }
.segment-label-buttons button.selected.label-set_piece { color: #111; background: var(--set-piece); border-color: var(--set-piece); }
.segment-label-buttons button.selected small { color: #17202a; }
.save-segment-labels { width: 100%; height: 38px; margin-top: 7px; color: #101710; background: var(--accent); border: 1px solid var(--accent); border-radius: 7px; cursor: pointer; font-weight: 850; }
.save-segment-labels.delete-all { color: #fff; background: #8d3038; border-color: #bd4a55; }
.event-class.multi { display: inline-flex; align-items: center; gap: 5px; background: transparent; padding: 0; }
.event-class.multi b { color: #7f8b99; font-size: 11px; }
.event-label { display: inline-flex; padding: 5px 8px; color: #111; border-radius: 6px; font-size: 12px; font-weight: 900; }
.event-label.label-shot, .queue-label.label-shot { background: var(--shot); }
.event-label.label-save, .queue-label.label-save { background: var(--save); }
.event-label.label-set_piece, .queue-label.label-set_piece { background: var(--set-piece); }
.queue-item.segment-queue { grid-template-columns: 4px 72px minmax(120px, 1fr) 48px; }
.multi-labels { display: flex; flex-wrap: wrap; gap: 3px; }
.queue-label { padding: 2px 5px; color: #111; border-radius: 4px; font-style: normal; font-size: 10px; font-weight: 850; }
.whistle-mini { color: #f4d76a; font-style: normal; }
#segmentBatchActions { display: none !important; }
'''


if __name__ == "__main__":
    hierarchical.main()
