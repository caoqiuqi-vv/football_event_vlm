#!/usr/bin/env python3
'''GT-anchor-safe multi-user review UI layered on the v32 HTTPS service.'''

from __future__ import annotations

import server_hierarchical as hierarchical
import server_multiuser_v32 as v32


_html_v32 = hierarchical.patch_hierarchical_html
_js_v32 = hierarchical.patch_hierarchical_js


def patch_html_v33(text: str) -> str:
    text = _html_v32(text).replace(
        "multiuser-v31-second-confirmation-header-20260903",
        "multiuser-v33-gt-anchor-safe-20260903",
        1,
    )
    old = '<small class="prediction-caption">模型预测</small>'
    new = '<small id="taskSourceCaption" class="prediction-caption">审核来源</small><span id="modelEvidenceSummary" class="model-evidence-summary"></span>'
    if old not in text:
        raise RuntimeError("v33 prediction caption anchor missing")
    return text.replace(old, new, 1)


def patch_js_v33(text: str) -> str:
    text = _js_v32(text)
    old_representative = '''function segmentRepresentative(group) {
  return group.find((item) => item.id === state.currentEventId) ||
    [...group].sort((a, b) => Number(b.score || 0) - Number(a.score || 0))[0];
}'''
    new_representative = '''function segmentRepresentative(group) {
  return group.find((item) => item.id === state.currentEventId) ||
    group.find((item) => item.task_source === "gt" && (item.matching_gt_times || []).length) ||
    [...group].sort((a, b) => Number(b.score || 0) - Number(a.score || 0))[0];
}'''
    if old_representative not in text:
        raise RuntimeError("v33 representative anchor missing")
    text = text.replace(old_representative, new_representative, 1)

    old_initial = '    state.selectedSegmentLabels = new Set(group.filter((item) => item.review.status !== "deleted").map((item) => item.label));'
    new_initial = '''    const wasReviewed = group.some((item) => item.review.status !== "unreviewed");
    const gtLabels = new Set(group.flatMap((item) => item.gt_labels || []));
    const gtItems = group.filter((item) => (item.matching_gt_times || []).length || gtLabels.has(item.label));
    const initialItems = wasReviewed ? group.filter((item) => item.review.status !== "deleted") : (gtItems.length ? gtItems : group);
    state.selectedSegmentLabels = new Set(initialItems.map((item) => item.label));'''
    if old_initial not in text:
        raise RuntimeError("v33 initial label anchor missing")
    text = text.replace(old_initial, new_initial, 1)

    old_header = '''  classElement.className = `event-class multi`;
  classElement.innerHTML = [...new Set(group.map((item) => item.label))].map((label) => `<span class="event-label label-${label}">${LABEL_NAMES[label] || label}</span>`).join('<b>＋</b>');'''
    new_header = '''  const explicitGtLabels = [...new Set(group.flatMap((item) => item.gt_labels || []))];
  const legacyGtItems = group.filter((item) => (item.matching_gt_times || []).length);
  const taskGtLabels = explicitGtLabels.length ? explicitGtLabels : [...new Set(legacyGtItems.map((item) => item.label))];
  const taskGtTimes = [...new Set(group.flatMap((item) => item.gt_times || item.matching_gt_times || []).map(Number).filter(Number.isFinite))].sort((a, b) => a - b);
  const explicitCandidateLabels = [...new Set(group.flatMap((item) => item.candidate_labels || []))];
  const legacyCandidateTimes = group.flatMap((item) => (item.evidence_anchors || []).filter((anchor) => anchor.source === "candidate").map((anchor) => Number(anchor.time_sec)));
  const taskCandidateTimes = [...new Set([...group.flatMap((item) => item.candidate_times || []), ...legacyCandidateTimes].map(Number).filter(Number.isFinite))].sort((a, b) => a - b);
  const taskCandidateLabels = explicitCandidateLabels.length ? explicitCandidateLabels : (taskCandidateTimes.length ? [...new Set(group.map((item) => item.label))] : []);
  const isGtTask = taskGtLabels.length > 0 || group.some((item) => item.task_source === "gt");
  const primaryLabels = isGtTask ? taskGtLabels : [...new Set(group.map((item) => item.label))];
  classElement.className = `event-class multi ${isGtTask ? "gt-task" : "candidate-task"}`;
  classElement.innerHTML = primaryLabels.map((label) => `<span class="event-label label-${label}">${LABEL_NAMES[label] || label}</span>`).join('<b>＋</b>');
  $("#taskSourceCaption").textContent = isGtTask ? "现有 GT · 必须质检" : "模型候选 · 检查漏标";
  const modelSummary = $("#modelEvidenceSummary");
  if (taskCandidateLabels.length) {
    const labelText = taskCandidateLabels.map((label) => LABEL_NAMES[label] || label).join("+");
    const timeText = taskCandidateTimes.length ? ` @ ${taskCandidateTimes.map(formatTime).join(" / ")}` : "";
    modelSummary.textContent = `模型：${labelText}${timeText}`;
    modelSummary.classList.remove("no-candidate");
  } else {
    modelSummary.textContent = "模型：未覆盖";
    modelSummary.classList.add("no-candidate");
  }
  if (isGtTask && taskGtTimes.length) $("#eventTime").textContent = formatTime(taskGtTimes[0]);'''
    if old_header not in text:
        raise RuntimeError("v33 event header anchor missing")
    return text.replace(old_header, new_header, 1)


hierarchical.patch_hierarchical_html = patch_html_v33
hierarchical.patch_hierarchical_js = patch_js_v33
hierarchical.HIERARCHICAL_CSS += r'''
/* v33: the annotation under review and model evidence are separate concepts. */
.prediction-kicker { flex-wrap:wrap; row-gap:5px; }
#taskSourceCaption { color:#b9c5d2; }
#taskSourceCaption::before { content:""; display:inline-block; width:6px; height:6px; margin-right:5px; border-radius:50%; background:#f0c75e; vertical-align:1px; }
.model-evidence-summary { max-width:min(46vw,520px); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#83a6c5; font-size:10px; }
.model-evidence-summary.no-candidate { color:#687687; }
.event-class.gt-task { outline:1px solid #e5bd4c66; outline-offset:4px; }
.event-class.candidate-task { outline:1px solid #6ea9d966; outline-offset:4px; }
@media (max-width:760px) { .model-evidence-summary { max-width:72vw; flex-basis:100%; order:5; } }
'''


if __name__ == "__main__":
    v32.main()
