#!/usr/bin/env python3
# v33 plus canonical GT/candidate evidence display and multi-label semantics.
from __future__ import annotations
import server_hierarchical as hierarchical
import server_multiuser_v33 as v33

_html_v33=hierarchical.patch_hierarchical_html
_js_v33=hierarchical.patch_hierarchical_js

def patch_html_v34(text:str)->str:
 text=_html_v33(text).replace('multiuser-v33-gt-anchor-safe-20260903','multiuser-v34-canonical-multilabel-20260903',1)
 anchor='<span id="modelEvidenceSummary" class="model-evidence-summary"></span>'
 if anchor not in text:raise RuntimeError('v34 model summary anchor missing')
 text=text.replace(anchor,anchor+'<span id="canonicalMergeNotice" class="canonical-merge-notice"></span>',1)
 text=text.replace('模型预测默认选中，可补选或取消','一次播放、一次提交；同一时刻可同时选择多个标签',1)
 return text

def patch_js_v34(text:str)->str:
 text=_js_v33(text)
 old='''  const modelSummary = $("#modelEvidenceSummary");
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
 new='''  const modelSummary = $("#modelEvidenceSummary");
  const mergeNotice = $("#canonicalMergeNotice");
  const candidateAnchors = group.flatMap((item) => (item.evidence_anchors || []).filter((anchor) => anchor.source === "candidate"));
  const evidenceByTime = new Map();
  candidateAnchors.forEach((anchor) => {
    const time = Number(anchor.time_sec);
    if (!Number.isFinite(time)) return;
    const key = time.toFixed(3);
    if (!evidenceByTime.has(key)) evidenceByTime.set(key, { time, labels: new Set() });
    const allowed = anchor.family === "shot_save" ? ["shot", "save"] : anchor.family === "set_piece" ? ["set_piece"] : taskCandidateLabels;
    taskCandidateLabels.filter((label) => allowed.includes(label)).forEach((label) => evidenceByTime.get(key).labels.add(label));
  });
  if (evidenceByTime.size) {
    modelSummary.textContent = [...evidenceByTime.values()].sort((a,b)=>a.time-b.time).map((item) => `${formatTime(item.time)} [${[...item.labels].map((label)=>LABEL_NAMES[label]||label).join("+")}]`).join("；");
    modelSummary.classList.remove("no-candidate");
  } else {
    modelSummary.textContent = "模型未覆盖";
    modelSummary.classList.add("no-candidate");
  }
  const distances = taskGtTimes.flatMap((gt) => taskCandidateTimes.map((candidate) => Math.abs(gt - candidate)));
  const nearestDelta = distances.length ? Math.min(...distances) : Infinity;
  if (isGtTask && nearestDelta <= 3) {
    mergeNotice.textContent = `已并入 GT · Δ${nearestDelta.toFixed(1)}s · 只审核一次`;
    mergeNotice.classList.add("merged");
  } else {
    mergeNotice.textContent = isGtTask ? "GT 必须质检" : "多标签候选 · 一次审核";
    mergeNotice.classList.remove("merged");
  }
  if (isGtTask && taskGtTimes.length) $("#eventTime").textContent = formatTime(taskGtTimes[0]);'''
 if old not in text:raise RuntimeError('v34 v33 evidence block missing')
 return text.replace(old,new,1)

hierarchical.patch_hierarchical_html=patch_html_v34
hierarchical.patch_hierarchical_js=patch_js_v34
hierarchical.HIERARCHICAL_CSS += r'''
.prediction-kicker{column-gap:8px}.model-evidence-summary{padding:3px 7px;border-radius:6px;background:#142536;color:#9cc8eb;max-width:min(54vw,680px)}
.canonical-merge-notice{display:inline-flex;padding:3px 7px;border:1px solid #536274;border-radius:6px;color:#9ba9b8;background:#161d25;font-size:10px;font-weight:750;white-space:nowrap}
.canonical-merge-notice.merged{border-color:#3a8064;color:#8ee0bd;background:#10271f}
.segment-label-editor strong::after{content:" · 多标签一次确认";color:#78a9cf;font-size:10px;font-weight:650}
@media(max-width:760px){.canonical-merge-notice{flex-basis:auto}.model-evidence-summary{max-width:76vw}}
'''

if __name__=='__main__':v33.v32.main()
