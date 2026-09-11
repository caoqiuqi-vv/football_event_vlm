#!/usr/bin/env python3
"""Three-way primary-label editing with persistent human-added segment labels."""

from __future__ import annotations

import json
from http import HTTPStatus
from urllib.parse import unquote, urlparse

import server_hierarchical as hierarchical
import server_hierarchical_v14  # noqa: F401


_html_v14 = hierarchical.patch_hierarchical_html
_js_v14 = hierarchical.patch_hierarchical_js
_StoreV14 = hierarchical.HierarchicalReviewStore
_HandlerV14 = hierarchical.HierarchicalReviewHandler


class ThreeLabelSegmentStore(_StoreV14):
    """Persist all selected primary labels, including labels absent from the model."""

    def _insert_human_label(self, anchor: dict, label: str) -> str:
        segment_id = str(anchor.get("segment_id") or anchor["id"])
        event_id = f"{segment_id}_human_added_{label}"
        payload = {key: value for key, value in anchor.items() if key != "review"}
        labels = list(dict.fromkeys([*payload.get("segment_labels", []), label]))
        score = float(payload.get("dino_scores", {}).get(label, 0.0))
        payload.update({
            "id": event_id,
            "label": label,
            "score": score,
            "segment_labels": labels,
            "human_added": True,
            "evaluation_status": "human_added",
            "matching_gt_times": [],
        })
        with self.lock, self.connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO events
                   (id, video_id, source_label, source_time_sec, source_score, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event_id, anchor["video_id"], label, float(anchor["time_sec"]), score,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            connection.execute("INSERT OR IGNORE INTO reviews(event_id) VALUES (?)", (event_id,))
        return event_id

    def update_segment_labels(self, anchor_id: str, data: dict) -> dict:
        anchor = self._event_by_id(anchor_id)
        video_id = str(anchor["video_id"])
        segment_id = str(anchor.get("segment_id") or anchor["id"])
        valid = set(hierarchical.PRIMARY_LABELS)
        selected = list(dict.fromkeys(str(item) for item in data.get("selected_labels", [])))
        invalid = set(selected) - valid
        if invalid:
            raise ValueError(f"Invalid primary labels: {sorted(invalid)}")

        details_by_label = data.get("secondary_labels_by_label", {}) or {}
        set_piece_types = set(hierarchical.SET_PIECE_TYPES)
        if "set_piece" in selected and not set(details_by_label.get("set_piece", [])) & set_piece_types:
            raise ValueError("确认定位球前必须选择具体定位球类型")

        group = [
            event for event in self.events_for_video(video_id)
            if str(event.get("segment_id") or event["id"]) == segment_id
        ]
        by_label = {event["label"]: event for event in group}
        for label in selected:
            if label not in by_label:
                event_id = self._insert_human_label(anchor, label)
                by_label[label] = self._event_by_id(event_id)

        reviewer = str(data.get("reviewer", ""))
        note = str(data.get("note", ""))
        corrected_time = data.get("corrected_time_sec", anchor.get("time_sec"))
        active_label = str(data.get("active_label", anchor.get("label", "")))
        attribution_by_label = data.get("attribution_by_label", {}) or {}
        apply_time_to_all = bool(data.get("apply_time_to_all", False))
        for label, event in by_label.items():
            if label not in selected:
                self.update_review(event["id"], {
                    "status": "deleted", "reviewer": reviewer,
                    "note": "同片段多标签审核：未选中",
                })
                continue
            review = event.get("review", {})
            evidence = event.get("team_evidence", {})
            is_active = label == active_label
            attribution = attribution_by_label.get(label, {}) or {}
            event_team = attribution.get(
                "event_team",
                data.get("event_team", "unknown") if is_active else (review.get("event_team") or evidence.get("suggested_event_team") or "unknown"),
            )
            goal_side = attribution.get(
                "goal_side",
                data.get("goal_side", "unknown") if is_active else (review.get("goal_side") or evidence.get("suggested_goal_side") or ("not_applicable" if label == "set_piece" else "unknown")),
            )
            self.update_review(event["id"], {
                "status": "modified" if event.get("human_added") else "accepted",
                "corrected_label": label,
                "secondary_labels": list(details_by_label.get(label, [])),
                "corrected_time_sec": corrected_time if (is_active or apply_time_to_all) else (review.get("corrected_time_sec") or event["time_sec"]),
                "reviewer": reviewer,
                "note": note if is_active else (review.get("note") or "同片段多标签审核"),
                "event_team": event_team,
                "goal_side": goal_side,
            })
        return {
            "video_id": video_id,
            "segment_id": segment_id,
            "events": self.events_for_video(video_id),
        }


class ThreeLabelSegmentHandler(_HandlerV14):
    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path.startswith("/api/events/") and path.endswith("/segment-decision"):
            try:
                event_id = path.split("/")[3]
                return self.send_json(self.store.update_segment_labels(event_id, self.read_json()))
            except (ValueError, json.JSONDecodeError) as error:
                return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except KeyError as error:
                return self.send_json({"error": f"Not found: {error}"}, HTTPStatus.NOT_FOUND)
            except Exception as error:
                return self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        return super().do_POST()


def _replace_function(text: str, name: str, next_name: str, replacement: str) -> str:
    start = text.index(f"function {name}(")
    if text[max(0, start - 6):start] == "async ":
        start -= 6
    end = text.index(f"function {next_name}(", start)
    if text[max(0, end - 6):end] == "async ":
        end -= 6
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def patch_html_v15(text: str) -> str:
    text = _html_v14(text).replace(
        "hierarchical-v14-vertical-speed-rail-20260818",
        "hierarchical-v15-three-label-flow-20260818",
    )
    segment_start = text.index('          <section id="segmentLabelEditor"')
    segment_end = text.index('          </section>', segment_start) + len('          </section>')
    segment = text[segment_start:segment_end]
    text = text[:segment_start] + text[segment_end:]
    rapid_start = text.index('          <div class="rapid-actions"')
    rapid_end = text.index('          </div>', rapid_start) + len('          </div>')
    text = text[:rapid_end] + "\n" + segment + text[rapid_end:]
    text = text.replace(
        '<strong>本片段事件标签（可多选）</strong>',
        '<strong>① 本片段主事件（可多选）</strong>',
        1,
    )
    text = text.replace(
        '<span id="segmentLabelHint">一次播放、一次提交</span>',
        '<span id="segmentLabelHint">模型预测默认选中，可补选或取消</span>',
        1,
    )
    text = text.replace(
        '<span>事件详情（按已选标签补充）</span>',
        '<span>② 事件详情（按已选标签补充）</span>',
        1,
    )
    return text


def patch_js_v15(text: str) -> str:
    text = _js_v14(text)
    editor = r'''function renderSegmentLabelEditor() {
  const group = segmentEvents();
  const labels = state.bootstrap.primary_labels || ["shot", "save", "set_piece"];
  $("#segmentLabelButtons").innerHTML = labels.map((label) => {
    const item = group.find((candidate) => candidate.label === label);
    const selected = state.selectedSegmentLabels.has(label);
    const predicted = Boolean(item && !item.human_added);
    const score = Number(item?.score ?? currentEvent()?.dino_scores?.[label] ?? 0);
    const frameScore = Number(item?.frame_detection_scores?.[label] ?? currentEvent()?.frame_detection_scores?.[label] ?? 0);
    const evidence = predicted ? `模型 D ${score.toFixed(2)} · F ${frameScore.toFixed(2)}` : (item?.human_added ? "人工已补充" : "人工补充事件");
    return `<button type="button" data-segment-label="${label}" class="label-${label} ${selected ? 'selected' : ''} ${predicted ? 'model-predicted' : 'manual-option'}"><i>${selected ? '✓' : '+'}</i><span>${LABEL_NAMES[label] || label}</span><small>${evidence}</small></button>`;
  }).join("");
  $("#segmentLabelButtons").querySelectorAll("button").forEach((button) => {
    button.onclick = () => {
      const label = button.dataset.segmentLabel;
      if (state.selectedSegmentLabels.has(label)) state.selectedSegmentLabels.delete(label);
      else state.selectedSegmentLabels.add(label);
      renderSegmentLabelEditor();
      renderClassButtons();
    };
  });
  const selectedNames = labels.filter((label) => state.selectedSegmentLabels.has(label)).map((label) => LABEL_NAMES[label]);
  $("#segmentLabelHint").textContent = selectedNames.length ? `最终保留：${selectedNames.join(" + ")}` : "未选择事件：将删除本片段预测";
  $("#saveSegmentLabelsBtn").classList.toggle("delete-all", selectedNames.length === 0);
  $("#saveSegmentLabelsBtn").textContent = selectedNames.length ? "确认所选标签并播放下一片段" : "删除本片段全部预测并播放下一片段";
}'''
    text = _replace_function(text, "renderSegmentLabelEditor", "saveMultiLabelSegment", editor)

    save = r'''async function saveMultiLabelSegment() {
  const event = currentEvent();
  if (!event) return;
  const selected = state.selectedSegmentLabels;
  if (selected.has("set_piece") && !(state.bootstrap.set_piece_type_labels || []).some((label) => state.secondaryLabels.has(label))) {
    toast("已保留定位球，请先选择任意球、点球、角球等具体类型");
    $("#setPieceTypePanel").scrollIntoView({ block: "nearest" });
    return;
  }
  const shotDetails = new Set(state.bootstrap.shot_detail_labels || []);
  const setPieceDetails = new Set(state.bootstrap.set_piece_type_labels || []);
  const detailsByLabel = {
    shot: [...state.secondaryLabels].filter((label) => shotDetails.has(label)),
    save: [],
    set_piece: [...state.secondaryLabels].filter((label) => setPieceDetails.has(label)),
  };
  const payload = {
    selected_labels: [...selected],
    secondary_labels_by_label: detailsByLabel,
    active_label: event.label,
    corrected_time_sec: Number($("#correctedTime").value),
    reviewer: $("#reviewerInput").value.trim(),
    note: $("#note").value.trim(),
    event_team: state.selectedTeam || "unknown",
    goal_side: state.selectedGoalSide || "unknown",
  };
  $("#player").play().catch(() => {});
  try {
    const result = await api(`/api/events/${event.id}/segment-decision`, { method: "POST", body: JSON.stringify(payload) });
    state.video.events = result.events;
    const kept = [...selected].map((label) => LABEL_NAMES[label] || label);
    await refreshBootstrap();
    applyFilters();
    toast(kept.length ? `已保存多标签：${kept.join(" + ")}` : "已删除本片段全部预测");
    nextEvent(1, true, true);
  } catch (error) { toast(error.message); }
}'''
    text = _replace_function(text, "saveMultiLabelSegment", "playEventContext", save)
    return text


hierarchical.HierarchicalReviewStore = ThreeLabelSegmentStore
hierarchical.HierarchicalReviewHandler = ThreeLabelSegmentHandler
hierarchical.patch_hierarchical_html = patch_html_v15
hierarchical.patch_hierarchical_js = patch_js_v15
hierarchical.HIERARCHICAL_CSS += r'''
/* v15: navigation first, then one continuous coarse-to-fine annotation flow. */
.rapid-actions { margin-bottom: 5px; }
.segment-label-editor { margin: 0 0 5px; border-color: #526579; }
.segment-label-buttons { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.segment-label-buttons button.manual-option:not(.selected) { border-style: dashed; background: #141d27; opacity: .78; }
.segment-label-buttons button.model-predicted:not(.selected) small { color: #d59ba3; }
.segment-label-buttons button i { font-style: normal; }
.detail-heading span { font-weight: 850; }
'''


if __name__ == "__main__":
    hierarchical.main()
