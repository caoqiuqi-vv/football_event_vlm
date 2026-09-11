#!/usr/bin/env python3
"""Optimistic review actions backed by compact, ordered segment saves."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v15 as v15
import server_hierarchical_v19  # noqa: F401


_html_v19 = hierarchical.patch_hierarchical_html
_js_v19 = hierarchical.patch_hierarchical_js
_StoreV19 = hierarchical.HierarchicalReviewStore


class CompactSegmentStore(_StoreV19):
    def update_segment_labels(self, anchor_id: str, data: dict) -> dict:
        result = super().update_segment_labels(anchor_id, data)
        if not data.get("compact_response"):
            return result
        segment_id = str(result["segment_id"])
        events = result.pop("events")
        result["segment_events"] = [
            event for event in events
            if str(event.get("segment_id") or event["id"]) == segment_id
        ]
        return result


def patch_html_v20(text: str) -> str:
    return _html_v19(text).replace(
        "hierarchical-v19-stable-review-scroll-20260818",
        "hierarchical-v20-instant-actions-20260818",
    )


def patch_js_v20(text: str) -> str:
    text = _js_v19(text)
    replacement = r'''let segmentSaveQueue = Promise.resolve();
let bootstrapRefreshTimer = null;

function mergeSavedSegment(result) {
  if (!state.video || result.video_id !== state.currentVideoId || !result.segment_events) return;
  const key = String(result.segment_id);
  const untouched = state.video.events.filter((item) => String(item.segment_id || item.id) !== key);
  state.video.events = [...untouched, ...result.segment_events].sort((a, b) => Number(a.time_sec) - Number(b.time_sec) || String(a.label).localeCompare(String(b.label)));
}

function scheduleBootstrapRefresh() {
  window.clearTimeout(bootstrapRefreshTimer);
  bootstrapRefreshTimer = window.setTimeout(() => {
    refreshBootstrap().catch((error) => console.warn("background bootstrap refresh failed", error));
  }, 650);
}

async function saveMultiLabelSegment() {
  const event = currentEvent();
  if (!event) return;
  const selectedLabels = [...state.selectedSegmentLabels];
  const selected = new Set(selectedLabels);
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
    selected_labels: selectedLabels,
    secondary_labels_by_label: detailsByLabel,
    active_label: event.label,
    corrected_time_sec: Number($("#correctedTime").value),
    reviewer: $("#reviewerInput").value.trim(),
    note: $("#note").value.trim(),
    event_team: state.selectedTeam || "unknown",
    goal_side: state.selectedGoalSide || "unknown",
    compact_response: true,
  };
  const videoId = state.currentVideoId;
  const eventId = event.id;
  const key = segmentKey(event);
  const originalGroup = segmentEvents(event).map((item) => JSON.parse(JSON.stringify(item)));
  state.video.events.forEach((item) => {
    if (segmentKey(item) !== key) return;
    item.review = { ...item.review, status: selected.has(item.label) ? (item.human_added ? "modified" : "accepted") : "deleted" };
  });

  // Navigation and video playback are immediate. Persistence continues in order
  // so rapid annotation cannot reorder database writes.
  applyFilters();
  toast(selectedLabels.length ? "已确认 · 后台保存中" : "已删除 · 后台保存中");
  nextEvent(1, true, true);

  segmentSaveQueue = segmentSaveQueue.catch(() => {}).then(async () => {
    try {
      const result = await api(`/api/events/${eventId}/segment-decision`, { method: "POST", body: JSON.stringify(payload) });
      mergeSavedSegment(result);
      scheduleBootstrapRefresh();
      if (state.currentVideoId === videoId) applyFilters();
    } catch (error) {
      if (state.currentVideoId === videoId && state.video) {
        const untouched = state.video.events.filter((item) => segmentKey(item) !== key);
        state.video.events = [...untouched, ...originalGroup].sort((a, b) => Number(a.time_sec) - Number(b.time_sec) || String(a.label).localeCompare(String(b.label)));
        applyFilters();
      }
      toast(`保存失败，已恢复待审核状态：${error.message}`);
    }
  });
}'''
    return v15._replace_function(text, "saveMultiLabelSegment", "playEventContext", replacement)


hierarchical.HierarchicalReviewStore = CompactSegmentStore
hierarchical.patch_hierarchical_html = patch_html_v20
hierarchical.patch_hierarchical_js = patch_js_v20
hierarchical.HIERARCHICAL_CSS += r'''
/* v20: acknowledge the click immediately while persistence runs in background. */
.toast { transition: opacity .12s ease, transform .12s ease; }
'''


if __name__ == "__main__":
    hierarchical.main()
