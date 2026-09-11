#!/usr/bin/env python3
"""v30 multi-user review plus an explicit second-confirmation workflow."""

from __future__ import annotations

import json
from http.server import ThreadingHTTPServer

import server as base
import server_hierarchical as hierarchical
import server_multiuser_v30 as v30


NEEDS_CONFIRMATION = "needs_confirmation"
base.VALID_STATUSES.add(NEEDS_CONFIRMATION)

_StoreV30 = v30.MultiUserReviewStore
_HandlerV30 = v30.MultiUserReviewHandler
_html_v30 = hierarchical.patch_hierarchical_html
_js_v30 = hierarchical.patch_hierarchical_js


class SecondConfirmationStore(_StoreV30):
    """Persist ambiguity separately from accepted training annotations."""

    def update_segment_labels(self, anchor_id: str, data: dict) -> dict:
        if data.get("review_outcome") != NEEDS_CONFIRMATION:
            return super().update_segment_labels(anchor_id, data)

        anchor = self._event_by_id(anchor_id)
        video_id = str(anchor["video_id"])
        segment_id = str(anchor.get("segment_id") or anchor["id"])
        group = [
            event for event in self.events_for_video(video_id)
            if str(event.get("segment_id") or event["id"]) == segment_id
        ]
        reviewer = str(data.get("reviewer", ""))[:120]
        note = str(data.get("note", ""))[:2000]
        with self.lock, self.connect() as connection:
            for event in group:
                current = connection.execute(
                    "SELECT * FROM reviews WHERE event_id=?", (event["id"],)
                ).fetchone()
                if current is None:
                    raise KeyError(event["id"])
                connection.execute(
                    "INSERT INTO review_history(event_id, revision, state_json, created_at) VALUES (?, ?, ?, ?)",
                    (event["id"], current["revision"], json.dumps(dict(current)), v30.utc_now()),
                )
                connection.execute(
                    """UPDATE reviews SET status=?, note=?, reviewer=?,
                              revision=revision+1, updated_at=? WHERE event_id=?""",
                    (NEEDS_CONFIRMATION, note, reviewer, v30.utc_now(), event["id"]),
                )
        events = self.events_for_video(video_id)
        result = {"video_id": video_id, "segment_id": segment_id, "events": events}
        if data.get("compact_response"):
            result.pop("events")
            result["segment_events"] = [
                event for event in events
                if str(event.get("segment_id") or event["id"]) == segment_id
            ]
        return result

    def bootstrap_for_user(self, user: dict) -> dict:
        result = super().bootstrap_for_user(user)
        allowed = self.allowed_video_ids(str(user["user_id"]))
        placeholders = ",".join("?" for _ in allowed)
        with self.connect() as connection:
            row = connection.execute(
                f"""SELECT COUNT(*) AS count FROM reviews r JOIN events e ON e.id=r.event_id
                    WHERE e.video_id IN ({placeholders}) AND r.status=?""",
                (*sorted(allowed), NEEDS_CONFIRMATION),
            ).fetchone()
        result["status_counts"][NEEDS_CONFIRMATION] = int(row["count"])
        return result

    def export_rows(self, include_unreviewed: bool = False) -> list[dict]:
        # Pending rows are deliberately excluded from training-data exports.
        return [
            row for row in super().export_rows(include_unreviewed)
            if row.get("review_status") != NEEDS_CONFIRMATION
        ]


def patch_html_v31(text: str) -> str:
    text = _html_v30(text).replace(
        "multiuser-v30-video-assigned-20260902",
        "multiuser-v31-second-confirmation-header-20260903",
        1,
    )
    status_anchor = '<option value="all">全部状态</option><option value="unreviewed">未审核</option>'
    if status_anchor not in text:
        raise RuntimeError("v31 status filter anchor missing")
    text = text.replace(
        status_anchor,
        status_anchor + '<option value="needs_confirmation">待二次确认</option>',
        1,
    )
    prediction_anchor = '<small class="prediction-caption">模型预测</small>'
    if prediction_anchor not in text:
        raise RuntimeError("v31 prediction heading anchor missing")
    return text.replace(
        prediction_anchor,
        prediction_anchor + '\n<span id="needsConfirmationBtn" class="prediction-pending-button" role="button" tabindex="0" title="标记为待二次确认（Q）">待二次确认 <kbd>Q</kbd></span>',
        1,
    )


def patch_js_v31(text: str) -> str:
    text = _js_v30(text)
    status_anchor = '  if (group.some((item) => item.review.status === "unreviewed")) return "unreviewed";'
    if status_anchor not in text:
        raise RuntimeError("v31 segment status anchor missing")
    text = text.replace(
        status_anchor,
        status_anchor + '\n  if (group.some((item) => item.review.status === "needs_confirmation")) return "needs_confirmation";',
        1,
    )
    progress_anchor = '  $("#progressText").textContent = `${reviewed} / ${total} 已审核`;'
    if progress_anchor not in text:
        raise RuntimeError("v31 progress anchor missing")
    text = text.replace(
        progress_anchor,
        '  $("#progressText").textContent = `${reviewed} / ${total} 已初审 · ${Number(counts.needs_confirmation || 0)} 待二次确认`;',
        1,
    )
    function_anchor = 'async function saveMultiLabelSegment() {'
    if function_anchor not in text:
        raise RuntimeError("v31 save function anchor missing")
    pending_function = r'''async function saveNeedsConfirmation() {
  const event = currentEvent();
  if (!event) return;
  const videoId = state.currentVideoId, eventId = event.id, key = segmentKey(event);
  const originalGroup = segmentEvents(event).map((item) => JSON.parse(JSON.stringify(item)));
  const payload = {
    review_outcome: "needs_confirmation",
    reviewer: $("#reviewerInput").value.trim(),
    note: $("#note").value.trim(),
    compact_response: true,
    expected_revisions: Object.fromEntries(segmentEvents(event).map((item) => [item.id, Number(item.review.revision || 0)])),
  };
  state.video.events.forEach((item) => {
    if (segmentKey(item) === key) item.review = { ...item.review, status: "needs_confirmation" };
  });
  applyFilters();
  toast("已标记待二次确认 · 后台保存中");
  nextEvent(1, true, true);
  segmentSaveQueue = segmentSaveQueue.catch(() => {}).then(async () => {
    try {
      const result = await api(`/api/events/${eventId}/segment-decision`, { method: "POST", body: JSON.stringify(payload) });
      mergeSavedSegment(result); scheduleBootstrapRefresh();
      if (state.currentVideoId === videoId) applyFilters();
    } catch (error) {
      if (state.currentVideoId === videoId && state.video) {
        const untouched = state.video.events.filter((item) => segmentKey(item) !== key);
        state.video.events = [...untouched, ...originalGroup].sort((a, b) => Number(a.time_sec) - Number(b.time_sec) || String(a.label).localeCompare(String(b.label)));
        applyFilters();
      }
      toast(`保存失败，已恢复原状态：${error.message}`);
    }
  });
}

'''
    text = text.replace(function_anchor, pending_function + function_anchor, 1)
    bind_anchor = '  $("#acceptBtn").onclick = saveMultiLabelSegment;'
    if bind_anchor not in text:
        raise RuntimeError("v31 button binding anchor missing")
    text = text.replace(
        bind_anchor,
        bind_anchor + '\n  $("#needsConfirmationBtn").onclick = saveNeedsConfirmation;',
        1,
    )
    key_anchor = '    else if (["x", "X", "Delete", "Backspace"].includes(event.key)) { state.selectedSegmentLabels.clear(); renderSegmentLabelEditor(); renderTeamAttribution(); saveMultiLabelSegment(); }'
    if key_anchor not in text:
        raise RuntimeError("v31 keyboard anchor missing")
    return text.replace(
        key_anchor,
        key_anchor + '\n    else if (["q", "Q"].includes(event.key)) saveNeedsConfirmation();',
        1,
    )


hierarchical.patch_hierarchical_html = patch_html_v31
hierarchical.patch_hierarchical_js = patch_js_v31
hierarchical.HIERARCHICAL_CSS += r'''
.prediction-pending-button {
  display:inline-flex; align-items:center; gap:5px; height:26px; padding:0 8px;
  border:1px solid #7b6a32; border-radius:7px; color:#d7bd67; background:#171b20;
  font-size:10px; font-weight:700; letter-spacing:0; cursor:pointer; white-space:nowrap;
}
.prediction-pending-button:hover { color:#f1d77b; border-color:#b99c42; background:#242217; }
.prediction-pending-button kbd { color:#a99762; font-size:9px; }
.queue-item.needs_confirmation { box-shadow:inset 3px 0 #e7c65a; }
.queue-item.needs_confirmation::after { content:"待二次确认"; color:#e7c65a; font-size:9px; }
'''


class SecondConfirmationHandler(_HandlerV30):
    pass


def main() -> None:
    args = v30.parse_args()
    store = SecondConfirmationStore(args.manifest, args.db, args.access_config)
    server = ThreadingHTTPServer((args.host, args.port), SecondConfirmationHandler)
    server.store = store  # type: ignore[attr-defined]
    print(f"Multi-user football review UI v31: http://{args.host}:{args.port}", flush=True)
    print(f"Review database: {store.db_path}", flush=True)
    print(f"Assigned users: {len(store.users_by_id)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
