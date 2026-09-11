#!/usr/bin/env python3
"""Hierarchical football review UI with backward-compatible detail labels."""

from __future__ import annotations

import csv
import io
import json
import mimetypes
import sqlite3
import threading
import webbrowser
from http.server import ThreadingHTTPServer

import server as base
import server_patched as patched


PRIMARY_LABELS = ("shot", "save", "set_piece")
SHOT_DETAILS = ("shot_on_target", "goal", "own_goal")
SET_PIECE_TYPES = ("free_kick", "penalty", "corner", "kickoff", "other_set_piece")
SECONDARY_LABELS = set(SHOT_DETAILS + SET_PIECE_TYPES)
LEGACY_TO_PRIMARY = {
    "shot_on_target": "shot", "goal": "shot", "own_goal": "shot",
    "free_kick": "set_piece", "penalty": "set_piece", "corner": "set_piece",
    "kickoff": "set_piece", "other_set_piece": "set_piece",
}


class HierarchicalReviewStore(base.ReviewStore):
    """Add secondary event labels without invalidating an existing SQLite DB."""

    def _initialize(self) -> None:
        super()._initialize()
        with self.connect() as connection:
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(reviews)")}
            if "secondary_labels_json" not in columns:
                connection.execute(
                    "ALTER TABLE reviews ADD COLUMN secondary_labels_json TEXT NOT NULL DEFAULT '[]'"
                )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict:
        payload = json.loads(row["payload_json"])
        corrected = row["corrected_label"]
        try:
            details = json.loads(row["secondary_labels_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            details = []
        if corrected in LEGACY_TO_PRIMARY:
            details = list(dict.fromkeys([*details, corrected]))
            corrected = LEGACY_TO_PRIMARY[corrected]
        payload["review"] = {
            "status": row["status"], "corrected_label": corrected,
            "secondary_labels": details,
            "corrected_time_sec": row["corrected_time_sec"], "note": row["note"],
            "reviewer": row["reviewer"], "revision": row["revision"],
            "updated_at": row["updated_at"],
        }
        return payload

    def events_for_video(self, video_id: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT e.*, r.status, r.corrected_label, r.corrected_time_sec,
                          r.secondary_labels_json, r.note, r.reviewer, r.revision, r.updated_at
                   FROM events e JOIN reviews r ON r.event_id=e.id
                   WHERE e.video_id=? ORDER BY e.source_time_sec, e.source_label""",
                (video_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def bootstrap(self) -> dict:
        result = super().bootstrap()
        result.update({
            "primary_labels": list(PRIMARY_LABELS),
            "shot_detail_labels": list(SHOT_DETAILS),
            "set_piece_type_labels": list(SET_PIECE_TYPES),
            "label_schema_version": 2,
        })
        return result

    def _source_primary(self, connection: sqlite3.Connection, event_id: str) -> str:
        row = connection.execute("SELECT source_label FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        return LEGACY_TO_PRIMARY.get(row["source_label"], row["source_label"])

    def update_review(self, event_id: str, data: dict) -> dict:
        status = str(data.get("status", "")).strip()
        if status not in base.VALID_STATUSES - {"unreviewed"}:
            raise ValueError(f"Invalid review status: {status}")
        corrected_label = data.get("corrected_label")
        corrected_time = data.get("corrected_time_sec")
        details = list(dict.fromkeys(str(item) for item in data.get("secondary_labels", [])))
        if corrected_time is not None:
            corrected_time = max(0.0, float(corrected_time))
        invalid = set(details) - SECONDARY_LABELS
        if invalid:
            raise ValueError(f"Invalid secondary labels: {sorted(invalid)}")

        with self.lock, self.connect() as connection:
            current = connection.execute("SELECT * FROM reviews WHERE event_id=?", (event_id,)).fetchone()
            if current is None:
                raise KeyError(event_id)
            source_primary = self._source_primary(connection, event_id)
            primary = corrected_label or source_primary
            if primary not in PRIMARY_LABELS:
                raise ValueError(f"Invalid primary label: {primary}")
            if any(item in SHOT_DETAILS for item in details) and primary != "shot":
                raise ValueError("射门详情只能用于射门事件")
            set_piece_types = [item for item in details if item in SET_PIECE_TYPES]
            if set_piece_types and primary != "set_piece":
                raise ValueError("定位球类型只能用于定位球事件")
            if len(set_piece_types) > 1:
                raise ValueError("定位球事件类型只能选择一个")
            if status != "deleted" and primary == "set_piece" and not set_piece_types:
                raise ValueError("确认定位球前，必须选择任意球、点球、角球、中圈开球或其他")
            if status == "deleted":
                corrected_label, corrected_time, details = None, None, []
            else:
                corrected_label = None if primary == source_primary else primary

            connection.execute(
                "INSERT INTO review_history(event_id, revision, state_json, created_at) VALUES (?, ?, ?, ?)",
                (event_id, current["revision"], json.dumps(dict(current)), base.utc_now()),
            )
            connection.execute(
                """UPDATE reviews SET status=?, corrected_label=?, corrected_time_sec=?,
                          secondary_labels_json=?, note=?, reviewer=?, revision=revision+1, updated_at=?
                   WHERE event_id=?""",
                (status, corrected_label, corrected_time, json.dumps(details),
                 str(data.get("note", ""))[:2000], str(data.get("reviewer", ""))[:120],
                 base.utc_now(), event_id),
            )
            row = connection.execute(
                """SELECT e.*, r.status, r.corrected_label, r.corrected_time_sec,
                          r.secondary_labels_json, r.note, r.reviewer, r.revision, r.updated_at
                   FROM events e JOIN reviews r ON r.event_id=e.id WHERE e.id=?""",
                (event_id,),
            ).fetchone()
        return self._event_from_row(row)

    def undo(self, event_id: str) -> dict:
        with self.lock, self.connect() as connection:
            previous = connection.execute(
                "SELECT * FROM review_history WHERE event_id=? ORDER BY id DESC LIMIT 1", (event_id,)
            ).fetchone()
            if previous is None:
                raise ValueError("No previous review state")
            state = json.loads(previous["state_json"])
            connection.execute(
                """UPDATE reviews SET status=?, corrected_label=?, corrected_time_sec=?,
                          secondary_labels_json=?, note=?, reviewer=?, revision=?, updated_at=?
                   WHERE event_id=?""",
                (state["status"], state["corrected_label"], state["corrected_time_sec"],
                 state.get("secondary_labels_json", "[]"), state["note"], state["reviewer"],
                 state["revision"], state["updated_at"], event_id),
            )
            connection.execute("DELETE FROM review_history WHERE id=?", (previous["id"],))
            row = connection.execute(
                """SELECT e.*, r.status, r.corrected_label, r.corrected_time_sec,
                          r.secondary_labels_json, r.note, r.reviewer, r.revision, r.updated_at
                   FROM events e JOIN reviews r ON r.event_id=e.id WHERE e.id=?""",
                (event_id,),
            ).fetchone()
        return self._event_from_row(row)

    def export_rows(self, include_unreviewed: bool = False) -> list[dict]:
        rows = []
        for video_id in self.videos:
            for event in self.events_for_video(video_id):
                review = event["review"]
                if review["status"] == "deleted" or (
                    review["status"] == "unreviewed" and not include_unreviewed
                ):
                    continue
                primary = review["corrected_label"] or LEGACY_TO_PRIMARY.get(event["label"], event["label"])
                output_time = review["corrected_time_sec"] if review["corrected_time_sec"] is not None else event["time_sec"]
                rows.append({
                    "video_id": video_id, "label": primary,
                    "secondary_labels": json.dumps(review["secondary_labels"], ensure_ascii=False),
                    "time_sec": output_time, "score": event["score"],
                    "review_status": review["status"], "source_label": event["label"],
                    "source_time_sec": event["time_sec"],
                    "frame_detection_score": event.get("frame_detection_scores", {}).get(event["label"], 0),
                    "evaluation_status": event.get("evaluation_status", ""),
                    "match_tolerance_sec": event.get("match_tolerance_sec", ""),
                    "matching_gt_times": json.dumps(event.get("matching_gt_times", [])),
                    "reviewer": review["reviewer"], "note": review["note"], "event_id": event["id"],
                })
        return sorted(rows, key=lambda row: (row["video_id"], row["time_sec"], row["label"]))


def replace_required(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"hierarchical UI patch anchor not found: {old[:100]!r}")
    return text.replace(old, new, 1)


def patch_hierarchical_js(text: str) -> str:
    text = patched.patch_javascript(text)
    text = replace_required(
        text,
        'const LABEL_NAMES = { shot: "射门", save: "扑救", set_piece: "定位球", free_kick: "任意球", penalty: "点球", corner: "角球", shot_on_target: "射正" };',
        'const LABEL_NAMES = { shot: "射门", save: "扑救", set_piece: "定位球", free_kick: "任意球", penalty: "点球", corner: "角球", kickoff: "中圈开球", other_set_piece: "其他定位球", shot_on_target: "射正", goal: "进球", own_goal: "乌龙球" };',
    )
    text = replace_required(
        text,
        "  filtered: [], selectedLabel: null, correctedTime: null,",
        "  filtered: [], selectedLabel: null, secondaryLabels: new Set(), correctedTime: null,",
    )
    text = replace_required(
        text,
        '''  state.selectedLabel = event.review.corrected_label || (state.bootstrap.review_labels.includes(sourceLabel) ? sourceLabel : null);
  state.correctedTime = event.review.corrected_time_sec ?? event.time_sec;''',
        '''  const legacyPrimary = ["free_kick", "penalty", "corner", "kickoff", "other_set_piece"].includes(sourceLabel) ? "set_piece" : (["shot_on_target", "goal", "own_goal"].includes(sourceLabel) ? "shot" : sourceLabel);
  state.selectedLabel = event.review.corrected_label || legacyPrimary;
  state.secondaryLabels = new Set(event.review.secondary_labels || []);
  state.correctedTime = event.review.corrected_time_sec ?? event.time_sec;''',
    )
    old_buttons = '''function renderClassButtons() {
  $("#classButtons").innerHTML = state.bootstrap.review_labels.map((label, index) =>
    `<button data-label="${label}" class="${state.selectedLabel === label ? "selected" : ""}">${index + 1} ${LABEL_NAMES[label]}</button>`
  ).join("");
  $("#classButtons").querySelectorAll("button").forEach((button) => {
    button.onclick = () => { state.selectedLabel = button.dataset.label; renderClassButtons(); };
  });
}'''
    new_buttons = '''function renderClassButtons() {
  const primaryLabels = state.bootstrap.primary_labels || ["shot", "save", "set_piece"];
  $("#classButtons").innerHTML = primaryLabels.map((label, index) =>
    `<button data-label="${label}" class="${state.selectedLabel === label ? "selected" : ""}">${index + 1} ${LABEL_NAMES[label]}</button>`
  ).join("");
  $("#classButtons").querySelectorAll("button").forEach((button) => {
    button.onclick = () => {
      state.selectedLabel = button.dataset.label;
      if (state.selectedLabel !== "shot") (state.bootstrap.shot_detail_labels || []).forEach((x) => state.secondaryLabels.delete(x));
      if (state.selectedLabel !== "set_piece") (state.bootstrap.set_piece_type_labels || []).forEach((x) => state.secondaryLabels.delete(x));
      renderClassButtons();
    };
  });
  const shotPanel = $("#shotDetailPanel");
  shotPanel.classList.toggle("hidden", state.selectedLabel !== "shot");
  $("#shotDetailButtons").innerHTML = (state.bootstrap.shot_detail_labels || []).map((label) =>
    `<button data-detail="${label}" class="${state.secondaryLabels.has(label) ? "selected" : ""}">${LABEL_NAMES[label]}</button>`
  ).join("");
  $("#shotDetailButtons").querySelectorAll("button").forEach((button) => {
    button.onclick = () => {
      const label = button.dataset.detail;
      if (state.secondaryLabels.has(label)) {
        state.secondaryLabels.delete(label);
        if (label === "shot_on_target") { state.secondaryLabels.delete("goal"); state.secondaryLabels.delete("own_goal"); }
        if (label === "goal") state.secondaryLabels.delete("own_goal");
      } else {
        state.secondaryLabels.add(label);
        if (label === "goal") state.secondaryLabels.add("shot_on_target");
        if (label === "own_goal") { state.secondaryLabels.add("goal"); state.secondaryLabels.add("shot_on_target"); }
      }
      renderClassButtons();
    };
  });
  const setPiecePanel = $("#setPieceTypePanel");
  setPiecePanel.classList.toggle("hidden", state.selectedLabel !== "set_piece");
  $("#setPieceTypeButtons").innerHTML = (state.bootstrap.set_piece_type_labels || []).map((label) =>
    `<button data-detail="${label}" class="${state.secondaryLabels.has(label) ? "selected" : ""}">${LABEL_NAMES[label]}</button>`
  ).join("");
  $("#setPieceTypeButtons").querySelectorAll("button").forEach((button) => {
    button.onclick = () => {
      (state.bootstrap.set_piece_type_labels || []).forEach((x) => state.secondaryLabels.delete(x));
      state.secondaryLabels.add(button.dataset.detail);
      renderClassButtons();
    };
  });
}'''
    text = replace_required(text, old_buttons, new_buttons)
    text = replace_required(
        text,
        '''  const payload = { status, reviewer: $("#reviewerInput").value.trim(), note: $("#note").value.trim() };
  if (modified) {
    payload.corrected_label = state.selectedLabel;
    payload.corrected_time_sec = Number($("#correctedTime").value);
  }''',
        '''  if (status !== "deleted" && state.selectedLabel === "set_piece" && !(state.bootstrap.set_piece_type_labels || []).some((label) => state.secondaryLabels.has(label))) {
    toast("请先选择定位球类型");
    $("#setPieceTypePanel").scrollIntoView({ block: "nearest" });
    return;
  }
  const payload = { status, reviewer: $("#reviewerInput").value.trim(), note: $("#note").value.trim(), corrected_label: state.selectedLabel, secondary_labels: [...state.secondaryLabels], corrected_time_sec: Number($("#correctedTime").value) };''',
    )
    return text


def patch_hierarchical_html(text: str) -> str:
    text = patched.patch_html(text)
    return replace_required(
        text,
        '''            <span>修正为</span>
            <div id="classButtons" class="class-buttons"></div>''',
        '''            <span>主事件（可修正）</span>
            <div id="classButtons" class="class-buttons primary-label-buttons"></div>
            <section id="shotDetailPanel" class="conditional-label-panel hidden">
              <strong>射门结果 <small>可多选；进球自动包含射正</small></strong>
              <div id="shotDetailButtons" class="detail-buttons shot-details"></div>
            </section>
            <section id="setPieceTypePanel" class="conditional-label-panel required hidden">
              <strong>定位球类型 <small>必选一项</small></strong>
              <div id="setPieceTypeButtons" class="detail-buttons set-piece-types"></div>
            </section>''',
    )


HIERARCHICAL_CSS = """
.conditional-label-panel { margin: 9px 0; padding: 10px; border: 1px solid #344150; border-radius: 8px; background: #131b24; }
.conditional-label-panel.required { border-color: #7a642c; background: #231e12; }
.conditional-label-panel strong { display: flex; justify-content: space-between; color: #e6edf5; font-size: 12px; }
.conditional-label-panel small { color: #8793a3; font-weight: 400; }
.detail-buttons { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 8px; }
.set-piece-types { grid-template-columns: repeat(3, 1fr); }
.detail-buttons button { min-height: 38px; border: 1px solid #415064; color: #d4dde7; background: #202a36; border-radius: 7px; cursor: pointer; font-weight: 700; }
.detail-buttons button.selected { color: #101710; background: var(--accent); border-color: var(--accent); box-shadow: 0 0 0 2px #d7ff3822; }
.primary-label-buttons button { height: 38px; font-size: 13px; }
"""


class HierarchicalReviewHandler(patched.PatchedReviewHandler):
    server_version = "FootballEventReview/2.0"

    def send_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (base.STATIC_ROOT / relative).resolve()
        if base.STATIC_ROOT not in target.parents and target != base.STATIC_ROOT:
            return self.send_json({"error": "Invalid path"}, 400)
        if not target.exists() or not target.is_file():
            return self.send_json({"error": "Not found"}, 404)
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.name in {"app.js", "index.html", "styles.css"}:
            text = target.read_text(encoding="utf-8")
            if target.name == "app.js":
                text = patch_hierarchical_js(text)
            elif target.name == "index.html":
                text = patch_hierarchical_html(text)
            else:
                text += patched.CSS_PATCH + HIERARCHICAL_CSS
            return self.send_bytes(text.encode("utf-8"), f"{content_type}; charset=utf-8")
        self.send_bytes(target.read_bytes(), content_type)


def main() -> None:
    args = base.parse_args()
    store = HierarchicalReviewStore(args.manifest, args.db)
    server = ThreadingHTTPServer((args.host, args.port), HierarchicalReviewHandler)
    server.store = store  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}"
    print(f"Football event review UI (hierarchical): {url}", flush=True)
    print(f"Review database: {store.db_path}", flush=True)
    if args.open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
