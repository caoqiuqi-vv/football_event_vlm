#!/usr/bin/env python3
"""Team-aware review UI with event actor and target-goal attribution."""

from __future__ import annotations

import json
import sqlite3

import server_hierarchical as hierarchical
import server_hierarchical_v7  # noqa: F401


VALID_EVENT_TEAMS = {"teamA", "teamB", "unknown"}
VALID_GOAL_SIDES = {"left", "right", "unknown", "not_applicable"}
_BaseHierarchicalReviewStore = hierarchical.HierarchicalReviewStore


class TeamAwareReviewStore(_BaseHierarchicalReviewStore):
    """Persist team attribution without invalidating an existing review DB."""

    def _initialize(self) -> None:
        super()._initialize()
        with self.connect() as connection:
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(reviews)")}
            if "event_team" not in columns:
                connection.execute("ALTER TABLE reviews ADD COLUMN event_team TEXT")
            if "goal_side" not in columns:
                connection.execute("ALTER TABLE reviews ADD COLUMN goal_side TEXT")

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict:
        payload = _BaseHierarchicalReviewStore._event_from_row(row)
        keys = set(row.keys())
        payload["review"]["event_team"] = row["event_team"] if "event_team" in keys else None
        payload["review"]["goal_side"] = row["goal_side"] if "goal_side" in keys else None
        return payload

    @staticmethod
    def _select_sql(where: str) -> str:
        return f"""SELECT e.*, r.status, r.corrected_label, r.corrected_time_sec,
                          r.secondary_labels_json, r.event_team, r.goal_side,
                          r.note, r.reviewer, r.revision, r.updated_at
                   FROM events e JOIN reviews r ON r.event_id=e.id {where}"""

    def events_for_video(self, video_id: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                self._select_sql("WHERE e.video_id=? ORDER BY e.source_time_sec, e.source_label"),
                (video_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def _event_by_id(self, event_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(self._select_sql("WHERE e.id=?"), (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        return self._event_from_row(row)

    def bootstrap(self) -> dict:
        result = super().bootstrap()
        result.update({
            "event_teams": ["teamA", "teamB", "unknown"],
            "goal_sides": ["left", "right", "unknown", "not_applicable"],
            "team_attribution_schema_version": 1,
            "event_team_semantics": {
                "shot": "shooter_team",
                "save": "goalkeeper_team",
                "set_piece": "kicking_team",
            },
        })
        return result

    def update_review(self, event_id: str, data: dict) -> dict:
        event_team = data.get("event_team")
        goal_side = data.get("goal_side")
        if event_team is not None and event_team not in VALID_EVENT_TEAMS:
            raise ValueError(f"Invalid event team: {event_team}")
        if goal_side is not None and goal_side not in VALID_GOAL_SIDES:
            raise ValueError(f"Invalid goal side: {goal_side}")

        result = super().update_review(event_id, data)
        with self.lock, self.connect() as connection:
            current = connection.execute(
                "SELECT event_team, goal_side FROM reviews WHERE event_id=?", (event_id,)
            ).fetchone()
            if current is None:
                raise KeyError(event_id)
            if data.get("status") == "deleted":
                event_team, goal_side = None, None
            else:
                if event_team is None:
                    event_team = current["event_team"]
                if goal_side is None:
                    goal_side = current["goal_side"]
            connection.execute(
                "UPDATE reviews SET event_team=?, goal_side=? WHERE event_id=?",
                (event_team, goal_side, event_id),
            )
        return self._event_by_id(event_id)

    def undo(self, event_id: str) -> dict:
        with self.connect() as connection:
            previous = connection.execute(
                "SELECT state_json FROM review_history WHERE event_id=? ORDER BY id DESC LIMIT 1",
                (event_id,),
            ).fetchone()
        if previous is None:
            raise ValueError("No previous review state")
        state = json.loads(previous["state_json"])
        super().undo(event_id)
        with self.lock, self.connect() as connection:
            connection.execute(
                "UPDATE reviews SET event_team=?, goal_side=? WHERE event_id=?",
                (state.get("event_team"), state.get("goal_side"), event_id),
            )
        return self._event_by_id(event_id)

    def export_rows(self, include_unreviewed: bool = False) -> list[dict]:
        rows = super().export_rows(include_unreviewed)
        events = {
            event["id"]: event
            for video_id in self.videos
            for event in self.events_for_video(video_id)
        }
        for row in rows:
            event = events[row["event_id"]]
            evidence = event.get("team_evidence", {})
            review = event["review"]
            row["event_team"] = review.get("event_team") or "unknown"
            row["goal_side"] = review.get("goal_side") or "unknown"
            row["suggested_event_team"] = evidence.get("suggested_event_team", "unknown")
            row["event_team_confidence"] = evidence.get("event_team_confidence", 0)
            row["suggested_goal_side"] = evidence.get("suggested_goal_side", "unknown")
            row["goal_side_confidence"] = evidence.get("goal_side_confidence", 0)
        return rows


_html_v7 = hierarchical.patch_hierarchical_html
_js_v7 = hierarchical.patch_hierarchical_js


def patch_html_v8(text: str) -> str:
    text = _html_v7(text).replace(
        "hierarchical-v7-fullheight-20260813", "hierarchical-v8-team-aware-20260818"
    )
    anchor = '            <button id="toggleAdvanced" class="advanced-toggle" type="button">'
    panel = '''            <section id="teamAttributionPanel" class="team-attribution-panel">
              <div class="team-heading">
                <strong id="teamRoleTitle">事件执行方</strong>
                <span id="teamSuggestionState">暂无自动建议</span>
              </div>
              <div id="teamButtons" class="team-buttons"></div>
              <div id="goalSideBlock" class="goal-side-block">
                <span>目标球门</span>
                <div id="goalSideButtons" class="goal-side-buttons"></div>
              </div>
              <div id="teamEvidenceText" class="team-evidence-text">未接入该时段的分队证据</div>
            </section>
'''
    if anchor not in text:
        raise RuntimeError("team UI anchor missing")
    return text.replace(anchor, panel + anchor, 1)


def patch_js_v8(text: str) -> str:
    text = _js_v7(text)
    text = text.replace(
        "  filtered: [], selectedLabel: null, secondaryLabels: new Set(), correctedTime: null,",
        "  filtered: [], selectedLabel: null, secondaryLabels: new Set(), selectedTeam: 'unknown', selectedGoalSide: 'unknown', correctedTime: null,",
        1,
    )
    anchor = "function playEventContext(event) {"
    addition = r'''function teamPalette(event, team) {
  const palette = event.team_evidence?.team_palette || state.video?.team_palette || {};
  return palette[team] || { hex: team === "teamA" ? "#d5b64c" : "#477bb5", display_name: team === "teamA" ? "队伍 A" : "队伍 B" };
}

function renderTeamAttribution() {
  const event = currentEvent();
  if (!event) return;
  const evidence = event.team_evidence || {};
  const review = event.review || {};
  state.selectedTeam = review.event_team || evidence.suggested_event_team || "unknown";
  state.selectedGoalSide = review.goal_side || evidence.suggested_goal_side || (event.label === "set_piece" ? "not_applicable" : "unknown");
  const roleNames = { shot: "射门方", save: "扑救方", set_piece: "定位球执行方" };
  $("#teamRoleTitle").textContent = roleNames[state.selectedLabel] || "事件执行方";
  const teamConfidence = Number(evidence.event_team_confidence || 0);
  const suggestion = evidence.suggested_event_team;
  $("#teamSuggestionState").textContent = suggestion && suggestion !== "unknown"
    ? `自动建议 ${Math.round(teamConfidence * 100)}%`
    : "待人工判断";
  $("#teamSuggestionState").classList.toggle("confident", teamConfidence >= 0.7);

  $("#teamButtons").innerHTML = ["teamA", "teamB", "unknown"].map((team) => {
    const palette = teamPalette(event, team);
    const swatch = team === "unknown" ? "" : `<i style="background:${palette.hex || '#777'}"></i>`;
    const name = team === "unknown" ? "无法判断" : (palette.display_name || (team === "teamA" ? "队伍 A" : "队伍 B"));
    return `<button type="button" data-team="${team}" class="${state.selectedTeam === team ? 'selected' : ''}">${swatch}<span>${name}</span></button>`;
  }).join("");
  $("#teamButtons").querySelectorAll("button").forEach((button) => {
    button.onclick = () => { state.selectedTeam = button.dataset.team; $("#teamButtons").querySelectorAll("button").forEach((item) => item.classList.toggle("selected", item.dataset.team === state.selectedTeam)); };
  });

  const showGoal = state.selectedLabel === "shot" || state.selectedLabel === "save";
  $("#goalSideBlock").classList.toggle("hidden", !showGoal);
  if (showGoal) {
    const sides = [["left", "← 左侧球门"], ["right", "右侧球门 →"], ["unknown", "画面不明确"]];
    $("#goalSideButtons").innerHTML = sides.map(([side, name]) =>
      `<button type="button" data-side="${side}" class="${state.selectedGoalSide === side ? 'selected' : ''}">${name}</button>`
    ).join("");
    $("#goalSideButtons").querySelectorAll("button").forEach((button) => {
      button.onclick = () => { state.selectedGoalSide = button.dataset.side; $("#goalSideButtons").querySelectorAll("button").forEach((item) => item.classList.toggle("selected", item.dataset.side === state.selectedGoalSide)); };
    });
  } else {
    state.selectedGoalSide = "not_applicable";
  }
  const parts = [];
  if (evidence.coverage_status) parts.push(evidence.coverage_status === "covered" ? "该时段已有分队结果" : "该时段分队未覆盖");
  if (evidence.possession_team && evidence.possession_team !== "unknown") parts.push(`触球/控球：${evidence.possession_team}`);
  if (evidence.visible_goal_sides?.length) parts.push(`可见球门：${evidence.visible_goal_sides.join("、")}`);
  if (evidence.reason) parts.push(evidence.reason);
  $("#teamEvidenceText").textContent = parts.join(" · ") || "未接入该时段的分队证据，保留人工选择";
}

'''
    if anchor not in text:
        raise RuntimeError("team JS function anchor missing")
    text = text.replace(anchor, addition + anchor, 1)
    text = text.replace(
        "  renderClassButtons();\n  const status = event.review.status;",
        "  renderClassButtons();\n  renderTeamAttribution();\n  const status = event.review.status;",
        1,
    )
    text = text.replace(
        "  const payload = { status, reviewer: $(\"#reviewerInput\").value.trim(), note: $(\"#note\").value.trim(), corrected_label: state.selectedLabel, secondary_labels: [...state.secondaryLabels], corrected_time_sec: Number($(\"#correctedTime\").value) };",
        "  const payload = { status, reviewer: $(\"#reviewerInput\").value.trim(), note: $(\"#note\").value.trim(), corrected_label: state.selectedLabel, secondary_labels: [...state.secondaryLabels], corrected_time_sec: Number($(\"#correctedTime\").value), event_team: state.selectedTeam || 'unknown', goal_side: state.selectedGoalSide || 'unknown' };",
        1,
    )
    # A primary-label change can change the actor semantics and goal-side visibility.
    text = text.replace(
        "      renderClassButtons();\n    };\n  });\n  const current = currentEvent();",
        "      renderClassButtons();\n      renderTeamAttribution();\n    };\n  });\n  const current = currentEvent();",
        1,
    )
    return text


hierarchical.HierarchicalReviewStore = TeamAwareReviewStore
hierarchical.patch_hierarchical_html = patch_html_v8
hierarchical.patch_hierarchical_js = patch_js_v8
hierarchical.HIERARCHICAL_CSS += r'''
.team-attribution-panel { margin: 7px 0; padding: 8px; border: 1px solid #314354; border-radius: 8px; background: #101923; }
.team-heading { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.team-heading strong { color: #eef3f8; font-size: 12px; }
.team-heading span { color: #91a0b1; font-size: 10px; }
.team-heading span.confident { color: var(--green); font-weight: 750; }
.team-buttons, .goal-side-buttons { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; }
.team-buttons button, .goal-side-buttons button { min-height: 34px; display: flex; align-items: center; justify-content: center; gap: 6px; color: #cbd5df; background: #1b2632; border: 1px solid #3a4a5c; border-radius: 7px; cursor: pointer; font-weight: 700; }
.team-buttons button i { width: 14px; height: 14px; border: 2px solid #dce5ee; border-radius: 50%; box-shadow: 0 0 0 1px #0008; }
.team-buttons button.selected, .goal-side-buttons button.selected { color: #101710; background: var(--accent); border-color: var(--accent); }
.goal-side-block { display: grid; grid-template-columns: 75px 1fr; align-items: center; gap: 7px; margin-top: 6px; }
.goal-side-block > span { color: #8f9dac; font-size: 10px; }
.team-evidence-text { margin-top: 6px; color: #748394; font-size: 9px; line-height: 1.35; }
'''


if __name__ == "__main__":
    hierarchical.main()
