#!/usr/bin/env python3
"""v35 plus versioned team-colour, pitch-side, and event-team review.

The source manifest and historical event reviews remain immutable inputs.  Human
team calibration is stored in separate versioned SQLite tables, while event-level
team/field-side decisions extend the existing review history.
"""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
import ssl
from http import HTTPStatus
from pathlib import Path
from urllib.parse import unquote, urlparse

import server as base
import server_hierarchical as hierarchical
import server_multiuser_v30 as multiuser
import server_multiuser_v35 as v35


VALID_TEAMS = {"teamA", "teamB", "unknown"}
VALID_FIELD_SIDES = {"left", "middle", "right", "unknown", "not_applicable"}
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

_BaseStore = v35.v34.v33.v32.ProxyReviewStore
_BaseHandler = v35.v34.v33.v32.ProxyReviewHandler
_html_v35 = hierarchical.patch_hierarchical_html
_js_v35 = hierarchical.patch_hierarchical_js


def _default_sample(video: dict) -> tuple[float, float]:
    duration = max(0.0, float(video.get("duration_sec", 0.0) or 0.0))
    events = sorted(
        float(item.get("time_sec", 0.0) or 0.0)
        for item in video.get("events", [])
        if 30.0 <= float(item.get("time_sec", 0.0) or 0.0) <= max(30.0, duration - 15.0)
    )
    anchor = events[min(len(events) // 4, len(events) - 1)] if events else min(90.0, duration * 0.2)
    start = max(0.0, min(max(0.0, duration - 30.0), anchor - 10.0))
    return round(start, 3), round(min(duration, start + 30.0), 3)


def _manifest_palette(video: dict, team: str, fallback: str) -> dict:
    item = (video.get("team_palette") or {}).get(team) or {}
    return {
        "hex": str(item.get("hex") or fallback),
        "display_name": str(item.get("display_name") or ("队伍 A" if team == "teamA" else "队伍 B")),
        "color_name": str(item.get("color_name") or "待确认"),
    }


class TeamCalibrationStore(_BaseStore):
    """Store video-level team identity independently from model suggestions."""

    def _initialize(self) -> None:
        super()._initialize()
        with self.connect() as connection:
            review_columns = {row["name"] for row in connection.execute("PRAGMA table_info(reviews)")}
            if "field_side" not in review_columns:
                connection.execute("ALTER TABLE reviews ADD COLUMN field_side TEXT")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS video_team_profiles (
                    video_id TEXT PRIMARY KEY,
                    team_a_hex TEXT NOT NULL,
                    team_a_name TEXT NOT NULL,
                    team_b_hex TEXT NOT NULL,
                    team_b_name TEXT NOT NULL,
                    period_split_sec REAL,
                    first_period_left_team TEXT NOT NULL DEFAULT 'unknown',
                    second_period_left_team TEXT NOT NULL DEFAULT 'unknown',
                    sample_start_sec REAL NOT NULL DEFAULT 0,
                    sample_end_sec REAL NOT NULL DEFAULT 30,
                    status TEXT NOT NULL DEFAULT 'unconfirmed',
                    reviewer TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL DEFAULT 0,
                    source_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS video_team_profile_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS team_profile_history_video_idx
                    ON video_team_profile_history(video_id, id);
                """
            )
            for video_id, video in self.videos.items():
                team_a = _manifest_palette(video, "teamA", "#d5b64c")
                team_b = _manifest_palette(video, "teamB", "#477bb5")
                calibration = video.get("team_calibration") or {}
                default_start, default_end = _default_sample(video)
                source = {
                    "team_palette": video.get("team_palette") or {},
                    "team_cluster": video.get("team_cluster") or {},
                    "team_calibration": calibration,
                    "policy": "team_cluster_upper_body_colour_is_suggestion_only",
                }
                connection.execute(
                    """INSERT OR IGNORE INTO video_team_profiles
                       (video_id, team_a_hex, team_a_name, team_b_hex, team_b_name,
                        period_split_sec, first_period_left_team, second_period_left_team,
                        sample_start_sec, sample_end_sec, status, source_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unconfirmed', ?)""",
                    (
                        video_id, team_a["hex"], team_a["display_name"],
                        team_b["hex"], team_b["display_name"],
                        calibration.get("period_split_sec"),
                        calibration.get("first_period_left_team", "unknown"),
                        calibration.get("second_period_left_team", "unknown"),
                        float(calibration.get("sample_start_sec", default_start)),
                        float(calibration.get("sample_end_sec", default_end)),
                        json.dumps(source, ensure_ascii=False),
                    ),
                )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict:
        payload = _BaseStore._event_from_row(row)
        payload["review"]["field_side"] = row["field_side"] if "field_side" in row.keys() else None
        return payload

    @staticmethod
    def _event_select(where: str) -> str:
        return f"""SELECT e.*, r.status, r.corrected_label, r.corrected_time_sec,
                          r.secondary_labels_json, r.event_team, r.goal_side, r.field_side,
                          r.note, r.reviewer, r.revision, r.updated_at
                   FROM events e JOIN reviews r ON r.event_id=e.id {where}"""

    def events_for_video(self, video_id: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                self._event_select("WHERE e.video_id=? ORDER BY e.source_time_sec, e.source_label"),
                (video_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def _event_by_id(self, event_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(self._event_select("WHERE e.id=?"), (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        return self._event_from_row(row)

    def update_review(self, event_id: str, data: dict) -> dict:
        field_side = data.get("field_side")
        if field_side is not None and field_side not in VALID_FIELD_SIDES:
            raise ValueError(f"Invalid field side: {field_side}")
        result = super().update_review(event_id, data)
        with self.lock, self.connect() as connection:
            current = connection.execute(
                "SELECT field_side FROM reviews WHERE event_id=?", (event_id,)
            ).fetchone()
            if current is None:
                raise KeyError(event_id)
            value = None if data.get("status") == "deleted" else (
                current["field_side"] if field_side is None else field_side
            )
            connection.execute("UPDATE reviews SET field_side=? WHERE event_id=?", (value, event_id))
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
                "UPDATE reviews SET field_side=? WHERE event_id=?",
                (state.get("field_side"), event_id),
            )
        return self._event_by_id(event_id)

    def update_segment_labels(self, anchor_id: str, data: dict) -> dict:
        result = super().update_segment_labels(anchor_id, data)
        if data.get("review_outcome") == "needs_confirmation":
            return result
        video_id = str(result["video_id"])
        segment_id = str(result["segment_id"])
        selected = set(str(item) for item in data.get("selected_labels", []))
        attributions = data.get("attribution_by_label", {}) or {}
        with self.lock, self.connect() as connection:
            rows = connection.execute(
                "SELECT id, source_label FROM events WHERE video_id=?", (video_id,)
            ).fetchall()
            events = self.events_for_video(video_id)
            event_by_id = {item["id"]: item for item in events}
            for row in rows:
                item = event_by_id.get(row["id"])
                if item is None or str(item.get("segment_id") or item["id"]) != segment_id:
                    continue
                label = str(item["label"])
                value = None if label not in selected else str(
                    (attributions.get(label) or {}).get("field_side", "unknown")
                )
                if value is not None and value not in VALID_FIELD_SIDES:
                    raise ValueError(f"Invalid field side: {value}")
                connection.execute(
                    "UPDATE reviews SET field_side=? WHERE event_id=?", (value, item["id"])
                )
        result["events"] = self.events_for_video(video_id)
        if data.get("compact_response"):
            result.pop("events", None)
            result["segment_events"] = [
                item for item in self.events_for_video(video_id)
                if str(item.get("segment_id") or item["id"]) == segment_id
            ]
        return result

    @staticmethod
    def _profile_from_row(row: sqlite3.Row) -> dict:
        source = json.loads(row["source_json"] or "{}")
        return {
            "video_id": row["video_id"],
            "teams": {
                "teamA": {"hex": row["team_a_hex"], "display_name": row["team_a_name"]},
                "teamB": {"hex": row["team_b_hex"], "display_name": row["team_b_name"]},
            },
            "period_split_sec": row["period_split_sec"],
            "first_period_left_team": row["first_period_left_team"],
            "second_period_left_team": row["second_period_left_team"],
            "sample_start_sec": row["sample_start_sec"],
            "sample_end_sec": row["sample_end_sec"],
            "status": row["status"],
            "reviewer": row["reviewer"],
            "revision": row["revision"],
            "updated_at": row["updated_at"],
            "source": source,
        }

    def team_profile(self, video_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM video_team_profiles WHERE video_id=?", (video_id,)
            ).fetchone()
        if row is None:
            raise KeyError(video_id)
        return self._profile_from_row(row)

    @staticmethod
    def _clean_team_name(value: object, fallback: str) -> str:
        name = str(value or "").strip()[:60]
        return name or fallback

    def update_team_profile_for_user(self, user: dict, video_id: str, data: dict) -> dict:
        user_id = str(user["user_id"])
        if not self.owns_video(user_id, video_id):
            raise PermissionError(video_id)
        expected_revision = data.get("expected_revision")
        teams = data.get("teams") or {}
        team_a = teams.get("teamA") or {}
        team_b = teams.get("teamB") or {}
        team_a_hex = str(team_a.get("hex") or "").strip().lower()
        team_b_hex = str(team_b.get("hex") or "").strip().lower()
        if not HEX_RE.fullmatch(team_a_hex) or not HEX_RE.fullmatch(team_b_hex):
            raise ValueError("队伍颜色必须是 #RRGGBB")
        first_left = str(data.get("first_period_left_team", "unknown"))
        second_left = str(data.get("second_period_left_team", "unknown"))
        if first_left not in VALID_TEAMS or second_left not in VALID_TEAMS:
            raise ValueError("左右半场队伍必须是 teamA/teamB/unknown")
        split = data.get("period_split_sec")
        split = None if split in {None, ""} else max(0.0, float(split))
        status = str(data.get("status") or "confirmed")
        if status not in {"unconfirmed", "confirmed"}:
            raise ValueError("Invalid team profile status")
        request_id = secrets.token_hex(16)
        with self.lock, self.connect() as connection:
            current = connection.execute(
                "SELECT * FROM video_team_profiles WHERE video_id=?", (video_id,)
            ).fetchone()
            if current is None:
                raise KeyError(video_id)
            if expected_revision is None or int(expected_revision) != int(current["revision"]):
                raise multiuser.RevisionConflict(
                    f"stale team profile revision expected={expected_revision} actual={current['revision']}"
                )
            connection.execute(
                """INSERT INTO video_team_profile_history
                   (video_id, revision, state_json, created_at) VALUES (?, ?, ?, ?)""",
                (video_id, current["revision"], json.dumps(dict(current), ensure_ascii=False), base.utc_now()),
            )
            connection.execute(
                """UPDATE video_team_profiles SET
                     team_a_hex=?, team_a_name=?, team_b_hex=?, team_b_name=?,
                     period_split_sec=?, first_period_left_team=?, second_period_left_team=?,
                     status=?, reviewer=?, revision=revision+1, updated_at=?
                   WHERE video_id=?""",
                (
                    team_a_hex, self._clean_team_name(team_a.get("display_name"), "队伍 A"),
                    team_b_hex, self._clean_team_name(team_b.get("display_name"), "队伍 B"),
                    split, first_left, second_left, status, str(user["display_name"])[:120],
                    base.utc_now(), video_id,
                ),
            )
        result = self.team_profile(video_id)
        self._write_audit(
            request_id=request_id, user_id=user_id, video_id=video_id,
            segment_id="__video_team_profile__", action="team-profile",
            payload=data, result=result, status="committed",
        )
        result["request_id"] = request_id
        return result

    def bootstrap_for_user(self, user: dict) -> dict:
        result = super().bootstrap_for_user(user)
        allowed = self.allowed_video_ids(str(user["user_id"]))
        placeholders = ",".join("?" for _ in allowed)
        with self.connect() as connection:
            count = connection.execute(
                f"SELECT COUNT(*) FROM video_team_profiles WHERE video_id IN ({placeholders}) AND status='confirmed'",
                tuple(sorted(allowed)),
            ).fetchone()[0]
        result["team_profile_summary"] = {"confirmed": int(count), "total": len(allowed)}
        result["team_profile_schema_version"] = 2
        result["field_sides"] = sorted(VALID_FIELD_SIDES)
        return result

    def video_payload_for_user(self, user_id: str, video_id: str) -> dict:
        payload = super().video_payload_for_user(user_id, video_id)
        payload["team_profile"] = self.team_profile(video_id)
        # The human-confirmed profile is authoritative for all event buttons.
        payload["team_palette"] = payload["team_profile"]["teams"]
        return payload

    def export_rows(self, include_unreviewed: bool = False) -> list[dict]:
        rows = super().export_rows(include_unreviewed)
        events = {
            item["id"]: item
            for video_id in self.videos
            for item in self.events_for_video(video_id)
        }
        profiles = {video_id: self.team_profile(video_id) for video_id in self.videos}
        for row in rows:
            event = events[row["event_id"]]
            profile = profiles[row["video_id"]]
            row["field_side"] = event["review"].get("field_side") or "unknown"
            row["team_profile_revision"] = profile["revision"]
            row["team_a_hex"] = profile["teams"]["teamA"]["hex"]
            row["team_a_name"] = profile["teams"]["teamA"]["display_name"]
            row["team_b_hex"] = profile["teams"]["teamB"]["hex"]
            row["team_b_name"] = profile["teams"]["teamB"]["display_name"]
            split = profile.get("period_split_sec")
            row["period_split_sec"] = split
            if split is None:
                row["event_period"] = "unknown"
                row["left_team_at_event"] = "unknown"
            elif float(row["time_sec"]) < float(split):
                row["event_period"] = "first"
                row["left_team_at_event"] = profile["first_period_left_team"]
            else:
                row["event_period"] = "second"
                row["left_team_at_event"] = profile["second_period_left_team"]
        return rows


def patch_html_v36(text: str) -> str:
    text = _html_v35(text).replace(
        "multiuser-v34-canonical-multilabel-20260903",
        "multiuser-v36-team-calibration-20260907",
        1,
    )
    anchor = '<div id="emptyState" class="empty-state">请选择一个候选事件</div>'
    if anchor not in text:
        raise RuntimeError("v36 team setup anchor missing")
    panel = r'''<section id="teamSetupPanel" class="team-setup-panel">
          <button id="teamSetupToggle" class="team-setup-summary" type="button">
            <span><b id="teamSetupStatus">队伍信息待确认</b><small>上衣主色 · 左右半场</small></span>
            <span id="teamSetupBadges" class="team-setup-badges"></span><i>▾</i>
          </button>
          <div id="teamSetupEditor" class="team-setup-editor">
            <div class="team-sample-row"><button id="playTeamSample" type="button">▶ 播放队伍抽样片段</button><small id="teamSampleRange"></small></div>
            <div class="team-colour-grid">
              <label><span>队伍 A 上衣主色</span><div><input id="teamAColor" type="color"><input id="teamAName" type="text" maxlength="60" placeholder="队伍 A"></div></label>
              <label><span>队伍 B 上衣主色</span><div><input id="teamBColor" type="color"><input id="teamBName" type="text" maxlength="60" placeholder="队伍 B"></div></label>
            </div>
            <div class="colour-presets" id="colourPresets"><small>点选色块修改当前队伍颜色</small></div>
            <div class="period-side-grid">
              <label><span>上半场左侧</span><select id="firstPeriodLeftTeam"><option value="unknown">不明确</option><option value="teamA">队伍 A</option><option value="teamB">队伍 B</option></select></label>
              <label><span>半场切换时间</span><div><input id="periodSplitSec" type="number" min="0" step="0.1"><button id="useCurrentAsSplit" type="button">当前画面</button></div></label>
              <label><span>下半场左侧</span><select id="secondPeriodLeftTeam"><option value="unknown">不明确</option><option value="teamA">队伍 A</option><option value="teamB">队伍 B</option></select></label>
            </div>
            <div class="team-setup-actions"><button id="swapTeamColours" type="button">交换 A/B 颜色</button><button id="confirmTeamSetup" type="button">确认队伍与半场信息</button></div>
          </div>
        </section>

        '''
    return text.replace(anchor, panel + anchor, 1)


def patch_js_v36(text: str) -> str:
    text = _js_v35(text)
    state_anchor = "hitboxes: [], contextEnd: null, timelineDrawPending: false,"
    if state_anchor not in text:
        raise RuntimeError("v36 state anchor missing")
    text = text.replace(
        state_anchor,
        state_anchor + "\n  teamSetupOpen: true, activeTeamColour: 'teamA', teamSampleEnd: null,",
        1,
    )

    old_palette = '''function teamPalette(event, team) {
  const palette = event.team_evidence?.team_palette || state.video?.team_palette || {};
  return palette[team] || { hex: team === "teamA" ? "#d5b64c" : "#477bb5", display_name: team === "teamA" ? "队伍 A" : "队伍 B" };
}'''
    new_palette = '''function teamPalette(event, team) {
  const confirmed = state.video?.team_profile?.teams || {};
  const palette = confirmed[team] ? confirmed : (event?.team_evidence?.team_palette || state.video?.team_palette || {});
  return palette[team] || { hex: team === "teamA" ? "#d5b64c" : "#477bb5", display_name: team === "teamA" ? "队伍 A" : "队伍 B" };
}'''
    if old_palette not in text:
        raise RuntimeError("v36 teamPalette anchor missing")
    text = text.replace(old_palette, new_palette, 1)

    init_anchor = '''        goal_side_confidence: Number(evidence.goal_side_confidence || 0),
      };'''
    if init_anchor not in text:
        raise RuntimeError("v36 attribution init anchor missing")
    text = text.replace(
        init_anchor,
        '''        goal_side_confidence: Number(evidence.goal_side_confidence || 0),
        field_side: review.field_side || "unknown",
      };''',
        1,
    )
    card_anchor = '''    return `<article class="attribution-card label-${label}"><header><strong>${LABEL_NAMES[label]} · ${roles[label]}</strong><small>${evidence.suggested_event_team && evidence.suggested_event_team !== 'unknown' ? `自动建议 ${Math.round(Number(evidence.event_team_confidence || 0) * 100)}%` : '人工确认'}</small></header><div class="compact-team-row">${teams}</div>${goal}</article>`;'''
    if card_anchor not in text:
        raise RuntimeError("v36 attribution card anchor missing")
    text = text.replace(
        card_anchor,
        '''    const field = `<div class="compact-field-row"><span>事件区域</span>${[["left", "← 左半场"], ["middle", "中场"], ["right", "右半场 →"], ["unknown", "不明确"]].map(([side, name]) => `<button data-attr-label="${label}" data-field-side="${side}" class="${value.field_side === side ? 'selected' : ''}">${name}</button>`).join("")}</div>`;
    return `<article class="attribution-card label-${label}"><header><strong>${LABEL_NAMES[label]} · ${roles[label]}</strong><small>${evidence.suggested_event_team && evidence.suggested_event_team !== 'unknown' ? `自动建议 ${Math.round(Number(evidence.event_team_confidence || 0) * 100)}%` : '人工确认'}</small></header><div class="compact-team-row">${teams}</div>${field}${goal}</article>`;''',
        1,
    )
    click_anchor = '''  $("#attributionCards").querySelectorAll("[data-side]").forEach((button) => button.onclick = () => { state.attributionByLabel[button.dataset.attrLabel].goal_side = button.dataset.side; state.attributionByLabel[button.dataset.attrLabel].goal_side_source = "manual"; renderTeamAttribution(); });'''
    if click_anchor not in text:
        raise RuntimeError("v36 attribution click anchor missing")
    text = text.replace(
        click_anchor,
        click_anchor + '''
  $("#attributionCards").querySelectorAll("[data-field-side]").forEach((button) => button.onclick = () => { state.attributionByLabel[button.dataset.attrLabel].field_side = button.dataset.fieldSide; renderTeamAttribution(); });''',
        1,
    )

    helpers_anchor = "async function loadVideo(videoId) {"
    if helpers_anchor not in text:
        raise RuntimeError("v36 load video anchor missing")
    helpers = r'''const TEAM_COLOUR_PRESETS = ["#e53935", "#fb8c00", "#fdd835", "#43a047", "#00acc1", "#1e88e5", "#8e24aa", "#f5f5f5", "#9e9e9e", "#212121"];

function teamName(team) {
  return state.video?.team_profile?.teams?.[team]?.display_name || (team === "teamA" ? "队伍 A" : "队伍 B");
}

function renderTeamSetup() {
  const profile = state.video?.team_profile;
  if (!profile) return;
  const confirmed = profile.status === "confirmed";
  $("#teamSetupStatus").textContent = confirmed ? "队伍信息已确认" : "队伍信息待确认";
  $("#teamSetupPanel").classList.toggle("confirmed", confirmed);
  $("#teamSetupEditor").classList.toggle("hidden", !state.teamSetupOpen);
  const a = profile.teams.teamA, b = profile.teams.teamB;
  $("#teamSetupBadges").innerHTML = [["A", a], ["B", b]].map(([key, item]) => `<b><i style="background:${item.hex}"></i>${key} ${item.display_name}</b>`).join("");
  $("#teamAColor").value = a.hex; $("#teamBColor").value = b.hex;
  $("#teamAName").value = a.display_name; $("#teamBName").value = b.display_name;
  $("#periodSplitSec").value = profile.period_split_sec ?? "";
  $("#firstPeriodLeftTeam").value = profile.first_period_left_team || "unknown";
  $("#secondPeriodLeftTeam").value = profile.second_period_left_team || "unknown";
  $("#firstPeriodLeftTeam").options[1].textContent = `A · ${a.display_name}`;
  $("#firstPeriodLeftTeam").options[2].textContent = `B · ${b.display_name}`;
  $("#secondPeriodLeftTeam").options[1].textContent = `A · ${a.display_name}`;
  $("#secondPeriodLeftTeam").options[2].textContent = `B · ${b.display_name}`;
  $("#teamSampleRange").textContent = `${formatTime(profile.sample_start_sec)} – ${formatTime(profile.sample_end_sec)}`;
  $("#colourPresets").innerHTML = `<small>当前修改 ${state.activeTeamColour === 'teamA' ? '队伍 A' : '队伍 B'}</small>` + TEAM_COLOUR_PRESETS.map((colour) => `<button data-preset-colour="${colour}" style="background:${colour}" title="${colour}"></button>`).join("");
  $("#colourPresets").querySelectorAll("button").forEach((button) => button.onclick = () => { $(`#${state.activeTeamColour === 'teamA' ? 'teamAColor' : 'teamBColor'}`).value = button.dataset.presetColour; });
}

async function saveTeamSetup() {
  const profile = state.video?.team_profile; if (!profile) return;
  const payload = {
    expected_revision: profile.revision, status: "confirmed",
    teams: {
      teamA: { hex: $("#teamAColor").value, display_name: $("#teamAName").value.trim() || "队伍 A" },
      teamB: { hex: $("#teamBColor").value, display_name: $("#teamBName").value.trim() || "队伍 B" },
    },
    period_split_sec: $("#periodSplitSec").value,
    first_period_left_team: $("#firstPeriodLeftTeam").value,
    second_period_left_team: $("#secondPeriodLeftTeam").value,
  };
  try {
    state.video.team_profile = await api(`/api/videos/${state.currentVideoId}/team-profile`, { method: "POST", body: JSON.stringify(payload) });
    state.video.team_palette = state.video.team_profile.teams; state.teamSetupOpen = false;
    renderTeamSetup(); renderTeamAttribution(); toast("队伍颜色与左右半场已保存");
  } catch (error) { toast(`队伍信息保存失败：${error.message}`); }
}

function bindTeamSetup() {
  $("#teamSetupToggle").onclick = () => { state.teamSetupOpen = !state.teamSetupOpen; renderTeamSetup(); };
  $("#teamAColor").onclick = () => { state.activeTeamColour = "teamA"; };
  $("#teamBColor").onclick = () => { state.activeTeamColour = "teamB"; };
  $("#teamAName").onfocus = () => { state.activeTeamColour = "teamA"; };
  $("#teamBName").onfocus = () => { state.activeTeamColour = "teamB"; };
  $("#swapTeamColours").onclick = () => {
    const colour = $("#teamAColor").value, name = $("#teamAName").value;
    $("#teamAColor").value = $("#teamBColor").value; $("#teamAName").value = $("#teamBName").value;
    $("#teamBColor").value = colour; $("#teamBName").value = name;
  };
  $("#useCurrentAsSplit").onclick = () => { $("#periodSplitSec").value = $("#player").currentTime.toFixed(1); };
  $("#playTeamSample").onclick = () => {
    const profile = state.video.team_profile, player = $("#player");
    player.currentTime = Number(profile.sample_start_sec); state.contextEnd = Number(profile.sample_end_sec);
    player.play().catch(() => toast("请点击画面开始播放"));
  };
  $("#confirmTeamSetup").onclick = saveTeamSetup;
}

'''
    text = text.replace(helpers_anchor, helpers + helpers_anchor, 1)
    load_anchor = '''  state.video = await api(`/api/videos/${videoId}`);
  const player = $("#player");'''
    if load_anchor not in text:
        raise RuntimeError("v36 video payload anchor missing")
    text = text.replace(
        load_anchor,
        '''  state.video = await api(`/api/videos/${videoId}`);
  state.teamSetupOpen = state.video.team_profile?.status !== "confirmed";
  renderTeamSetup();
  const player = $("#player");''',
        1,
    )
    init_anchor = '''  setupTimeline();
  $("#videoSelect").onchange = (event) => loadVideo(event.target.value);'''
    if init_anchor not in text:
        raise RuntimeError("v36 init anchor missing")
    text = text.replace(
        init_anchor,
        '''  setupTimeline();
  bindTeamSetup();
  $("#videoSelect").onchange = (event) => loadVideo(event.target.value);''',
        1,
    )
    return text


hierarchical.patch_hierarchical_html = patch_html_v36
hierarchical.patch_hierarchical_js = patch_js_v36
hierarchical.HIERARCHICAL_CSS += r'''
.team-setup-panel{margin:7px 0 9px;border:1px solid #3a4a5b;border-radius:9px;background:#111a24;overflow:hidden}.team-setup-panel.confirmed{border-color:#3f745e}.team-setup-summary{display:flex;width:100%;align-items:center;justify-content:space-between;gap:8px;padding:8px 10px;color:#e6edf5;border:0;background:#172330;cursor:pointer;text-align:left}.team-setup-summary>span:first-child{display:grid}.team-setup-summary small{color:#748596;font-size:8px}.team-setup-summary>i{color:#7e8b98}.team-setup-badges{display:flex;gap:5px;min-width:0}.team-setup-badges b{display:flex;align-items:center;gap:4px;max-width:120px;padding:3px 6px;border:1px solid #405061;border-radius:6px;color:#bbc7d2;font-size:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.team-setup-badges i{width:12px;height:12px;border:2px solid #fff9;border-radius:50%;flex:none}.team-setup-editor{display:grid;gap:7px;padding:8px}.team-sample-row,.team-setup-actions{display:flex;align-items:center;justify-content:space-between;gap:7px}.team-sample-row button,.team-setup-actions button,.period-side-grid button{min-height:31px;padding:0 9px;color:#d7e1eb;border:1px solid #435366;border-radius:6px;background:#1c2936;cursor:pointer}.team-sample-row small{color:#8291a0;font:9px ui-monospace,monospace}.team-colour-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}.team-colour-grid label>span,.period-side-grid label>span{display:block;margin-bottom:3px;color:#8e9dad;font-size:8px}.team-colour-grid label>div{display:grid;grid-template-columns:36px 1fr;gap:4px}.team-colour-grid input[type=color]{width:36px;height:34px;padding:2px;border:1px solid #46586a;border-radius:6px;background:#101820}.team-colour-grid input[type=text],.period-side-grid input,.period-side-grid select{box-sizing:border-box;width:100%;height:34px;padding:0 7px;color:#e5edf5;border:1px solid #405164;border-radius:6px;background:#0f1822}.colour-presets{display:flex;align-items:center;gap:4px;overflow-x:auto}.colour-presets small{margin-right:3px;color:#8291a0;font-size:8px;white-space:nowrap}.colour-presets button{width:22px;height:22px;flex:none;border:2px solid #ffffff88;border-radius:50%;cursor:pointer}.period-side-grid{display:grid;grid-template-columns:1fr 1.25fr 1fr;gap:6px}.period-side-grid label>div{display:grid;grid-template-columns:1fr auto;gap:4px}.team-setup-actions button:last-child{flex:1;color:#0f1b12;border-color:#70d3a7;background:#70d3a7;font-weight:800}.compact-field-row{display:grid;grid-template-columns:66px repeat(4,minmax(0,1fr));align-items:center;gap:4px;margin-top:4px}.compact-field-row>span{color:#8290a0;font-size:9px;text-align:center}.compact-field-row button{min-width:0;min-height:29px;color:#c8d1db;border:1px solid #394858;border-radius:6px;background:#19232e;cursor:pointer;font-size:9px;white-space:nowrap}.compact-field-row button.selected{color:#101710;border-color:var(--accent);background:var(--accent);font-weight:800}@media(max-width:760px){.team-colour-grid,.period-side-grid{grid-template-columns:1fr}.team-setup-badges b{max-width:85px}.compact-field-row{grid-template-columns:58px repeat(2,minmax(0,1fr))}}
'''


class TeamCalibrationHandler(_BaseHandler):
    server_version = "FootballEventReview/3.6"

    @property
    def store(self) -> TeamCalibrationStore:
        return self.server.store  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/healthz":
            return self.send_json({"ok": True, "version": "v36-team-calibration"})
        return super().do_GET()

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path.startswith("/api/videos/") and path.endswith("/team-profile"):
            if not self.same_origin_post():
                return self.send_json({"error": "Cross-site request rejected"}, HTTPStatus.FORBIDDEN)
            user = self.require_user()
            if user is None:
                return
            try:
                video_id = path.split("/")[3]
                return self.send_json(
                    self.store.update_team_profile_for_user(user, video_id, self.read_json())
                )
            except multiuser.RevisionConflict as error:
                return self.send_json(
                    {"error": str(error), "code": "revision_conflict"}, HTTPStatus.CONFLICT
                )
            except PermissionError:
                return self.send_json({"error": "Forbidden"}, HTTPStatus.FORBIDDEN)
            except (ValueError, json.JSONDecodeError) as error:
                return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except KeyError as error:
                return self.send_json({"error": f"Not found: {error}"}, HTTPStatus.NOT_FOUND)
            except Exception as error:  # noqa: BLE001 - surfaced to private UI
                return self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        return super().do_POST()


def main() -> None:
    args = v35.v34.v33.v32.parse_args()
    store = TeamCalibrationStore(args.manifest, args.db, args.access_config, args.proxy_root)
    server = v35.v34.v33.v32.ReviewThreadingHTTPServer(
        (args.host, args.port), TeamCalibrationHandler
    )
    server.store = store  # type: ignore[attr-defined]
    if bool(args.tls_cert) != bool(args.tls_key):
        raise ValueError("--tls-cert and --tls-key must be provided together")
    scheme = "http"
    if args.tls_cert and args.tls_key:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
        server.is_tls = True  # type: ignore[attr-defined]
        scheme = "https"
    else:
        server.is_tls = False  # type: ignore[attr-defined]
    print(f"Multi-user football review UI v36: {scheme}://{args.host}:{args.port}", flush=True)
    print(f"Review database: {store.db_path}", flush=True)
    print(f"Assigned users: {len(store.users_by_id)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
