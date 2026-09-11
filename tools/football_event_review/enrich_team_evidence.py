#!/usr/bin/env python3
"""Attach team palettes and goal-side evidence to a football review manifest.

The script is intentionally conservative: team clustering alone identifies two
kit groups, but it does not prove which group executed an event. Therefore it
never invents ``suggested_event_team`` without explicit possession evidence.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from colorsys import rgb_to_hsv
from pathlib import Path
from typing import Any


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def opencv_lab_to_hex(lab: list[float]) -> str:
    """Convert OpenCV 8-bit Lab coordinates to an sRGB hex color."""
    if len(lab) != 3:
        return "#777777"
    lightness = float(lab[0]) * 100.0 / 255.0
    a = float(lab[1]) - 128.0
    b = float(lab[2]) - 128.0
    fy = (lightness + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def inverse(value: float) -> float:
        return value**3 if value > delta else 3.0 * delta**2 * (value - 4.0 / 29.0)

    x = 0.95047 * inverse(fx)
    y = 1.00000 * inverse(fy)
    z = 1.08883 * inverse(fz)
    linear = (
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    )

    def gamma(value: float) -> int:
        value = 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1.0 / 2.4) - 0.055
        return round(clamp(value) * 255.0)

    return "#{:02x}{:02x}{:02x}".format(*(gamma(value) for value in linear))


def color_name(hex_color: str) -> str:
    red, green, blue = (int(hex_color[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
    hue, saturation, value = rgb_to_hsv(red, green, blue)
    if value < 0.22:
        return "黑色"
    if saturation < 0.14 and value > 0.82:
        return "白色"
    if saturation < 0.18:
        return "灰色"
    degrees = hue * 360.0
    if degrees < 15 or degrees >= 345:
        return "红色"
    if degrees < 45:
        return "橙色"
    if degrees < 70:
        return "黄色"
    if degrees < 165:
        return "绿色"
    if degrees < 200:
        return "青色"
    if degrees < 255:
        return "蓝色"
    if degrees < 290:
        return "紫色"
    return "粉/紫色"


def team_assignment_paths(root: Path, video_id: str) -> list[Path]:
    video_root = root / video_id
    if not video_root.exists():
        return []
    return sorted(video_root.glob("*/team_assignments.json"))


def part_index(path: Path) -> int:
    name = path.parent.name
    try:
        return int(name.rsplit("_part_", 1)[1])
    except (IndexError, ValueError):
        return 0


def max_assignment_frame(data: dict[str, Any]) -> int:
    maximum = -1
    for track in data.get("tracks", []):
        for frame_range in track.get("active_frame_ranges", []):
            if len(frame_range) >= 2:
                maximum = max(maximum, int(frame_range[1]))
    for track in data.get("goal_tracking", {}).get("tracks", []):
        maximum = max(maximum, int(track.get("last_frame", -1)))
    return maximum


def palette_from_assignment(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    anchors = data.get("teams", {}).get("anchors", {})
    clusters = {
        str(item.get("cluster_id")): item
        for item in data.get("teams", {}).get("clusters", [])
    }
    output: dict[str, dict[str, Any]] = {}
    for cluster_id, team in anchors.items():
        if team not in {"teamA", "teamB"}:
            continue
        cluster = clusters.get(str(cluster_id), {})
        hex_color = opencv_lab_to_hex(cluster.get("center_lab", []))
        output[team] = {
            "hex": hex_color,
            "color_name": color_name(hex_color),
            "display_name": f"队伍 {'A' if team == 'teamA' else 'B'} · {color_name(hex_color)}",
            "cluster_id": int(cluster_id),
            "quality": round(float(cluster.get("mean_quality", 0.0) or 0.0), 4),
        }
    return output


def load_team_summary(root: Path, video_id: str) -> dict[str, Any]:
    paths = team_assignment_paths(root, video_id)
    if not paths:
        return {"parts": [], "palette": {}, "reliable": False, "coverage_end_sec": 0.0}
    parts = []
    palette: dict[str, dict[str, Any]] = {}
    reliable = True
    coverage_end_sec = 0.0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        fps = float(data.get("fps", 0.0) or 0.0)
        end_frame = max_assignment_frame(data)
        # Existing outputs use original frame IDs inside each video part. part_0000
        # therefore safely begins at zero; later parts require explicit offsets.
        start_sec = float(data.get("part_start_sec", 0.0) or 0.0)
        end_sec = start_sec + (end_frame / fps if fps > 0 and end_frame >= 0 else 0.0)
        coverage_end_sec = max(coverage_end_sec, end_sec)
        part_palette = palette_from_assignment(data)
        if not palette:
            palette = part_palette
        reliable = reliable and bool(data.get("teams", {}).get("reliable_two_team_clustering"))
        parts.append({
            "path": str(path.resolve()),
            "part_index": part_index(path),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "reliable": bool(data.get("teams", {}).get("reliable_two_team_clustering")),
        })
    return {"parts": parts, "palette": palette, "reliable": reliable, "coverage_end_sec": coverage_end_sec}


def goal_evidence_by_event(detection_path: Path, events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not detection_path.exists():
        return {}
    with gzip.open(detection_path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    metadata = data.get("metadata", {})
    fps = float(metadata.get("fps", 0.0) or 0.0)
    width = float(metadata.get("image_size", {}).get("width", 0.0) or 0.0)
    if fps <= 0 or width <= 0:
        return {}
    event_ranges = []
    for event in events:
        if event.get("label") not in {"shot", "save"}:
            continue
        start = max(0.0, float(event.get("support_start_sec", event["time_sec"] - 3.0)))
        end = max(start, float(event.get("support_end_sec", event["time_sec"] + 5.0)))
        event_ranges.append((event["id"], int(math.floor(start * fps)), int(math.ceil(end * fps))))
    stats = {event_id: {"left": set(), "right": set(), "left_conf": [], "right_conf": []} for event_id, _, _ in event_ranges}
    for frame in data.get("frames", []):
        frame_id = int(frame.get("f", -1))
        active = [(event_id, start, end) for event_id, start, end in event_ranges if start <= frame_id <= end]
        if not active:
            continue
        for obj in frame.get("o", []):
            if len(obj) < 6 or int(obj[0]) != 2:
                continue
            confidence = float(obj[1])
            center_x = 0.5 * (float(obj[2]) + float(obj[4]))
            side = "left" if center_x < width * 0.5 else "right"
            for event_id, _, _ in active:
                stats[event_id][side].add(frame_id)
                stats[event_id][f"{side}_conf"].append(confidence)
    result = {}
    for event_id, item in stats.items():
        left = len(item["left"])
        right = len(item["right"])
        visible = [side for side, count in (("left", left), ("right", right)) if count]
        total = left + right
        suggested = "unknown"
        confidence = 0.0
        if total:
            dominant_side, dominant = max((("left", left), ("right", right)), key=lambda pair: pair[1])
            other = min(left, right)
            dominance = dominant / total
            mean_conf = sum(item[f"{dominant_side}_conf"]) / max(1, len(item[f"{dominant_side}_conf"]))
            confidence = clamp(dominance * mean_conf)
            if dominant >= 3 and dominant >= max(2, other * 2) and confidence >= 0.45:
                suggested = dominant_side
        result[event_id] = {
            "visible_goal_sides": visible,
            "suggested_goal_side": suggested,
            "goal_side_confidence": round(confidence, 4),
            "goal_frame_counts": {"left": left, "right": right},
        }
    del data
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--team-results-root", type=Path, required=True)
    parser.add_argument("--clip-plan", type=Path, help="Representative clip plan with original timestamps")
    parser.add_argument("--detection-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    clip_plan = {}
    if args.clip_plan:
        plan = json.loads(args.clip_plan.read_text(encoding="utf-8"))
        clip_plan = {str(item["video_id"]): item for item in plan.get("videos", [])}
    covered_events = 0
    goal_suggestions = 0
    for video in manifest.get("videos", []):
        video_id = str(video["video_id"])
        team = load_team_summary(args.team_results_root, video_id)
        sample = clip_plan.get(video_id, {})
        video["team_palette"] = team["palette"]
        video["team_cluster"] = {
            "reliable": team["reliable"],
            "coverage_end_sec": round(team["coverage_end_sec"], 3),
            "source_time_offset_sec": float(sample.get("sample_start_sec", 0.0) or 0.0),
            "parts": team["parts"],
        }
        video["team_calibration"] = {
            "sample_start_sec": float(sample.get("sample_start_sec", 0.0) or 0.0),
            "sample_end_sec": float(sample.get("sample_end_sec", 30.0) or 30.0),
            "sample_clip_path": str(sample.get("clip_path", "")),
            "selection_reason": str(sample.get("selection_reason", "team_cluster_default")),
            "colour_region_mode": "upper",
            "period_split_sec": None,
            "first_period_left_team": "unknown",
            "second_period_left_team": "unknown",
        }
        goal_by_event = {}
        if args.detection_root:
            detection_path = args.detection_root / video_id / "compact_detection_tracks.json.gz"
            goal_by_event = goal_evidence_by_event(detection_path, video.get("events", []))
        for event in video.get("events", []):
            # A representative clip establishes kit palettes for the match,
            # but it cannot establish possession for an arbitrary event.
            covered = bool(team["palette"])
            evidence = {
                "coverage_status": "palette_available" if covered else "not_covered",
                "suggested_event_team": "unknown",
                "event_team_confidence": 0.0,
                "possession_team": "unknown",
                "team_palette": team["palette"],
                "reason": (
                    "已有两队上衣主色，但缺少可靠的事件触球归属；请人工确认执行方"
                    if covered else "代表片段未得到可靠队伍颜色；请先人工修正队伍颜色"
                ),
                **goal_by_event.get(event["id"], {}),
            }
            if event.get("label") == "set_piece":
                evidence["suggested_goal_side"] = "not_applicable"
                evidence["goal_side_confidence"] = 1.0
            if covered:
                covered_events += 1
            if evidence.get("suggested_goal_side") in {"left", "right"}:
                goal_suggestions += 1
            event["team_evidence"] = evidence

    manifest["schema_version"] = max(4, int(manifest.get("schema_version", 1)))
    manifest.setdefault("source", {}).update({
        "team_results_root": str(args.team_results_root.resolve()),
        "team_attribution_policy": "conservative_no_team_guess_without_possession",
        "goal_side_policy": "dominant_visible_goal_in_event_support_window",
    })
    manifest.setdefault("summary", {}).update({
        "team_cluster_covered_events": covered_events,
        "goal_side_auto_suggestions": goal_suggestions,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "team_cluster_covered_events": covered_events,
        "goal_side_auto_suggestions": goal_suggestions,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
