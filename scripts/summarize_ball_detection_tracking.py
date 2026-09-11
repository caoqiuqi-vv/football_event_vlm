#!/usr/bin/env python
"""Summarize complementary YOLO train/val and RF-DETR test18 ball tracks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


YOLO_COUNT_KEYS = (
    "sampled_frames",
    "raw_candidate_frames",
    "raw_candidates",
    "tracked_frames",
    "pseudo_frames",
    "observed_frames",
    "raw_repair_frames",
    "interpolated_frames",
    "heatmap_usable_frames",
    "motion_usable_frames",
    "tracks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-root", type=Path, required=True)
    parser.add_argument("--rfdetr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_yolo(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("*/track_summary.json"))
    aggregate = Counter()
    videos: dict[str, Any] = {}
    incomplete: list[str] = []
    for path in files:
        row = json.loads(path.read_text(encoding="utf-8"))
        video_id = str(row["video_id"])
        item = {key: int(row.get(key, 0)) for key in YOLO_COUNT_KEYS}
        item["raw_detection_frame_fraction"] = ratio(
            item["raw_candidate_frames"], item["sampled_frames"]
        )
        item["tracked_frame_fraction"] = ratio(
            item["tracked_frames"], item["sampled_frames"]
        )
        item["pseudo_frame_fraction"] = ratio(
            item["pseudo_frames"], item["sampled_frames"]
        )
        item["heatmap_usable_fraction"] = ratio(
            item["heatmap_usable_frames"], item["sampled_frames"]
        )
        videos[video_id] = item
        aggregate.update({key: item[key] for key in YOLO_COUNT_KEYS})
        if not (path.parent / "tracking_complete.json").is_file():
            incomplete.append(video_id)
    result = {key: int(aggregate[key]) for key in YOLO_COUNT_KEYS}
    result.update(
        {
            "video_count": len(videos),
            "incomplete_video_ids": incomplete,
            "raw_detection_frame_fraction": ratio(
                result["raw_candidate_frames"], result["sampled_frames"]
            ),
            "tracked_frame_fraction": ratio(
                result["tracked_frames"], result["sampled_frames"]
            ),
            "pseudo_frame_fraction": ratio(
                result["pseudo_frames"], result["sampled_frames"]
            ),
            "heatmap_usable_fraction": ratio(
                result["heatmap_usable_frames"], result["sampled_frames"]
            ),
            "motion_usable_fraction": ratio(
                result["motion_usable_frames"], result["sampled_frames"]
            ),
        }
    )
    return {"root": str(root), "aggregate": result, "videos": videos}


def summarize_rfdetr(root: Path) -> dict[str, Any]:
    videos: dict[str, Any] = {}
    aggregate = Counter()
    for video_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        video = Counter()
        part_count = 0
        for summary_path in sorted(video_dir.glob("part_*/inference_summary.json")):
            tracking_path = summary_path.parent / "tracking_summary.json"
            manifest_path = summary_path.parent / "canonical_v2" / "manifest.json"
            if not tracking_path.is_file() or not manifest_path.is_file():
                continue
            inference = json.loads(summary_path.read_text(encoding="utf-8"))
            tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
            canonical = inference.get("canonical_v2", {})
            video["frames"] += int(canonical.get("frame_count", 0))
            video["detections"] += int(canonical.get("detection_count", 0))
            video["tracks"] += int(tracking.get("track_count", 0))
            video["confirmed_tracks"] += int(tracking.get("confirmed_count", 0))
            video["track_state_rows"] += int(tracking.get("state_rows", 0))
            observations = tracking.get("observation_counts", {})
            video["observed_track_states"] += int(observations.get("observed", 0))
            video["predicted_track_states"] += int(observations.get("predicted", 0))
            gap_fill = tracking.get("gap_fill", {})
            video["gap_fill_events"] += int(gap_fill.get("event_count", 0))
            video["gap_fill_frames"] += int(gap_fill.get("frame_count", 0))
            part_count += 1
        if part_count == 0:
            continue
        item = dict(video)
        item["parts"] = part_count
        item["detections_per_frame"] = ratio(item.get("detections", 0), item.get("frames", 0))
        item["track_state_rows_per_frame"] = ratio(
            item.get("track_state_rows", 0), item.get("frames", 0)
        )
        videos[video_dir.name] = item
        aggregate.update(video)
        aggregate["parts"] += part_count
    result = {key: int(value) for key, value in aggregate.items()}
    result["video_count"] = len(videos)
    result["detections_per_frame"] = ratio(
        result.get("detections", 0), result.get("frames", 0)
    )
    result["track_state_rows_per_frame"] = ratio(
        result.get("track_state_rows", 0), result.get("frames", 0)
    )
    return {"root": str(root), "aggregate": result, "videos": videos}


def markdown(summary: dict[str, Any]) -> str:
    yolo = summary["yolo"]["aggregate"]
    rf = summary["rfdetr"]["aggregate"]
    return f"""# 足球检测与跟踪汇总

两套结果按数据分区互补，不是同视频上的 detector ensemble：

- YOLO：{yolo['video_count']} 个非 test18 视频，用作训练/验证离线伪标签。
- RF-DETR：{rf['video_count']} 个 test18 视频，只用于独立评测和可视化，禁止进入训练。

| 来源 | 视频 | 抽样/处理帧 | 检测框 | 轨迹 | 跟踪状态行 |
|---|---:|---:|---:|---:|---:|
| YOLO + det_and_track | {yolo['video_count']} | {yolo['sampled_frames']} | {yolo['raw_candidates']} | {yolo['tracks']} | {yolo['pseudo_frames']} |
| RF-DETR hybrid | {rf['video_count']} | {rf.get('frames', 0)} | {rf.get('detections', 0)} | {rf.get('tracks', 0)} | {rf.get('track_state_rows', 0)} |

YOLO 的 raw detection frame coverage 为 {yolo['raw_detection_frame_fraction']:.2%}，
最终 pseudo coverage 为 {yolo['pseudo_frame_fraction']:.2%}，
heatmap 可用 coverage 为 {yolo['heatmap_usable_fraction']:.2%}，
motion 可用 coverage 为 {yolo['motion_usable_fraction']:.2%}。

RF-DETR 平均每帧 {rf.get('detections_per_frame', 0.0):.3f} 个检测框，
每帧 {rf.get('track_state_rows_per_frame', 0.0):.3f} 条跟踪状态；
其中 observed={rf.get('observed_track_states', 0)}，
predicted={rf.get('predicted_track_states', 0)}，
gap-filled frames={rf.get('gap_fill_frames', 0)}。

注意：RF-DETR 的“每帧跟踪状态行”不是 unique-frame coverage；同帧多轨迹会贡献多行。
"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "football-ball-detection-tracking-summary-v1",
        "partition_policy": {
            "yolo": "non-test18; eligible for train/validation pseudo supervision",
            "rfdetr": "test18 only; evaluation/visualization only",
        },
        "yolo": summarize_yolo(args.yolo_root.expanduser().resolve()),
        "rfdetr": summarize_rfdetr(args.rfdetr_root.expanduser().resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({key: value["aggregate"] for key, value in result.items() if key in {"yolo", "rfdetr"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
