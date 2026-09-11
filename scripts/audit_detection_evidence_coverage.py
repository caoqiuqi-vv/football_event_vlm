#!/usr/bin/env python
"""Audit detector evidence coverage around GT events (train split).

For every accepted shot/save/set_piece anchor in the train split, measure how
often the detector sees the ball / goal / players within a small time window
around the anchor, and compare with the background presence rate.  This
decides which evidence channels are viable as supervision targets before any
model change (docs/football_e17_hardneg_audit_and_plan_20260827.md, route E4).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLS_BALL, CLS_GOAL, CLS_PERSON = 1, 2, 0


def load_detection_index(det_dir: Path, video_id: str):
    """Return per-second presence info for ball/goal/person counts."""
    path = det_dir / video_id / "compact_detection_tracks.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    fps = float(data["metadata"]["sampling"]["source_fps"])
    ball_times, goal_times = [], []
    person_frames = []  # (time, count)
    for frame in data.get("frames", []):
        t = float(frame["f"]) / fps
        has_ball = has_goal = False
        persons = 0
        for obj in frame.get("o", []):
            cls = int(obj[0])
            if cls == CLS_BALL:
                has_ball = True
            elif cls == CLS_GOAL:
                has_goal = True
            elif cls == CLS_PERSON:
                persons += 1
        if has_ball:
            ball_times.append(t)
        if has_goal:
            goal_times.append(t)
        person_frames.append((t, persons))
    return {
        "fps": fps,
        "ball": np.asarray(ball_times),
        "goal": np.asarray(goal_times),
        "person": np.asarray(person_frames),
        "frames": len(data.get("frames", [])),
    }


def presence(times: np.ndarray, anchor: float, radius: float) -> bool:
    if times.size == 0:
        return False
    return bool(np.any(np.abs(times - anchor) <= radius))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--det-dir", type=Path,
                        default=Path("/mnt/data_16t/football/detection_and_track_result"))
    parser.add_argument("--radius-sec", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import train_football_events as base

    cfg = base.load_config(args.config, [])
    base.configure_label_schema(cfg)
    records, events_by_video = base.load_long_video_records(cfg, args.split)
    del records

    labels = list(base.LABELS)
    per_class = {label: {"n": 0, "ball": 0, "goal": 0, "either": 0,
                         "persons_median": []} for label in labels}
    missing_det, no_events = [], []
    bg_ball_rates, bg_goal_rates = [], []
    for (source, video_id), events in sorted(events_by_video.items()):
        det = load_detection_index(args.det_dir, video_id)
        if det is None:
            missing_det.append(video_id)
            continue
        accepted = [e for e in events if not e.is_ignored]
        if not accepted:
            no_events.append(video_id)
            continue
        # background rates from the raw frame stream
        total = det["frames"]
        if total:
            # ball/goal lists were built only when present; infer rates
            pass
        for event in accepted:
            anchor = float(event.anchor_time)
            for label, value in zip(labels, event.labels):
                if value <= 0:
                    continue
                slot = per_class[label]
                slot["n"] += 1
                ball = presence(det["ball"], anchor, args.radius_sec)
                goal = presence(det["goal"], anchor, args.radius_sec)
                slot["ball"] += int(ball)
                slot["goal"] += int(goal)
                slot["either"] += int(ball or goal)
                persons = det["person"]
                if persons.size:
                    near = np.abs(persons[:, 0] - anchor) <= args.radius_sec
                    if near.any():
                        slot["persons_median"].append(
                            float(np.median(persons[near, 1]))
                        )
    report = {"split": args.split, "radius_sec": args.radius_sec,
              "videos_missing_detection": missing_det,
              "videos_without_events": no_events, "per_class": {}}
    print(f"== detection evidence coverage (split={args.split}, +/-{args.radius_sec}s)")
    for label, slot in per_class.items():
        n = max(slot["n"], 1)
        pm = float(np.median(slot["persons_median"])) if slot["persons_median"] else None
        report["per_class"][label] = {
            "events": slot["n"],
            "ball_coverage": slot["ball"] / n,
            "goal_coverage": slot["goal"] / n,
            "either_coverage": slot["either"] / n,
            "persons_median_near_anchor": pm,
        }
        print(f"  {label}: events={slot['n']} ball={slot['ball']/n:.3f} "
              f"goal={slot['goal']/n:.3f} either={slot['either']/n:.3f} "
              f"persons_median={pm}")
    print(f"  videos missing detection files: {len(missing_det)}")
    print(f"  videos without events: {len(no_events)}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
