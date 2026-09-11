#!/usr/bin/env python
"""Sample high-confidence FP candidates from a review manifest for human verification.

Strategy (2026-08-20, GT-quality diagnostic):
  - set_piece: ALL fp with score >= 0.95 (model extremely confident, GT silent) + 8 random in [0.7, 0.95)
  - shot:      ALL fp with score >= 0.8 + 8 random in [0.6, 0.8)
  - save:      top-45 fp by score (>= 0.75) + 8 random in [0.6, 0.75)
  - per-video cap 8 events; reviewer answers: accept = GT missing, delete = wrong label/other_action.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SAMPLES = {
    "set_piece": {"high_min": 0.95, "high_cap": 46, "rand_min": 0.70, "rand_max": 0.95, "rand_cap": 24},
    "shot": {"high_min": 0.80, "high_cap": 38, "rand_min": 0.60, "rand_max": 0.80, "rand_cap": 16},
    "save": {"high_min": 0.75, "high_cap": 40, "rand_min": 0.60, "rand_max": 0.75, "rand_cap": 16},
}
TOTAL_BUDGET = 180


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    selected: list[tuple[str, dict]] = []  # (video_id, event)
    taken: set[str] = set()

    all_events: dict[str, list[dict]] = defaultdict(list)
    for video in manifest["videos"]:
        for e in video["events"]:
            all_events[e["label"]].append((video["video_id"], e))

    for lab, cfg in SAMPLES.items():
        pool = sorted(all_events[lab], key=lambda ve: -ve[1]["score"])
        high = [ve for ve in pool if ve[1]["score"] >= cfg["high_min"]][: cfg["high_cap"]]
        rest = [ve for ve in pool if cfg["rand_min"] <= ve[1]["score"] < cfg["rand_max"]]
        rand = rng.sample(rest, min(cfg["rand_cap"], len(rest)))
        for vid, e in high + rand:
            key = e["id"]
            if key in taken:
                continue
            taken.add(key)
            e["_sample_band"] = "high" if e["score"] >= cfg["high_min"] else "rand"
            selected.append((vid, e))

    by_label_band = Counter(f"{e['label']}/{e['_sample_band']}" for _, e in selected)
    by_label = Counter(e["label"] for _, e in selected)
    by_video = Counter(vid for vid, _ in selected)
    by_band_video = Counter((vid, e["_sample_band"]) for vid, e in selected)

    videos_out = []
    for video in manifest["videos"]:
        evs = [e for _, e in selected if _ == video["video_id"]]
        if not evs:
            continue
        out = {k: v for k, v in video.items() if k != "events"}
        for e in evs:
            e.pop("_sample_band", None)
        out["events"] = sorted(evs, key=lambda e: -e["score"])
        videos_out.append(out)

    out_manifest = {
        "schema_version": manifest.get("schema_version", 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": manifest["labels"],
        "review_labels": manifest.get("review_labels", manifest["labels"]),
        "source": {**manifest.get("source", {}), "sampling": SAMPLES},
        "videos": videos_out,
        "summary": {
            "num_videos": len(videos_out),
            "num_events": len(selected),
            "by_label_band": dict(sorted(by_label_band.items())),
            "by_label": dict(by_label),
            "top_videos": dict(sorted(by_video.items(), key=lambda kv: -kv[1])[:8]),
        },
    }
    args.output.write_text(json.dumps(out_manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out_manifest["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
