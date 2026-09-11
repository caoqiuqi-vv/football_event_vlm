#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand candidate centers into temporal views that cover +/- uncertainty.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--video-ids", required=True)
    parser.add_argument("--offsets", default="-5,0,5")
    args = parser.parse_args()
    offsets = [float(x) for x in args.offsets.split(",")]
    source_root, output_root = Path(args.input_root), Path(args.output_root)
    summary = {"offsets_sec": offsets, "videos": {}}
    for video_id in [x.strip() for x in args.video_ids.split(",") if x.strip()]:
        candidates = json.loads((source_root / video_id / "candidate_proposals.json").read_text())
        views = []
        for candidate_index, candidate in enumerate(candidates):
            for view_index, offset in enumerate(offsets):
                views.append({
                    "time_sec": max(float(candidate["time_sec"]) + offset, 0.0),
                    "confidence": float(candidate.get("confidence", 0.0)),
                    "event_type": "candidate_multiview",
                    "source": f"candidate={candidate_index};view={view_index};offset={offset:+g}",
                    "candidate_index": candidate_index,
                    "view_index": view_index,
                    "offset_sec": offset,
                    "candidate_time_sec": float(candidate["time_sec"]),
                })
        views.sort(key=lambda x: (x["time_sec"], x["candidate_index"], x["view_index"]))
        out = output_root / video_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "shot_proposals.json").write_text(json.dumps(views, ensure_ascii=False, indent=2))
        (out / "multiview_manifest.json").write_text(json.dumps(views, ensure_ascii=False, indent=2))
        summary["videos"][video_id] = {"candidates": len(candidates), "views": len(views)}
    summary["total_candidates"] = sum(x["candidates"] for x in summary["videos"].values())
    summary["total_views"] = sum(x["views"] for x in summary["videos"].values())
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
