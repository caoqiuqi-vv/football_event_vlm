#!/usr/bin/env python
"""Sharded full-timeline inference for a trained no-NMS goal retriever."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "football_e2e_spotter" / "src"
for path in (ROOT, PKG):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from football_e2e_spotter.goal_annotations import LABELS  # noqa: E402
from football_e2e_spotter.goal_retriever import GoalLongContextRetriever  # noqa: E402
from football_e2e_spotter.goal_retriever_eval import scan_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("invalid rank")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["model_config"]
    model = GoalLongContextRetriever(
        config["appearance_dim"], config["motion_dim"], config["audio_dim"],
        hidden_dim=config["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    video_ids = [str(item["media_id"]) for item in payload[args.split]][args.rank::args.world_size]
    rows, durations = [], {}
    for position, video_id in enumerate(video_ids):
        base = args.feature_root / args.split / video_id
        if not ((base / "metadata.json").is_file() or (base / "timeline.npz").is_file()):
            print(json.dumps({"video_id": video_id, "status": "missing_features"}), flush=True)
            continue
        predictions, duration = scan_video(
            model, feature_base=base, video_id=video_id, device=device, batch_size=args.batch_size,
        )
        rows.extend(predictions); durations[video_id] = duration
        print(json.dumps({"video_id": video_id, "progress": [position + 1, len(video_ids)], "hypotheses": len(predictions)}), flush=True)
    label_index = {label: index for index, label in enumerate(LABELS)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("football.goal_retriever_candidates.v1"),
        checkpoint=np.asarray(str(args.checkpoint.resolve())), split=np.asarray(args.split),
        video_ids=np.asarray([row["video_id"] for row in rows]),
        core_starts=np.asarray([row["core_start"] for row in rows], dtype=np.float32),
        slots=np.asarray([row["slot"] for row in rows], dtype=np.int16),
        labels=np.asarray([label_index[row["label"]] for row in rows], dtype=np.int8),
        times=np.asarray([row["time"] for row in rows], dtype=np.float32),
        scores=np.asarray([row["score"] for row in rows], dtype=np.float32),
        durations=np.asarray([row["duration"] for row in rows], dtype=np.float16),
        uncertainty=np.asarray([row["uncertainty"] for row in rows], dtype=np.float16),
        duration_video_ids=np.asarray(list(durations)),
        video_durations=np.asarray(list(durations.values()), dtype=np.float32),
    )
    done = args.output.with_suffix(".done.json")
    done.write_text(json.dumps({
        "rank": args.rank, "world_size": args.world_size, "videos": len(durations),
        "hypotheses": len(rows), "output": str(args.output), "no_nms": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

