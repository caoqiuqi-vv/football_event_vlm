#!/usr/bin/env python
"""Export per-core retriever memory/slots for ranking and Examiner training."""

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
from football_e2e_spotter.goal_retriever_eval import load_timeline  # noqa: E402


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
    args = parse_args(); device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["model_config"]
    model = GoalLongContextRetriever(
        config["appearance_dim"], config["motion_dim"], config["audio_dim"], hidden_dim=config["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval(); geometry = model.geometry
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    video_ids = [str(item["media_id"]) for item in payload[args.split]][args.rank::args.world_size]
    all_video, all_start, all_score, all_time, all_duration, all_uncertainty = [], [], [], [], [], []
    all_quality, all_slots, all_memory, duration_ids, video_durations = [], [], [], [], []
    for video_position, video_id in enumerate(video_ids):
        base = args.feature_root / args.split / video_id
        if not ((base / "metadata.json").is_file() or (base / "timeline.npz").is_file()):
            continue
        _timestamps, appearance, motion, audio = load_timeline(base)
        video_duration = float(len(appearance)); core_starts = np.arange(0.0, video_duration, geometry.core_seconds, dtype=np.float32)
        duration_ids.append(video_id); video_durations.append(video_duration)
        for begin in range(0, len(core_starts), args.batch_size):
            starts = core_starts[begin:begin + args.batch_size]
            streams = [[], [], []]; valid_rows = []
            for core_start in starts:
                context_start = float(core_start) - geometry.context_left_seconds
                source = np.floor(context_start + np.arange(geometry.context_steps)).astype(np.int64)
                valid = (source >= 0) & (source < len(appearance)); clipped = np.clip(source, 0, max(len(appearance) - 1, 0))
                for stream_index, values in enumerate((appearance, motion, audio)):
                    value = np.asarray(values[clipped], dtype=np.float32); value[~valid] = 0.0; streams[stream_index].append(value)
                valid_rows.append(valid)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    torch.from_numpy(np.stack(streams[0])).to(device),
                    torch.from_numpy(np.stack(streams[1])).to(device),
                    torch.from_numpy(np.stack(streams[2])).to(device),
                    torch.from_numpy(np.stack(valid_rows)).to(device),
                )
                probability = output["class_logits"].softmax(dim=-1)[..., :len(LABELS)]
                quality = output["quality_logit"].sigmoid()
                score = probability * quality.unsqueeze(-1).sqrt()
            batch_count = len(starts)
            all_video.extend([video_id] * batch_count); all_start.extend(starts.tolist())
            all_score.append(score.float().cpu().numpy().astype(np.float16))
            all_time.append((output["event_time"] + torch.as_tensor(starts, device=device)[:, None]).float().cpu().numpy())
            all_duration.append(output["region_duration"].float().cpu().numpy().astype(np.float16))
            all_uncertainty.append(output["temporal_uncertainty"].float().cpu().numpy().astype(np.float16))
            all_quality.append(quality.float().cpu().numpy().astype(np.float16))
            all_slots.append(output["slot_embedding"].float().cpu().numpy().astype(np.float16))
            all_memory.append(output["memory"].float().cpu().numpy().astype(np.float16))
        print(json.dumps({"video_id": video_id, "progress": [video_position + 1, len(video_ids)], "cores": len(core_starts)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, schema=np.asarray("football.goal_retriever_evidence.v1"),
        checkpoint=np.asarray(str(args.checkpoint.resolve())), split=np.asarray(args.split), labels=np.asarray(LABELS),
        video_ids=np.asarray(all_video), core_starts=np.asarray(all_start, dtype=np.float32),
        class_scores=np.concatenate(all_score, axis=0), event_times=np.concatenate(all_time, axis=0),
        region_durations=np.concatenate(all_duration, axis=0), uncertainty=np.concatenate(all_uncertainty, axis=0),
        quality=np.concatenate(all_quality, axis=0), slot_embeddings=np.concatenate(all_slots, axis=0),
        memory=np.concatenate(all_memory, axis=0), duration_video_ids=np.asarray(duration_ids),
        video_durations=np.asarray(video_durations, dtype=np.float32),
    )
    args.output.with_suffix(".done.json").write_text(json.dumps({
        "rank": args.rank, "world_size": args.world_size, "videos": len(duration_ids),
        "cores": len(all_video), "output": str(args.output), "no_nms": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

