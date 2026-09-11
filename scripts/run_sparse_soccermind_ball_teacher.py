#!/usr/bin/env python
"""Run one SoccerMind RF-DETR model over sparse event-window intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sampling-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-intervals", type=int, default=0)
    args = parser.parse_args()

    soccer_root = Path("/home/new_users/qiuqi/code/SoccerMind-main")
    engine_root = soccer_root / "engines" / "rf_detr"
    projects_root = soccer_root / "projects"
    for path in (engine_root, projects_root, soccer_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import inference_rf_detr_model as inference

    config_path = Path(args.config).resolve()
    config = inference.load_yaml(config_path)
    # Pseudo-label generation needs detector coordinates only.
    config.setdefault("inference", {}).setdefault("video", {})["save_video"] = False
    config["inference"].setdefault("canonical_output", {})["enabled"] = False
    config["inference"].setdefault("tracking", {})["enabled"] = False
    if "stationary_football_filter" in config:
        config["stationary_football_filter"]["enabled"] = False
    categories = inference.build_categories(config)
    prediction_config = inference.build_prediction_config(config, categories)
    model = inference.load_rfdetr_model(config)

    manifest = json.loads(Path(args.sampling_manifest).read_text())
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / f"shard_{args.shard_index:02d}_state.json"
    completed: set[str] = set()
    if state_path.is_file():
        completed = set(json.loads(state_path.read_text()).get("completed", []))
    processed = 0
    image_id = args.shard_index * 1_000_000_000 + 1
    for video_index, video in enumerate(manifest["videos"]):
        if video_index % args.num_shards != args.shard_index:
            continue
        video_id = str(video["video_id"])
        video_output = output_root / video_id
        video_output.mkdir(parents=True, exist_ok=True)
        output_path = video_output / "football_predictions.jsonl"
        for interval_index, interval in enumerate(video["intervals"]):
            interval_key = f"{video_id}:{interval_index}"
            if interval_key in completed:
                continue
            if args.max_intervals and processed >= args.max_intervals:
                return
            video_cfg: dict[str, Any] = dict(config["inference"]["video"])
            video_cfg.update(
                {
                    "start_time": float(interval["start_sec"]),
                    "end_time": float(interval["end_sec"]),
                    "max_seconds": "all",
                    "save_video": False,
                    "streaming": True,
                }
            )
            item = inference.SourceItem(
                source=str(video["video_path"]),
                kind="video",
                local_path=Path(video["video_path"]),
            )
            predictions, _, image_id = inference.predict_video_file(
                item,
                image_id,
                model,
                prediction_config,
                categories,
                video_output,
                [],
                video_cfg,
                tracking_config=None,
            )
            with output_path.open("a", encoding="utf-8") as handle:
                for prediction in predictions:
                    if int(prediction.get("category_id", -1)) != 0:
                        continue
                    row = {
                        "frame_index": int(prediction["frame_index"]),
                        "timestamp_seconds": float(prediction["timestamp_seconds"]),
                        "category_id": 0,
                        "category_name": "football",
                        "score": float(prediction.get("score", 0.0)),
                        "xywh": [float(value) for value in prediction["bbox"]],
                        "teacher": "SoccerMind RF-DETR Medium P2 SAHI160 Recheck320",
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            completed.add(interval_key)
            processed += 1
            state_path.write_text(
                json.dumps(
                    {
                        "shard_index": args.shard_index,
                        "num_shards": args.num_shards,
                        "completed": sorted(completed),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            print(
                f"shard={args.shard_index}/{args.num_shards} "
                f"video={video_id} interval={interval_index + 1}/{len(video['intervals'])} "
                f"detections={len(predictions)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
