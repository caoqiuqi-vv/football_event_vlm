from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.chunk_evaluation import evaluate_sequential_chunk_locator  # noqa: E402
from football_longform_v2.models import SequentialChunkLocator  # noqa: E402
from train_chunk_locator import dataset_from_config, load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_yaml(Path(args.config).expanduser().resolve())
    checkpoint = torch.load(
        Path(args.checkpoint).expanduser().resolve(), map_location="cpu", weights_only=False, mmap=True
    )
    dataset = dataset_from_config(config, "calibration")
    model_config = checkpoint["model_config"]
    model = SequentialChunkLocator(
        input_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dilations=tuple(model_config["dilations"]),
        dropout=float(model_config["dropout"]),
        shot_history_steps=int(model_config["shot_history_steps"]),
        max_offset_seconds=float(model_config["max_offset_seconds"]),
        detach_shot_context=bool(model_config["detach_shot_context"]),
        tubelets_per_chunk=int(model_config["tubelets_per_chunk"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    evaluation = config["evaluation"]
    deployment_gate = config["deployment_gate"]
    report = evaluate_sequential_chunk_locator(
        model, dataset, device=device,
        nms_radius_seconds=evaluation["nms_radius_seconds"],
        target_recall=evaluation["target_recall"],
        minimum_precision_at_target_recall=deployment_gate[
            "minimum_precision_at_target_recall"
        ],
        maximum_fp_per_minute=deployment_gate["maximum_fp_per_minute"],
        tolerances_seconds=tuple(float(item) for item in evaluation["tolerances_seconds"]),
        operating_tolerance_seconds=float(evaluation["operating_tolerance_seconds"]),
        uniform_baseline_intervals_seconds=tuple(
            float(item) for item in evaluation.get(
                "uniform_baseline_intervals_seconds", (3.0, 4.0)
            )
        ),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "selection_tuple": report["selection_tuple"],
        "video_count": report["video_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
