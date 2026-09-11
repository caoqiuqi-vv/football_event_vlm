from __future__ import annotations

"""Run a frozen A2 locator on an unlabeled cached VideoMAE timeline."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.chunk_evaluation import decode_class_peaks  # noqa: E402
from football_longform_v2.models import SequentialChunkLocator  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--operating-points", required=True)
    parser.add_argument("--timeline-dir", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    operating_path = Path(args.operating_points).expanduser().resolve()
    timeline_dir = Path(args.timeline_dir).expanduser().resolve()
    operating = json.loads(operating_path.read_text(encoding="utf-8"))
    if operating.get("schema") != "football_longform_v2.a2_operating_points.v1":
        raise ValueError("unsupported operating point schema")
    if sha256_file(checkpoint_path) != operating["checkpoint_sha256"]:
        raise RuntimeError("checkpoint does not match frozen operating points")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
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
    model.to(device).eval()
    features = torch.from_numpy(np.array(
        np.load(timeline_dir / "features.npy", mmap_mode="r", allow_pickle=False),
        dtype=np.float32, copy=True,
    ))
    timestamps = torch.from_numpy(np.array(
        np.load(timeline_dir / "timestamps.npy", mmap_mode="r", allow_pickle=False),
        dtype=np.float32, copy=True,
    ))
    if features.ndim != 2 or features.shape[0] != timestamps.numel():
        raise RuntimeError("invalid cached timeline arrays")
    valid = torch.ones(1, timestamps.numel(), dtype=torch.bool, device=device)
    chunk_phase = torch.arange(
        timestamps.numel(), device=device, dtype=torch.long
    ).remainder(int(model_config["tubelets_per_chunk"])).unsqueeze(0)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model(features.unsqueeze(0).to(device), valid, chunk_phase)
    labels = tuple(checkpoint["labels"])
    proposals = decode_class_peaks(
        outputs["class_logits"][0].float().cpu(),
        outputs["offsets"][0].float().cpu(),
        timestamps,
        labels,
        nms_radius_seconds=operating["nms_radius_seconds"],
    )
    events = []
    per_class = {}
    for label in labels:
        threshold = float(operating["thresholds"][label])
        selected = [item for item in proposals[label] if item.score >= threshold]
        per_class[label] = len(selected)
        events.extend({
            "video_id": args.video_id,
            "label": label,
            "timestamp": item.timestamp,
            "score": item.score,
            "threshold": threshold,
            "timeline_index": item.timeline_index,
            "requires_human_confirmation": True,
        } for item in selected)
    events.sort(key=lambda item: (item["timestamp"], item["label"]))
    payload = {
        "schema": "football_longform_v2.a2_predictions.v1",
        "video_id": args.video_id,
        "checkpoint_sha256": operating["checkpoint_sha256"],
        "operating_points_sha256": sha256_file(operating_path),
        "timeline_dir": str(timeline_dir),
        "tubelet_count": int(timestamps.numel()),
        "event_count": len(events),
        "per_class": per_class,
        "events": events,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "event_count": len(events), "per_class": per_class,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
