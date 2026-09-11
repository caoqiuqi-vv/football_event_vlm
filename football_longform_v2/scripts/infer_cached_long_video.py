from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.annotations import LABEL_TO_FAMILY
from football_longform_v2.config import load_config
from football_longform_v2.decoding import decode_family_conditioned_classes
from football_longform_v2.deployment import (
    OPERATING_POINTS_SCHEMA,
    apply_frozen_operating_points,
)
from football_longform_v2.evaluation import infer_continuous_timeline
from football_longform_v2.feature_store import load_aligned_npz, read_timeline_provenance
from football_longform_v2.models import TemporalLocator
from football_longform_v2.models.temporal_locator import MODEL_SCHEMA


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer one cached long video with frozen calibration thresholds."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--operating-points", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--core-seconds", type=float, default=120.0)
    parser.add_argument("--context-seconds", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    operating_path = Path(args.operating_points).expanduser().resolve()
    operating = json.loads(operating_path.read_text(encoding="utf-8"))
    if operating.get("schema") != OPERATING_POINTS_SCHEMA:
        raise ValueError("unsupported operating-points schema")
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint else Path(operating["checkpoint"]).expanduser().resolve()
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != operating.get("checkpoint_sha256"):
        raise RuntimeError("checkpoint hash disagrees with frozen operating points")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("model_schema") != MODEL_SCHEMA:
        raise RuntimeError("checkpoint model schema mismatch")

    config_sha256 = hashlib.sha256(Path(config["_config_path"]).read_bytes()).hexdigest()
    if payload.get("feature_config_sha256") != config_sha256:
        raise RuntimeError("checkpoint and runtime config hashes disagree")
    timeline_path = Path(args.timeline).expanduser().resolve()
    provenance = read_timeline_provenance(timeline_path)
    context_config = config["features"]["context"]
    expected_cache = {
        "backbone_arch": str(context_config["arch"]),
        "backbone_id": str(context_config["backbone_id"]),
        "config_sha256": config_sha256,
        "weights_sha256": str(payload.get("feature_weights_sha256")),
    }
    cache_errors = {
        key: {"expected": value, "actual": provenance.get(key)}
        for key, value in expected_cache.items() if provenance.get(key) != value
    }
    if cache_errors:
        raise RuntimeError(f"timeline provenance mismatch: {cache_errors}")

    device = torch.device(args.device)
    model = TemporalLocator.from_config(config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    timeline = load_aligned_npz(timeline_path)
    timeline_hz = float(config["features"]["timeline_hz"])
    output = infer_continuous_timeline(
        model, timeline, device=device,
        core_steps=int(round(args.core_seconds * timeline_hz)),
        context_steps=int(round(args.context_seconds * timeline_hz)),
    )
    if output.class_logits is None:
        raise RuntimeError("model did not emit class logits")
    labels = tuple(config["task"]["output_labels"])
    families = tuple(config["task"]["proposal_families"])
    local_radius = operating.get("class_local_search_radius_seconds") or {
        label: (2.0 if label == "shot" else 4.0) for label in labels
    }
    proposals = decode_family_conditioned_classes(
        output.logits, output.class_logits, timeline.timestamps, families, labels,
        label_to_family=LABEL_TO_FAMILY,
        family_nms_radius_seconds=config["decode"]["nms_radius_seconds"],
        class_nms_radius_seconds={
            label: (3.0 if label in {"shot", "save"} else 6.0) for label in labels
        },
        local_search_radius_seconds={label: float(local_radius[label]) for label in labels},
        max_class_per_minute={label: 1000.0 for label in labels},
    )
    duration_seconds = float(timeline.timestamps[-1]) if timeline.timestamps.numel() else 0.0
    events = apply_frozen_operating_points(
        proposals, operating, duration_minutes=max(duration_seconds / 60.0, 1.0 / 60.0),
        video_id=args.video_id,
    )
    counts = Counter(event["label"] for event in events)
    result = {
        "schema": "football_longform_v2.predictions.v1",
        "video_id": args.video_id, "timeline": str(timeline_path),
        "duration_seconds": duration_seconds, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "operating_points": str(operating_path),
        "external_labels_used": False,
        "event_count": len(events), "event_counts": dict(sorted(counts.items())),
        "events": events,
    }
    result_path = Path(args.output).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(result_path), "event_count": len(events),
        "event_counts": result["event_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
