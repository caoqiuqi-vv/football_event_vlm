from __future__ import annotations

"""Diagnose whether LF-A0 is limited by features or peak decoding.

This script is deliberately calibration-only.  It can be sharded across idle GPUs
and writes per-video evidence that is later merged with ``--aggregate``.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.annotations import LABEL_TO_FAMILY, load_events  # noqa: E402
from football_longform_v2.canonical import load_canonical_split  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.decoding import (  # noqa: E402
    Proposal,
    decode_family_conditioned_classes,
    decode_proposals,
)
from football_longform_v2.evaluation import (  # noqa: E402
    events_for_label,
    infer_continuous_timeline,
    scored_point_matches,
)
from football_longform_v2.feature_store import load_aligned_npz  # noqa: E402
from football_longform_v2.models import TemporalLocator  # noqa: E402
from football_longform_v2.schema import read_video_ids  # noqa: E402


TOLERANCES = (1.0, 2.0, 3.0, 5.0, 10.0)
NMS_RADII = (0.5, 1.0, 2.0, 3.0, 6.0)


def resolve(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "p50": None, "p90": None, "max": None}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "p10": float(torch.quantile(tensor, 0.10)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "max": float(tensor.max()),
    }


def nearest_distances(candidates: list[Proposal], events: list[float]) -> list[float]:
    times = [item.timestamp for item in candidates]
    if not times:
        return [math.inf for _ in events]
    return [min(abs(event - candidate) for candidate in times) for event in events]


def route_metrics(candidates: list[Proposal], events: list[float], duration: float) -> dict:
    distances = nearest_distances(candidates, events)
    recall = {}
    for tolerance in TOLERANCES:
        _, labels, _ = scored_point_matches(
            candidates, events, tolerance_seconds=tolerance
        )
        recall[str(tolerance)] = sum(labels) / len(events) if events else None
    finite = [value for value in distances if math.isfinite(value)]
    return {
        "candidate_count": len(candidates),
        "candidates_per_minute": len(candidates) / max(duration / 60.0, 1.0 / 60.0),
        "one_to_one_recall": recall,
        "event_to_nearest_candidate_seconds": quantiles(finite),
        "event_nearest_distances_seconds": distances,
    }


def uniform_candidates(timestamps: torch.Tensor, count: int, label: str) -> list[Proposal]:
    if count <= 0 or timestamps.numel() == 0:
        return []
    indices = torch.linspace(0, timestamps.numel() - 1, count + 2)[1:-1].round().long()
    return [
        Proposal(0, label, float(timestamps[index]), 0.5, int(index))
        for index in indices.tolist()
    ]


def dense_contrast(
    scores: torch.Tensor, timestamps: torch.Tensor, events: list[float]
) -> dict[str, float | None]:
    if not events:
        return {
            "event_max_2s_p10": None,
            "event_max_2s_p50": None,
            "event_max_2s_p90": None,
            "background_p50": None,
            "background_p90": None,
            "event_p10_minus_background_p90": None,
        }
    event_scores = []
    near_any = torch.zeros_like(timestamps, dtype=torch.bool)
    for event in events:
        near = (timestamps - float(event)).abs() <= 2.0
        event_scores.append(float(scores[near].max()) if near.any() else 0.0)
        near_any |= (timestamps - float(event)).abs() <= 5.0
    background = scores[~near_any]
    event_tensor = torch.tensor(event_scores)
    background_p50 = float(torch.quantile(background, 0.50)) if background.numel() else None
    background_p90 = float(torch.quantile(background, 0.90)) if background.numel() else None
    event_p10 = float(torch.quantile(event_tensor, 0.10))
    return {
        "event_max_2s_p10": event_p10,
        "event_max_2s_p50": float(torch.quantile(event_tensor, 0.50)),
        "event_max_2s_p90": float(torch.quantile(event_tensor, 0.90)),
        "background_p50": background_p50,
        "background_p90": background_p90,
        "event_p10_minus_background_p90": (
            event_p10 - background_p90 if background_p90 is not None else None
        ),
    }


def decode_direct_class_peaks(
    logits: torch.Tensor,
    timestamps: torch.Tensor,
    labels: tuple[str, ...],
    radius: float,
) -> dict[str, list[Proposal]]:
    decoded = decode_proposals(
        logits.unsqueeze(0),
        timestamps.unsqueeze(0),
        labels,
        threshold=0.0,
        nms_radius_seconds={label: radius for label in labels},
        max_per_minute={label: 1000.0 for label in labels},
    )[0]
    return {
        label: [proposal for proposal in decoded if proposal.family == label]
        for label in labels
    }


def aggregate(shards: list[Path], output: Path) -> None:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    per_video = [item for payload in payloads for item in payload["per_video"]]
    video_ids = [item["video_id"] for item in per_video]
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError("diagnostic shards contain duplicate videos")
    labels = tuple(payloads[0]["labels"])
    summary: dict[str, dict] = {}
    for label in labels:
        support = sum(item["labels"][label]["support"] for item in per_video)
        routes = sorted(per_video[0]["labels"][label]["routes"])
        route_summary = {}
        for route in routes:
            rows = [item["labels"][label]["routes"][route] for item in per_video]
            distances = [
                value for row in rows for value in row["event_nearest_distances_seconds"]
                if math.isfinite(value)
            ]
            route_summary[route] = {
                "candidate_count": sum(row["candidate_count"] for row in rows),
                "candidates_per_minute": (
                    sum(row["candidate_count"] for row in rows)
                    / max(sum(item["duration_seconds"] for item in per_video) / 60.0, 1e-9)
                ),
                "independent_event_recall": {
                    str(tolerance): (
                        sum(value <= tolerance for value in distances) / support if support else None
                    )
                    for tolerance in TOLERANCES
                },
                "event_to_nearest_candidate_seconds": quantiles(distances),
            }
        contrast_rows = [item["labels"][label]["dense_contrast"] for item in per_video]
        weighted_gaps = [
            (row["event_p10_minus_background_p90"], item["labels"][label]["support"])
            for row, item in zip(contrast_rows, per_video)
            if row["event_p10_minus_background_p90"] is not None
        ]
        summary[label] = {
            "support": support,
            "routes": route_summary,
            "mean_per_video_event_p10_minus_background_p90": (
                sum(value * weight for value, weight in weighted_gaps)
                / sum(weight for _, weight in weighted_gaps)
                if weighted_gaps else None
            ),
        }
    result = {
        "schema_version": "football_longform_v2.a0_failure_diagnostic.v1",
        "evaluation_split": "calibration",
        "evaluated_video_count": len(per_video),
        "evaluated_video_ids": sorted(video_ids),
        "checkpoint": payloads[0]["checkpoint"],
        "labels": summary,
        "per_video": sorted(per_video, key=lambda item: item["video_id"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "evaluated_video_count": len(per_video), "labels": summary}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregate", nargs="*")
    args = parser.parse_args()
    output_path = Path(args.output).expanduser().resolve()
    if args.aggregate is not None:
        if not args.aggregate:
            raise ValueError("--aggregate requires one or more shard JSON files")
        aggregate([Path(path).expanduser().resolve() for path in args.aggregate], output_path)
        return
    if not args.config or not args.checkpoint:
        raise ValueError("--config and --checkpoint are required outside aggregate mode")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")

    config = load_config(args.config)
    root = Path(config["_project_root"])
    paths = config["paths"]
    canonical = load_canonical_split(resolve(root, paths["canonical_manifest"]), "calibration")
    all_ids = read_video_ids(resolve(root, paths["calibration_ids"]))
    if tuple(all_ids) != tuple(canonical.media_ids):
        raise RuntimeError("canonical manifest and calibration ID list disagree")
    ids = all_ids[args.shard_index :: args.num_shards]
    feature_root = resolve(root, paths["feature_store"]) / "calibration"
    annotation_root = resolve(root, paths["annotations"])
    labels = tuple(config["task"]["output_labels"])
    families = tuple(config["task"]["proposal_families"])
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TemporalLocator.from_config(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    timeline_hz = float(config["features"]["timeline_hz"])
    decode = config["decode"]
    per_video = []
    for video_id in ids:
        timeline = load_aligned_npz(feature_root / video_id / "timeline.npz")
        events = load_events(
            annotation_root / f"{canonical.annotation_id_by_media_id[video_id]}.json"
        )
        output = infer_continuous_timeline(
            model,
            timeline,
            device=device,
            core_steps=int(round(120.0 * timeline_hz)),
            context_steps=int(round(30.0 * timeline_hz)),
        )
        if output.class_logits is None:
            raise RuntimeError("checkpoint did not emit class logits")
        conditioned = decode_family_conditioned_classes(
            output.logits,
            output.class_logits,
            timeline.timestamps,
            families,
            labels,
            label_to_family=LABEL_TO_FAMILY,
            family_nms_radius_seconds=decode["nms_radius_seconds"],
            class_nms_radius_seconds={
                label: 3.0 if label in {"shot", "save"} else 6.0 for label in labels
            },
            local_search_radius_seconds={
                label: 2.0 if label == "shot" else 4.0 for label in labels
            },
            max_class_per_minute={label: 1000.0 for label in labels},
        )
        direct_by_radius = {
            radius: decode_direct_class_peaks(
                output.class_logits, timeline.timestamps, labels, radius
            )
            for radius in NMS_RADII
        }
        global_any = decode_proposals(
            output.class_logits.max(dim=-1, keepdim=True).values.unsqueeze(0),
            timeline.timestamps.unsqueeze(0),
            ("any_event",),
            threshold=0.0,
            nms_radius_seconds={"any_event": 0.5},
            max_per_minute={"any_event": 1000.0},
        )[0]
        item = {
            "video_id": video_id,
            "duration_seconds": float(timeline.timestamps[-1]),
            "labels": {},
        }
        for label_index, label in enumerate(labels):
            event_times = events_for_label(events, label)
            routes = {"family_conditioned_current": conditioned[label]}
            routes["shared_any_class_nms_0.5s"] = global_any
            for radius in NMS_RADII:
                routes[f"direct_class_nms_{radius:g}s"] = direct_by_radius[radius][label]
            current_count = len(conditioned[label])
            routes["uniform_same_count"] = uniform_candidates(
                timeline.timestamps, current_count, label
            )
            item["labels"][label] = {
                "support": len(event_times),
                "dense_contrast": dense_contrast(
                    output.class_logits[:, label_index].sigmoid(),
                    timeline.timestamps,
                    event_times,
                ),
                "routes": {
                    name: route_metrics(candidates, event_times, item["duration_seconds"])
                    for name, candidates in routes.items()
                },
            }
        per_video.append(item)
        print(f"device={device} shard={args.shard_index}/{args.num_shards} video={video_id}", flush=True)
    result = {
        "schema_version": "football_longform_v2.a0_failure_diagnostic.shard.v1",
        "evaluation_split": "calibration",
        "checkpoint": str(checkpoint),
        "labels": list(labels),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "per_video": per_video,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
