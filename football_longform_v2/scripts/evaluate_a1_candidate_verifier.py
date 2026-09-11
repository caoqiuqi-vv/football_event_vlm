from __future__ import annotations

"""Evaluate an A1 RGB verifier on the A0 shared proposal stream.

This is the end-to-end calibration protocol missing from ordinary
event-centred clip validation.  Every A0 candidate is decoded once, A1 emits a
class score and a frame-level refined timestamp, then class-wise point NMS and
strict one-to-one matching are applied over each complete long video.

Only the canonical calibration split is accepted.  The sealed third-party
test set is deliberately unsupported until the model and thresholds are
frozen.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_football_events as legacy  # noqa: E402
from football_longform_v2.annotations import load_events  # noqa: E402
from football_longform_v2.decoding import Proposal  # noqa: E402
from football_longform_v2.evaluation import (  # noqa: E402
    average_precision,
    events_for_label,
    operating_point_at_recall,
    scored_point_matches,
)


CANDIDATE_SCHEMA = "football_longform_v2.shared_candidate_manifest.v1"
REPORT_SCHEMA = "football_longform_v2.a1_candidate_verifier_report.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centered_window(anchor: float, duration: float, video_duration: float) -> tuple[float, float]:
    start = max(min(anchor - 0.5 * duration, max(video_duration - duration, 0.0)), 0.0)
    return start, min(start + duration, video_duration)


def quantile(values: Iterable[float], q: float) -> float | None:
    data = list(values)
    if not data:
        return None
    return float(torch.quantile(torch.tensor(data, dtype=torch.float32), q))


def load_candidate_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CANDIDATE_SCHEMA:
        raise ValueError("unsupported candidate manifest schema")
    if payload.get("split") != "calibration":
        raise ValueError(
            "A1 verifier threshold selection is calibration-only; "
            f"got split={payload.get('split')}"
        )
    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("candidate manifest contains no videos")
    ids = [str(video.get("video_id", "")) for video in videos]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("candidate manifest has empty or duplicate video IDs")
    return payload


def repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def build_records(
    cfg: Any, manifest: dict[str, Any]
) -> tuple[list[legacy.LongVideoRecord], dict[tuple[str, str], list[legacy.FootballEvent]]]:
    annotation_root = repo_path(str(cfg.data.long_video.roots[0].annotations_dir))
    clip_duration = float(cfg.video.clip_duration)
    records: list[legacy.LongVideoRecord] = []
    events_by_video: dict[tuple[str, str], list[legacy.FootballEvent]] = {}
    source = "a0_shared_candidate_stream"
    for video in manifest["videos"]:
        video_id = str(video["video_id"])
        video_path = Path(str(video["source_video"]))
        if not video_path.is_file():
            raise FileNotFoundError(f"candidate source video is missing: {video_path}")
        annotation_path = annotation_root / f"{video_id}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"canonical annotation is missing: {annotation_path}")
        duration = float(video["duration_seconds"])
        events = legacy.load_annotation_events(annotation_path, source, video_id)
        events_by_video[(source, video_id)] = events
        for candidate in video["candidates"]:
            timestamp = float(candidate["timestamp"])
            start, end = centered_window(timestamp, clip_duration, duration)
            records.append(
                legacy.LongVideoRecord(
                    source=source,
                    split="calibration_candidates",
                    video_id=video_id,
                    sample_id=str(candidate["candidate_id"]),
                    video_path=str(video_path),
                    annotation_path=str(annotation_path),
                    anchor_time=timestamp,
                    base_clip_start=start,
                    base_clip_end=end,
                    video_duration=duration,
                    is_negative=False,
                    labels=tuple(0.0 for _ in legacy.LABELS),
                    label_mask=tuple(1.0 for _ in legacy.LABELS),
                )
            )
    return records, events_by_video


def build_dataset(
    cfg: Any,
    records: list[legacy.LongVideoRecord],
    events_by_video: dict[tuple[str, str], list[legacy.FootballEvent]],
) -> legacy.FootballLongVideoDataset:
    image_size = legacy.parse_image_size(cfg.video.image_size)
    normalize_on_cpu = not bool(
        cfg.data.get("preprocessing", {}).get("normalize_on_device", False)
    )
    return legacy.FootballLongVideoDataset(
        records,
        events_by_video,
        num_frames=legacy.effective_num_frames(cfg),
        image_size=image_size,
        clip_duration=float(cfg.video.clip_duration),
        sampling_duration=float(
            cfg.video.get("sampling_duration", cfg.video.clip_duration)
        ),
        sampling_temporal_jitter_sec=0.0,
        event_margin=float(cfg.video.get("event_margin", 0.5)),
        temporal_jitter_sec=0.0,
        is_train=False,
        hflip_prob=0.0,
        crop_provider=None,
        detector_view_mode="roi_only",
        global_image_size=image_size,
        normalize_on_cpu=normalize_on_cpu,
        decode_strategy=str(cfg.data.get("video_decode_strategy", "single_seek")),
        video_reader_cache_size=int(cfg.data.get("video_reader_cache_size", 2)),
        frame_supervision=True,
        frame_label_sigma_sec=legacy.frame_label_sigma_seconds(cfg),
        frame_label_ignore_radius_sec=legacy.frame_label_ignore_radius_seconds(cfg),
        positive_window_strategy="centered",
        positive_anchor_min_sec=float(cfg.video.get("event_margin", 0.5)),
        positive_anchor_max_sec=float(
            float(cfg.video.clip_duration) - float(cfg.video.get("event_margin", 0.5))
        ),
    )


def class_nms(rows: list[dict[str, Any]], radius: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        if all(
            abs(float(row["timestamp"]) - float(other["timestamp"])) > radius
            for other in kept
        ):
            kept.append(row)
    return sorted(kept, key=lambda item: float(item["timestamp"]))


def proposals_from_rows(rows: list[dict[str, Any]], label: str) -> list[Proposal]:
    return [
        Proposal(
            batch_index=0,
            family=label,
            timestamp=float(row["timestamp"]),
            score=float(row["score"]),
            timeline_index=index,
        )
        for index, row in enumerate(rows)
    ]


def evaluate_rows(
    rows_by_video_label: dict[tuple[str, str], list[dict[str, Any]]],
    events_by_video_label: dict[tuple[str, str], list[float]],
    labels: tuple[str, ...],
    durations: dict[str, float],
    *,
    tolerance: float,
    target_recall: dict[str, float],
) -> dict[str, Any]:
    total_minutes = sum(durations.values()) / 60.0
    result: dict[str, Any] = {}
    video_ids = sorted(durations)
    for label in labels:
        scores: list[float] = []
        matches: list[bool] = []
        support = 0
        for video_id in video_ids:
            events = events_by_video_label.get((video_id, label), [])
            support += len(events)
            proposals = proposals_from_rows(
                rows_by_video_label.get((video_id, label), []), label
            )
            video_scores, video_matches, _ = scored_point_matches(
                proposals, events, tolerance_seconds=tolerance
            )
            scores.extend(video_scores)
            matches.extend(video_matches)
        positives = [score for score, matched in zip(scores, matches) if matched]
        negatives = [score for score, matched in zip(scores, matches) if not matched]
        result[label] = {
            "support": support,
            "proposal_count": len(scores),
            "candidate_ceiling_recall": sum(matches) / support if support else None,
            "average_precision": average_precision(
                torch.tensor(scores), torch.tensor(matches), positive_count=support
            ),
            "operating_point": operating_point_at_recall(
                scores,
                matches,
                positive_count=support,
                target_recall=float(target_recall[label]),
                total_minutes=total_minutes,
            ),
            "positive_score_p10": quantile(positives, 0.10),
            "positive_score_p50": quantile(positives, 0.50),
            "negative_score_p90": quantile(negatives, 0.90),
            "positive_p10_minus_negative_p90": (
                quantile(positives, 0.10) - quantile(negatives, 0.90)
                if positives and negatives
                else None
            ),
        }
    return result


def source_candidate_ceiling(
    manifest: dict[str, Any],
    events_by_video_label: dict[tuple[str, str], list[float]],
    labels: tuple[str, ...],
    tolerance: float,
) -> dict[str, Any]:
    totals = {label: {"support": 0, "matched": 0} for label in labels}
    for video in manifest["videos"]:
        video_id = str(video["video_id"])
        source_proposals = [
            Proposal(
                batch_index=0,
                family="any_event",
                timestamp=float(candidate["timestamp"]),
                score=float(candidate["source_score"]),
                timeline_index=int(candidate["timeline_index"]),
            )
            for candidate in video["candidates"]
        ]
        for label in labels:
            events = events_by_video_label.get((video_id, label), [])
            _, matches, _ = scored_point_matches(
                source_proposals, events, tolerance_seconds=tolerance
            )
            totals[label]["support"] += len(events)
            totals[label]["matched"] += sum(matches)
    return {
        label: {
            **totals[label],
            "recall": (
                totals[label]["matched"] / totals[label]["support"]
                if totals[label]["support"]
                else None
            ),
        }
        for label in labels
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-ids", default="")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--score-fusion", choices=("clip", "clip_x_frame_peak"),
        default="clip_x_frame_peak",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    candidate_path = Path(args.candidates).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    cfg = legacy.load_config(str(config_path), [])
    legacy.configure_label_schema(cfg)
    if tuple(legacy.LABELS) != ("shot", "save", "corner", "freekick", "penalty"):
        raise ValueError(f"A1 verifier requires five_way labels, got {legacy.LABELS}")
    manifest = load_candidate_manifest(candidate_path)
    records, legacy_events = build_records(cfg, manifest)
    dataset = build_dataset(cfg, records, legacy_events)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=legacy.football_collate,
    )

    device = torch.device(args.device)
    model = legacy.make_model(cfg, use_cached_features=False, device=device)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    state = legacy.strip_module_prefix(checkpoint["model"])
    model.load_state_dict(state, strict=True)
    device_ids = [int(item) for item in args.device_ids.split(",") if item.strip()]
    if device_ids:
        if device.type != "cuda" or device.index != device_ids[0]:
            raise ValueError("--device must equal the first --device-ids entry")
        model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    model.eval()

    labels = tuple(legacy.LABELS)
    raw_rows_by_time_mode: dict[
        str, dict[tuple[str, str], list[dict[str, Any]]]
    ] = {
        "source": defaultdict(list),
        "frame_peak": defaultdict(list),
    }
    decoded = 0
    decode_failures = 0
    for batch in loader:
        with legacy.autocast_context(
            device, bool(cfg.train.amp), str(cfg.train.amp_dtype)
        ):
            outputs = legacy.forward_model_batch(
                model, batch, device, return_aux=True
            )
        if not isinstance(outputs, dict) or "frame_event_logits" not in outputs:
            raise RuntimeError("A1 checkpoint did not emit frame-level logits")
        clip_probs = outputs["logits"].float().sigmoid().cpu()
        frame_logits = outputs["frame_event_logits"].float().cpu()
        frame_probs = frame_logits.sigmoid()
        peak_indices = frame_logits.argmax(dim=1)
        frame_times = batch["frame_times"].float()
        peak_times = torch.gather(frame_times, 1, peak_indices)
        peak_probs = torch.gather(frame_probs, 1, peak_indices.unsqueeze(1)).squeeze(1)
        fused = clip_probs if args.score_fusion == "clip" else clip_probs * peak_probs
        for batch_index, meta in enumerate(batch["meta"]):
            decoded += 1
            if bool(meta.get("decode_failed", False)):
                decode_failures += 1
                continue
            video_id = str(meta["video_id"])
            for label_index, label in enumerate(labels):
                common = {
                    "candidate_id": str(meta["sample_id"]),
                    "source_timestamp": float(meta["anchor_time"]),
                    "score": float(fused[batch_index, label_index]),
                    "clip_score": float(clip_probs[batch_index, label_index]),
                    "frame_peak_score": float(peak_probs[batch_index, label_index]),
                }
                raw_rows_by_time_mode["source"][(video_id, label)].append(
                    {**common, "timestamp": float(meta["anchor_time"])}
                )
                raw_rows_by_time_mode["frame_peak"][(video_id, label)].append(
                    {
                        **common,
                        "timestamp": float(peak_times[batch_index, label_index]),
                    }
                )
        if decoded % max(args.batch_size * 20, 1) == 0:
            print(f"decoded={decoded}/{len(dataset)} failures={decode_failures}", flush=True)

    nms_radii = {"shot": 3.0, "save": 3.0, "corner": 6.0, "freekick": 6.0, "penalty": 6.0}
    rows_by_time_mode = {
        mode: {
            key: class_nms(rows, nms_radii[key[1]])
            for key, rows in rows_by_video.items()
        }
        for mode, rows_by_video in raw_rows_by_time_mode.items()
    }
    durations = {
        str(video["video_id"]): float(video["duration_seconds"])
        for video in manifest["videos"]
    }
    events_by_video_label: dict[tuple[str, str], list[float]] = {}
    annotation_root = repo_path(str(cfg.data.long_video.roots[0].annotations_dir))
    for video_id in durations:
        events = load_events(annotation_root / f"{video_id}.json")
        for label in labels:
            events_by_video_label[(video_id, label)] = events_for_label(events, label)
    targets = {"shot": 0.90, "save": 0.85, "corner": 0.85, "freekick": 0.85, "penalty": 0.85}
    metrics = {
        mode: {
            str(tolerance): evaluate_rows(
                rows_by_video_label,
                events_by_video_label,
                labels,
                durations,
                tolerance=tolerance,
                target_recall=targets,
            )
            for tolerance in (2.0, 3.0, 5.0)
        }
        for mode, rows_by_video_label in rows_by_time_mode.items()
    }
    source_ceiling = {
        str(tolerance): source_candidate_ceiling(
            manifest, events_by_video_label, labels, tolerance
        )
        for tolerance in (2.0, 3.0, 5.0)
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "evaluation_split": "calibration",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "score_fusion": args.score_fusion,
        "target_recall": targets,
        "nms_radius_seconds": nms_radii,
        "video_count": len(durations),
        "input_candidate_count": int(manifest["candidate_count"]),
        "decoded_candidate_count": decoded,
        "decode_failures": decode_failures,
        "duration_minutes": sum(durations.values()) / 60.0,
        "source_candidate_ceiling": source_ceiling,
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
