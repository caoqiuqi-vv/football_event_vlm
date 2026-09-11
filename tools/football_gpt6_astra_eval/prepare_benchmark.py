#!/usr/bin/env python3
"""Prepare fixed clip-classification and long-video localization protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LABELS = ("shot", "save", "free_kick", "corner", "kickoff")
DEFAULT_VIDEOS = (
    "2042520801973841921",
    "2041772596969549825",
    "2042526457476886530",
)
DEFAULT_ANNOTATIONS = Path(
    "outputs/football_event_annotations/"
    "test18_final_repaired_v5_original_goals_20260910_per_video"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-ids", nargs="+", default=list(DEFAULT_VIDEOS))
    parser.add_argument("--clip-duration", type=float, default=12.0)
    parser.add_argument("--anchor-radius", type=float, default=3.0)
    parser.add_argument("--positive-clips-per-video", type=int, default=20)
    parser.add_argument("--negative-clips-per-video", type=int, default=10)
    parser.add_argument("--core-duration", type=float, default=90.0)
    parser.add_argument("--context-duration", type=float, default=15.0)
    parser.add_argument(
        "--materialize", choices=("none", "clips", "chunks", "all"), default="none"
    )
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--overwrite-media", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def target_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for event in payload.get("events", []):
        label = event.get("semantic_label", event.get("label"))
        if label not in LABELS:
            continue
        events.append(
            {
                "id": str(event.get("id", event.get("source_id", ""))),
                "label": label,
                "time_sec": float(event["time_sec"]),
                "origin": event.get("event_origin", "unknown"),
            }
        )
    return sorted(events, key=lambda item: (item["time_sec"], item["label"], item["id"]))


def make_anchor_groups(events: list[dict[str, Any]], radius: float) -> list[dict[str, Any]]:
    """Make classification anchors while preserving the full event list in each sample."""
    unused = set(range(len(events)))
    groups: list[dict[str, Any]] = []
    while unused:
        seed_index = min(unused, key=lambda index: events[index]["time_sec"])
        center = events[seed_index]["time_sec"]
        members = [
            index for index in sorted(unused)
            if abs(events[index]["time_sec"] - center) <= radius
        ]
        # Recenter once so a shot/save pair near a boundary remains one multi-label sample.
        center = sum(events[index]["time_sec"] for index in members) / len(members)
        members = [
            index for index in sorted(unused)
            if abs(events[index]["time_sec"] - center) <= radius
        ]
        for index in members:
            unused.remove(index)
        selected = [events[index] for index in members]
        groups.append(
            {
                "anchor_sec": center,
                "labels": sorted({event["label"] for event in selected}),
                "events": selected,
            }
        )
    return groups


def stratified_groups(
    groups: list[dict[str, Any]], limit: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Greedily cover rare labels, then fill deterministically with shuffled groups."""
    if len(groups) <= limit:
        return groups
    remaining = list(groups)
    rng.shuffle(remaining)
    selected: list[dict[str, Any]] = []
    counts = Counter()
    while remaining and len(selected) < limit:
        def utility(group: dict[str, Any]) -> tuple[float, float]:
            rarity_gain = sum(1.0 / (1.0 + counts[label]) for label in group["labels"])
            return rarity_gain, -group["anchor_sec"]

        chosen = max(remaining, key=utility)
        remaining.remove(chosen)
        selected.append(chosen)
        counts.update(chosen["labels"])
    return sorted(selected, key=lambda item: item["anchor_sec"])


def background_centers(
    duration: float,
    events: list[dict[str, Any]],
    clip_duration: float,
    count: int,
    rng: random.Random,
) -> list[float]:
    half = clip_duration / 2.0
    candidates = []
    cursor = half
    while cursor <= duration - half:
        # A larger exclusion radius prevents nearby unlabelled context becoming a fake negative.
        if all(abs(cursor - event["time_sec"]) > half + 5.0 for event in events):
            candidates.append(cursor)
        cursor += 5.0
    rng.shuffle(candidates)
    return sorted(candidates[:count])


def clip_rows(
    video_id: str,
    video_path: Path,
    duration: float,
    events: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    half = args.clip_duration / 2.0
    groups = make_anchor_groups(events, args.anchor_radius)
    groups = stratified_groups(groups, args.positive_clips_per_video, rng)
    rows: list[dict[str, Any]] = []
    for ordinal, group in enumerate(groups):
        center = min(max(group["anchor_sec"], half), duration - half)
        rows.append(
            {
                "sample_id": f"{video_id}_positive_{ordinal:03d}",
                "video_id": video_id,
                "source_video": str(video_path),
                "clip_start_sec": round(center - half, 6),
                "clip_end_sec": round(center + half, 6),
                "anchor_sec_in_source": round(group["anchor_sec"], 6),
                "anchor_sec_in_clip": round(group["anchor_sec"] - (center - half), 6),
                "anchor_radius_sec": args.anchor_radius,
                "gt_labels": group["labels"],
                "gt_events": group["events"],
                "is_background": False,
                "media_path": f"clips/{video_id}_positive_{ordinal:03d}.mp4",
            }
        )
    for ordinal, center in enumerate(
        background_centers(
            duration, events, args.clip_duration, args.negative_clips_per_video, rng
        )
    ):
        rows.append(
            {
                "sample_id": f"{video_id}_background_{ordinal:03d}",
                "video_id": video_id,
                "source_video": str(video_path),
                "clip_start_sec": round(center - half, 6),
                "clip_end_sec": round(center + half, 6),
                "anchor_sec_in_source": round(center, 6),
                "anchor_sec_in_clip": half,
                "anchor_radius_sec": args.anchor_radius,
                "gt_labels": [],
                "gt_events": [],
                "is_background": True,
                "media_path": f"clips/{video_id}_background_{ordinal:03d}.mp4",
            }
        )
    return sorted(rows, key=lambda item: (item["video_id"], item["clip_start_sec"]))


def chunk_rows(
    video_id: str,
    video_path: Path,
    duration: float,
    events: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    core_start = 0.0
    ordinal = 0
    while core_start < duration:
        core_end = min(duration, core_start + args.core_duration)
        input_start = max(0.0, core_start - args.context_duration)
        input_end = min(duration, core_end + args.context_duration)
        owned = [event for event in events if core_start <= event["time_sec"] < core_end]
        rows.append(
            {
                "chunk_id": f"{video_id}_chunk_{ordinal:04d}",
                "video_id": video_id,
                "source_video": str(video_path),
                "input_start_sec": round(input_start, 6),
                "input_end_sec": round(input_end, 6),
                "core_start_sec": round(core_start, 6),
                "core_end_sec": round(core_end, 6),
                "core_start_sec_relative_to_chunk": round(core_start - input_start, 6),
                "core_end_sec_relative_to_chunk": round(core_end - input_start, 6),
                "gt_events": owned,
                "media_path": f"chunks/{video_id}_chunk_{ordinal:04d}.mp4",
            }
        )
        core_start = core_end
        ordinal += 1
    return rows


def materialize(row: dict[str, Any], output_dir: Path, overwrite: bool) -> None:
    destination = output_dir / row["media_path"]
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if "clip_start_sec" in row:
        start = row["clip_start_sec"]
        end = row["clip_end_sec"]
    else:
        start = row["input_start_sec"]
        end = row["input_end_sec"]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.6f}", "-i", row["source_video"],
        "-t", f"{end - start:.6f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not args.annotations_dir.is_dir():
        raise FileNotFoundError(args.annotations_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    clips: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    gt_snapshot: dict[str, Any] = {}
    videos_summary = []
    for video_id in args.video_ids:
        annotation_path = args.annotations_dir / f"{video_id}.json"
        payload = read_json(annotation_path)
        video_info = payload["video_source"]
        video_path = Path(video_info["video_path"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        duration = float(video_info["duration_sec"])
        events = target_events(payload)
        gt_snapshot[video_id] = events
        video_clips = clip_rows(video_id, video_path, duration, events, args, rng)
        video_chunks = chunk_rows(video_id, video_path, duration, events, args)
        clips.extend(video_clips)
        chunks.extend(video_chunks)
        videos_summary.append(
            {
                "video_id": video_id,
                "duration_sec": duration,
                "annotation_file": str(annotation_path.resolve()),
                "annotation_sha256": sha256(annotation_path),
                "target_events": len(events),
                "events_by_label": dict(sorted(Counter(e["label"] for e in events).items())),
                "clip_samples": len(video_clips),
                "chunks": len(video_chunks),
            }
        )

    write_jsonl(args.output_dir / "clip_manifest.jsonl", clips)
    write_jsonl(args.output_dir / "chunk_manifest.jsonl", chunks)
    write_json(args.output_dir / "gt_snapshot.json", gt_snapshot)
    prompt_source = Path(__file__).resolve().parent / "prompts"
    prompt_destination = args.output_dir / "prompts"
    prompt_destination.mkdir(exist_ok=True)
    for prompt in prompt_source.glob("*.txt"):
        shutil.copy2(prompt, prompt_destination / prompt.name)

    materialize_clips = args.materialize in {"clips", "all"}
    materialize_chunks = args.materialize in {"chunks", "all"}
    if materialize_clips:
        for row in clips:
            materialize(row, args.output_dir, args.overwrite_media)
    if materialize_chunks:
        for row in chunks:
            materialize(row, args.output_dir, args.overwrite_media)

    manifest = {
        "schema_version": "football_gpt6_astra_pilot_v1",
        "seed": args.seed,
        "target_labels": list(LABELS),
        "annotations_dir": str(args.annotations_dir.resolve()),
        "protocol_a": {
            "task": "anchored_multi_label_clip_classification",
            "clip_duration_sec": args.clip_duration,
            "anchor_radius_sec": args.anchor_radius,
            "samples": len(clips),
            "positive_samples": sum(not row["is_background"] for row in clips),
            "background_samples": sum(row["is_background"] for row in clips),
        },
        "protocol_b": {
            "task": "chunked_complete_video_event_localization",
            "core_duration_sec": args.core_duration,
            "context_duration_sec": args.context_duration,
            "chunks": len(chunks),
            "output_ownership": "[core_start_sec, core_end_sec)",
            "temporal_nms": False,
        },
        "videos": videos_summary,
    }
    write_json(args.output_dir / "benchmark_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

