#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from PIL import Image
import webdataset as wds

from dinov3.data.datasets.football_video_ssl import motion_local_crop
from dinov3.data.football_ssl import load_football_ssl_data_config
from football_detection_aware import RobustClipCropper


NEW_VIDEO_CUT = Path("/mnt/data_16t/football/video_cut")


def source_bucket(path: str) -> str:
    if "raw_video_hq_720P" in path:
        return "raw_hq"
    if "video_cut" in path:
        return "cut"
    return "new_raw"


def scene_group(record: dict[str, Any]) -> str:
    stem = Path(record["name"]).stem
    if stem.endswith("_cut") and "_" in stem:
        return f"cut_{stem.split('_', 1)[0]}"
    return f"video_{stem}"


def resolve_video(record: dict[str, Any]) -> Path:
    name = Path(record["name"]).name
    candidates = [NEW_VIDEO_CUT / name]
    candidates.extend(Path(value) for value in record.get("fallback_paths", []))
    candidates.append(Path(record["path"]))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file() and candidate.stat().st_size > 1_000_000:
            return candidate
    raise FileNotFoundError(f"No current path for {name}: {candidates}")


def jpeg_stream(command: list[str]) -> Iterator[bytes]:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    buffer = bytearray()
    while True:
        chunk = process.stdout.read(1 << 20)
        if not chunk:
            break
        buffer.extend(chunk)
        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                if len(buffer) > 2:
                    del buffer[:-2]
                break
            end = buffer.find(b"\xff\xd9", start + 2)
            if end < 0:
                if start:
                    del buffer[:start]
                break
            end += 2
            yield bytes(buffer[start:end])
            del buffer[:end]
    stderr = process.stderr.read().decode(errors="ignore") if process.stderr is not None else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed exit={return_code}: {stderr[-2000:]}")


def decode_jpeg(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def encode_jpeg(image: Image.Image, quality: int = 90) -> bytes:
    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=quality, optimize=False)
    return output.getvalue()


def frame_stats(image: Image.Image, previous: Image.Image | None) -> dict[str, float | int]:
    array = np.asarray(image.resize((192, 108), Image.Resampling.BILINEAR), dtype=np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    green = ((hsv[..., 0] >= 30) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 45)).mean()
    blur = cv2.Laplacian(gray, cv2.CV_32F).var()
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    dhash = 0
    for bit in bits.flat:
        dhash = (dhash << 1) | int(bit)
    motion = 0.0
    shot_change = 0
    if previous is not None:
        previous_array = np.asarray(previous.resize((192, 108), Image.Resampling.BILINEAR), dtype=np.uint8)
        difference = np.abs(array.astype(np.int16) - previous_array.astype(np.int16))
        motion = float(difference.mean())
        shot_change = int(motion >= 32.0)
    return {
        "brightness": float(gray.mean()),
        "green_ratio": float(green),
        "blur": float(blur),
        "motion": motion,
        "shot_change": shot_change,
        "dhash": int(dhash),
    }


def fallback_local(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = max((width - side) // 2, 0)
    top = max((height - side) // 2, 0)
    return image.crop((left, top, left + side, top + side))


def make_sample(
    record: dict[str, Any],
    frame_index: int,
    stride_sec: float,
    anchor_bytes: bytes,
    anchor: Image.Image,
    next_anchor: Image.Image | None,
    cropper: RobustClipCropper | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    video_id = Path(record["name"]).stem
    time_sec = frame_index * stride_sec
    stats = frame_stats(anchor, next_anchor)
    detector_valid = False
    detector_reason = "missing_index"
    local_source = "center_fallback"
    local = None
    if cropper is not None:
        width, height = anchor.size
        proposal = cropper.get_window_roi(
            video_id,
            max(time_sec - 1.0, 0.0),
            min(time_sec + 1.0, float(record["duration_sec"])),
            width,
            height,
            (320, 320),
        )
        detector_valid = bool(proposal.valid and proposal.bbox is not None)
        detector_reason = proposal.fallback_reason or ("ok" if detector_valid else "invalid")
        if detector_valid:
            local = anchor.crop(proposal.bbox)
            local_source = "detector"
    if local is None and next_anchor is not None and not stats["shot_change"]:
        local = motion_local_crop(anchor, next_anchor)
        if local is not None:
            local_source = "motion"
    if local is None:
        local = fallback_local(anchor)

    metadata = {
        "video_id": video_id,
        "time_sec": time_sec,
        "source_bucket": source_bucket(str(record["path"])),
        "scene_group": scene_group(record),
        "local_source": local_source,
        "detector_valid": detector_valid,
        "detector_reason": detector_reason,
        "width": anchor.width,
        "height": anchor.height,
        **stats,
    }
    key = f"{video_id}_{frame_index:06d}"
    sample = {
        "__key__": key,
        "jpg": anchor_bytes,
        "local.jpg": encode_jpeg(local, quality=90),
        "json": json.dumps(metadata, ensure_ascii=False).encode(),
    }
    return sample, metadata


def quality_decision(metadata: dict[str, Any], previous_dhash: int | None, duplicate_run: int) -> tuple[bool, str, int]:
    """Conservative sports-camera filtering without rejecting valid close-ups."""
    detector_valid = bool(metadata["detector_valid"])
    green = float(metadata["green_ratio"])
    motion = float(metadata["motion"])
    if not detector_valid and green < 0.02 and motion < 1.5:
        return False, "static_non_pitch", 0
    if previous_dhash is not None:
        hamming = (int(metadata["dhash"]) ^ previous_dhash).bit_count()
        if not detector_valid and hamming <= 1:
            duplicate_run += 1
            if duplicate_run % 4 != 0:
                return False, "near_duplicate", duplicate_run
        else:
            duplicate_run = 0
    return True, "keep", duplicate_run


def process_partition(
    worker_id: int,
    records: list[dict[str, Any]],
    output_dir: str,
    data_config_path: str,
    maxcount: int,
    maxsize: int,
    cuda_device: int | None = None,
) -> dict[str, Any]:
    cv2.setNumThreads(0)
    config = load_football_ssl_data_config(data_config_path)
    cropper = RobustClipCropper.from_config(config)
    pattern = str(Path(output_dir) / f"worker{worker_id:02d}-%06d.tar")
    summary: dict[str, Any] = {
        "worker": worker_id,
        "videos": 0,
        "samples": 0,
        "bytes": 0,
        "failures": [],
        "candidate_samples": 0,
        "filtered": Counter(),
        "local_sources": Counter(),
        "buckets": Counter(),
    }
    with wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize, encoder=False) as sink:
        for record in records:
            try:
                path = resolve_video(record)
                stride_sec = float(record["cache_stride_sec"])
                fps_filter = 1.0 / stride_sec
                max_frames = int(record["cache_max_samples"])
                if cuda_device is None:
                    input_options = ["-threads", "2"]
                    video_filter = f"fps={fps_filter:.8f},scale=1280:-2:force_original_aspect_ratio=decrease"
                else:
                    input_options = [
                        "-hwaccel", "cuda", "-hwaccel_device", str(cuda_device),
                        "-hwaccel_output_format", "cuda",
                    ]
                    video_filter = f"scale_cuda=1280:-2,hwdownload,format=nv12,fps={fps_filter:.8f}"
                command = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", *input_options,
                    "-i", str(path), "-vf", video_filter,
                    "-frames:v", str(max_frames),
                    "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
                ]
                previous_payload = None
                previous_image = None
                frame_index = 0
                previous_dhash = None
                duplicate_run = 0
                for payload in jpeg_stream(command):
                    image = decode_jpeg(payload)
                    if previous_payload is not None and previous_image is not None:
                        sample, metadata = make_sample(
                            record, frame_index - 1, stride_sec,
                            previous_payload, previous_image, image, cropper,
                        )
                        summary["candidate_samples"] += 1
                        keep, reason, duplicate_run = quality_decision(metadata, previous_dhash, duplicate_run)
                        previous_dhash = int(metadata["dhash"])
                        if keep:
                            sink.write(sample)
                            summary["samples"] += 1
                            summary["bytes"] += len(sample["jpg"]) + len(sample["local.jpg"])
                            summary["local_sources"][metadata["local_source"]] += 1
                            summary["buckets"][metadata["source_bucket"]] += 1
                        else:
                            summary["filtered"][reason] += 1
                    previous_payload, previous_image = payload, image
                    frame_index += 1
                if previous_payload is not None and previous_image is not None:
                    sample, metadata = make_sample(
                        record, frame_index - 1, stride_sec,
                        previous_payload, previous_image, None, cropper,
                    )
                    summary["candidate_samples"] += 1
                    keep, reason, duplicate_run = quality_decision(metadata, previous_dhash, duplicate_run)
                    previous_dhash = int(metadata["dhash"])
                    if keep:
                        sink.write(sample)
                        summary["samples"] += 1
                        summary["bytes"] += len(sample["jpg"]) + len(sample["local.jpg"])
                        summary["local_sources"][metadata["local_source"]] += 1
                        summary["buckets"][metadata["source_bucket"]] += 1
                    else:
                        summary["filtered"][reason] += 1
                summary["videos"] += 1
            except Exception as error:
                summary["failures"].append({"name": record.get("name"), "error": repr(error)})
    summary["local_sources"] = dict(summary["local_sources"])
    summary["filtered"] = dict(summary["filtered"])
    summary["buckets"] = dict(summary["buckets"])
    return summary


def build_plan(records: list[dict[str, Any]], max_videos: int, seed: int, max_samples_per_video: int = 0) -> list[dict[str, Any]]:
    counts = Counter(scene_group(record) for record in records)
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        bucket = source_bucket(str(record["path"]))
        stride = {"raw_hq": 2.0, "cut": 2.0, "new_raw": 0.5}[bucket]
        group_cap = max(2400 // counts[scene_group(record)], 120)
        available = max(int(float(record["duration_sec"]) // stride), 1)
        item = dict(record)
        item["cache_stride_sec"] = stride
        item["cache_max_samples"] = min(available, group_cap)
        if max_samples_per_video > 0:
            item["cache_max_samples"] = min(item["cache_max_samples"], max_samples_per_video)
        by_bucket[bucket].append(item)
    for values in by_bucket.values():
        rng.shuffle(values)
    if max_videos <= 0 or max_videos >= len(records):
        result = [item for values in by_bucket.values() for item in values]
        rng.shuffle(result)
        return result
    # Round-robin source selection makes small pilot caches diverse instead of
    # selecting the first N lexicographic cut clips.
    result = []
    order = ["new_raw", "raw_hq", "cut"]
    while len(result) < max_videos and any(by_bucket.values()):
        for bucket in order:
            if by_bucket[bucket] and len(result) < max_videos:
                result.append(by_bucket[bucket].pop())
    return result


def partition_by_work(records: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    partitions = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for record in sorted(records, key=lambda item: item["cache_max_samples"], reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        partitions[index].append(record)
        loads[index] += float(record["duration_sec"])
    return partitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sequential WebDataset cache for football SSL.")
    parser.add_argument("--manifest", default="outputs/football_ssl/manifests/football_all_valid_v2.jsonl")
    parser.add_argument("--data-config", default="configs/football_ssl/football_detector_local_v2.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cuda-devices", default="", help="Comma-separated physical GPU ids for NVDEC and scale_cuda")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-samples-per-video", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--maxcount", type=int, default=1024)
    parser.add_argument("--maxsize-mb", type=int, default=512)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(args.manifest).open() if line.strip()]
    plan = build_plan(records, args.max_videos, args.seed, args.max_samples_per_video)
    partitions = partition_by_work(plan, min(args.workers, len(plan)))
    cuda_devices = [int(value) for value in args.cuda_devices.split(",") if value.strip()]
    plan_summary = {
        "created_at": time.time(),
        "manifest": str(Path(args.manifest).resolve()),
        "videos": len(plan),
        "scene_groups": len({scene_group(record) for record in plan}),
        "video_duration_hours": sum(float(record["duration_sec"]) for record in plan) / 3600.0,
        "planned_samples": sum(record["cache_max_samples"] for record in plan),
        "planned_buckets": Counter(source_bucket(str(record["path"])) for record in plan),
        "planned_samples_by_bucket": dict(sum((
            Counter({source_bucket(str(record["path"])): int(record["cache_max_samples"])})
            for record in plan
        ), Counter())),
        "workers": len(partitions),
        "cuda_devices": cuda_devices,
    }
    plan_summary["planned_buckets"] = dict(plan_summary["planned_buckets"])
    (output / "build_plan.json").write_text(json.dumps(plan_summary, indent=2, ensure_ascii=False) + "\n")
    (output / "selected_videos.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in plan)
    )
    print(json.dumps(plan_summary, indent=2), flush=True)

    started = time.time()
    summaries = []
    with ProcessPoolExecutor(max_workers=len(partitions)) as executor:
        futures = [
            executor.submit(
                process_partition, worker_id, partition, str(output), args.data_config,
                args.maxcount, args.maxsize_mb * 1024 * 1024,
                cuda_devices[worker_id % len(cuda_devices)] if cuda_devices else None,
            )
            for worker_id, partition in enumerate(partitions)
        ]
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    final = {
        **plan_summary,
        "elapsed_sec": time.time() - started,
        "candidate_samples": sum(summary["candidate_samples"] for summary in summaries),
        "samples": sum(summary["samples"] for summary in summaries),
        "videos_complete": sum(summary["videos"] for summary in summaries),
        "bytes": sum(summary["bytes"] for summary in summaries),
        "failures": [failure for summary in summaries for failure in summary["failures"]],
        "filtered": dict(sum((Counter(summary["filtered"]) for summary in summaries), Counter())),
        "local_sources": dict(sum((Counter(summary["local_sources"]) for summary in summaries), Counter())),
        "buckets": dict(sum((Counter(summary["buckets"]) for summary in summaries), Counter())),
        "shards": len(list(output.glob("*.tar"))),
    }
    (output / "summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

