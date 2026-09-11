from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT / "src", WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from football_longform_v2.backbones import BackboneError, build_plain_backbone  # noqa: E402
from football_longform_v2.canonical import load_canonical_split
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.feature_store import align_feature_stream  # noqa: E402
from football_longform_v2.schema import assert_disjoint_splits, read_video_ids  # noqa: E402
from build_rgb_timeline import extract_streams, find_video, timeline_metadata  # noqa: E402

CACHE_SCHEMA = "football_longform_v2.timeline.official_a.v1"
ALLOWED_SPLITS = frozenset(("train", "calibration"))
REQUIRED_CACHE_KEYS = ("timestamps", "context", "motion", "context_valid", "motion_valid")


def resolve(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def scalar_text(data: Any, key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if getattr(value, "shape", ()) == ():
        return str(value.item())
    return None


def cache_validation(
    path: Path,
    *,
    context_dim: int,
    motion_dim: int,
    expected_config_sha256: str,
    expected_weights_sha256: str,
) -> tuple[bool, str]:
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_CACHE_KEYS if key not in data]
            if missing:
                return False, f"missing keys {missing}"
            timestamps = data["timestamps"]
            context, motion = data["context"], data["motion"]
            context_valid, motion_valid = data["context_valid"], data["motion_valid"]
            if timestamps.ndim != 1 or timestamps.size == 0:
                return False, "timestamps must be non-empty rank-1"
            if context.shape != (timestamps.size, context_dim):
                return False, f"invalid context shape {context.shape}"
            if motion.shape != (timestamps.size, motion_dim):
                return False, f"invalid motion shape {motion.shape}"
            if context_valid.shape != timestamps.shape or motion_valid.shape != timestamps.shape:
                return False, "valid masks do not match timestamps"
            if not (np.isfinite(timestamps).all() and np.isfinite(context).all() and np.isfinite(motion).all()):
                return False, "non-finite values"
            if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0.0):
                return False, "non-monotonic timestamps"
            if scalar_text(data, "cache_schema") != CACHE_SCHEMA:
                return False, "cache schema mismatch"
            if scalar_text(data, "config_sha256") != expected_config_sha256:
                return False, "config digest mismatch"
            if scalar_text(data, "weights_sha256") != expected_weights_sha256:
                return False, "weights digest mismatch"
    except (OSError, ValueError, KeyError) as error:
        return False, f"cannot read cache: {error}"
    return True, "valid"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_items(
    split_ids: dict[str, tuple[str, ...]], selected: tuple[str, ...]
) -> list[tuple[str, str]]:
    """Return the stable global extraction order used by every worker."""
    return [(split, video_id) for split in selected for video_id in split_ids[split]]


def shard_items(
    items: list[tuple[str, str]], *, num_shards: int, shard_index: int
) -> list[tuple[int, str, str]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return [
        (global_index, split, video_id)
        for global_index, (split, video_id) in enumerate(items)
        if global_index % num_shards == shard_index
    ]




@contextmanager
def cache_claim(feature_root: Path, split: str, video_id: str):
    """Acquire a persistent advisory per-timeline claim without deleting it."""
    path = feature_root / ".claims" / split / f"{video_id}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            pass
        yield acquired, path
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def source_duration_seconds(video: Path) -> float:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for duration: {video}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        capture.release()
    if fps <= 0.0 or frames <= 0.0:
        raise RuntimeError(f"invalid duration metadata for {video}: frames={frames} fps={fps}")
    return frames / fps


def cache_duration_validation(path: Path, *, source_duration_s: float) -> tuple[bool, str]:
    try:
        with np.load(path, allow_pickle=False) as data:
            final_timestamp = float(data["timestamps"][-1])
    except (OSError, ValueError, KeyError, IndexError) as error:
        return False, f"cannot inspect cache duration: {error}"
    tolerance = max(2.0, source_duration_s * 0.001)
    if abs(final_timestamp - source_duration_s) > tolerance:
        return False, (
            f"truncated cache final_timestamp={final_timestamp:.3f} "
            f"duration={source_duration_s:.3f} tolerance={tolerance:.3f}"
        )
    return True, "valid"


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def extract_with_batch_fallback(
    video: Path, model: torch.nn.Module, *, device: torch.device,
    image_size: tuple[int, int], context_hz: float, motion_hz: float,
    motion_grid: tuple[int, int], requested_batch_size: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int, float, bool]:
    candidates = [requested_batch_size] + [
        fallback for fallback in (16, 8) if fallback < requested_batch_size
    ]
    had_oom = False
    for batch_size in candidates:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            streams = extract_streams(
                video, model, device=device, image_size=image_size,
                context_hz=context_hz, motion_hz=motion_hz, motion_grid=motion_grid,
                context_batch_size=batch_size, max_seconds=None,
            )
            peak_mb = (float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
                       if device.type == "cuda" else 0.0)
            return streams, batch_size, peak_mb, had_oom
        except Exception as error:
            if not is_cuda_oom(error) or batch_size == candidates[-1]:
                raise
            had_oom = True
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise AssertionError("unreachable")


def write_cache(
    output: Path,
    *,
    context_times: np.ndarray,
    context_values: np.ndarray,
    motion_times: np.ndarray,
    motion_values: np.ndarray,
    config: dict[str, Any],
    config_sha256: str,
    weights_sha256: str,
    video: Path,
) -> int:
    features = config["features"]
    context, motion = features["context"], features["motion"]
    timeline_hz = float(features["timeline_hz"])
    end_time = max(float(context_times[-1]), float(motion_times[-1]))
    target_times = torch.arange(0.0, end_time + 0.5 / timeline_hz, 1.0 / timeline_hz)
    aligned_context, context_valid = align_feature_stream(
        torch.from_numpy(context_times), torch.from_numpy(context_values), target_times,
        max_gap_seconds=0.75 / float(context["source_hz"]),
    )
    aligned_motion, motion_valid = align_feature_stream(
        torch.from_numpy(motion_times), torch.from_numpy(motion_values), target_times,
        max_gap_seconds=0.75 / float(motion["source_hz"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            timestamps=target_times.numpy().astype(np.float32),
            context=aligned_context.numpy().astype(np.float16),
            motion=aligned_motion.numpy().astype(np.float16),
            context_valid=context_valid.numpy(),
            motion_valid=motion_valid.numpy(),
            cache_schema=np.asarray(CACHE_SCHEMA),
            config_sha256=np.asarray(config_sha256),
            weights_sha256=np.asarray(weights_sha256),
            **timeline_metadata(
                arch=str(context["arch"]),
                backbone_id=str(context.get("backbone_id", "official_lvd1689m")),
                weights=Path(str(context["weights"])),
                video=video,
                max_seconds=None,
            ),
        )
    os.replace(temporary, output)
    return int(target_times.numel())


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable frozen official-DINO LF-A0 feature extraction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--splits", nargs="+", default=["train", "calibration"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--opencv-threads", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--manifest-name")
    parser.add_argument("--run-id", default="parallel_default")
    args = parser.parse_args()
    selected = tuple(args.splits)
    unknown = sorted(set(selected) - ALLOWED_SPLITS)
    if unknown:
        raise ValueError(
            f"only {sorted(ALLOWED_SPLITS)} are extractable; refusing forbidden splits: {unknown}"
        )
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate --splits values")
    if args.context_batch_size < 1:
        raise ValueError("--context-batch-size must be >= 1")
    if args.opencv_threads < 0 or args.torch_threads < 0:
        raise ValueError("thread counts must be >= 0")
    if args.opencv_threads:
        import cv2

        cv2.setNumThreads(args.opencv_threads)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    config = load_config(args.config)
    root = Path(config["_project_root"])
    paths = config["paths"]
    split_ids = {
        "train": read_video_ids(resolve(root, paths["train_ids"])),
        "calibration": read_video_ids(resolve(root, paths["calibration_ids"])),
        "thirdparty_test": read_video_ids(resolve(root, paths["thirdparty_test_ids"])),
    }
    assert_disjoint_splits(*split_ids.values())
    canonical_train = load_canonical_split(resolve(root, paths["canonical_manifest"]), "train")
    canonical_calibration = load_canonical_split(resolve(root, paths["canonical_manifest"]), "calibration")
    if canonical_train.media_ids != split_ids["train"] or canonical_calibration.media_ids != split_ids["calibration"]:
        raise RuntimeError("canonical manifest and configured extraction IDs disagree")
    legacy = read_video_ids(resolve(root, paths["legacy_long6_ids"]))
    requested_ids = {video_id for split in selected for video_id in split_ids[split]}
    protected_ids = set(split_ids["thirdparty_test"])
    if requested_ids & protected_ids:
        raise RuntimeError("thirdparty18 IDs reached extraction request; aborting")
    if requested_ids & set(legacy):
        raise RuntimeError("legacy long6 holdout ID reached extraction request; aborting")

    feature_root = resolve(root, paths["feature_store"])
    video_root = Path(args.video_root).expanduser().resolve()
    context, motion = config["features"]["context"], config["features"]["motion"]
    weights = resolve(root, str(context["weights"]))
    if not weights.is_file():
        raise FileNotFoundError(f"official A weights not found: {weights}")
    config_sha256 = config_digest(Path(config["_config_path"]))
    weights_sha256 = sha256_file(weights)
    device = torch.device(args.device)
    try:
        model = build_plain_backbone(str(context["arch"]), weights).to(device).eval()
    except BackboneError as error:
        raise RuntimeError(f"cannot load frozen official context backbone: {error}") from error

    all_items = canonical_items(split_ids, selected)
    assigned_items = shard_items(
        all_items, num_shards=args.num_shards, shard_index=args.shard_index
    )
    manifest_path = feature_root / (args.manifest_name or (
        "manifest.json" if args.num_shards == 1
        else f"manifests/{args.run_id}/worker-{args.shard_index}.json"
    ))
    records: list[dict[str, Any]] = []
    total = len(all_items)
    shard_total = len(assigned_items)
    started = time.monotonic()
    completed = skipped_valid = skipped_missing_media = failed = 0
    manifest_base = {
        "cache_schema": CACHE_SCHEMA,
        "experiment_id": config["experiment_id"],
        "backbone_id": str(context.get("backbone_id", "official_lvd1689m")),
        "weights": str(weights),
        "weights_sha256": weights_sha256,
        "config": str(config["_config_path"]),
        "config_sha256": config_sha256,
        "requested_splits": list(selected),
        "forbidden_split": "thirdparty_test",
        "device": str(device),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_total": shard_total,
        "opencv_threads": args.opencv_threads,
        "torch_threads": args.torch_threads,
        "context_batch_size": args.context_batch_size,
        "run_id": args.run_id,
        "started_unix": time.time(),
    }
    for global_index, split, video_id in assigned_items:
        output = feature_root / split / video_id / "timeline.npz"
        valid, reason = cache_validation(
            output, context_dim=int(context["dim"]), motion_dim=int(motion["dim"]),
            expected_config_sha256=config_sha256, expected_weights_sha256=weights_sha256,
        ) if output.is_file() else (False, "missing")
        source_duration_s: float | None = None
        if valid:
            try:
                source_duration_s = source_duration_seconds(find_video(video_root, video_id))
                valid, reason = cache_duration_validation(
                    output, source_duration_s=source_duration_s
                )
            except FileNotFoundError as error:
                valid, reason = False, str(error)
        if valid:
            skipped_valid += 1
            records.append({
                "global_index": global_index, "split": split, "video_id": video_id,
                "status": "skipped_valid", "output": str(output),
                "source_duration_s": source_duration_s,
            })
        else:
            with cache_claim(feature_root, split, video_id) as (claimed, claim_path):
                if not claimed:
                    records.append({
                        "global_index": global_index, "split": split, "video_id": video_id,
                        "status": "claim_contended", "output": str(output),
                        "claim": str(claim_path), "previous_cache_reason": reason,
                    })
                else:
                    try:
                        valid, reason = cache_validation(
                            output, context_dim=int(context["dim"]), motion_dim=int(motion["dim"]),
                            expected_config_sha256=config_sha256, expected_weights_sha256=weights_sha256,
                        ) if output.is_file() else (False, "missing")
                        video = find_video(video_root, video_id)
                        source_duration_s = source_duration_seconds(video)
                        if valid:
                            valid, reason = cache_duration_validation(
                                output, source_duration_s=source_duration_s
                            )
                        if valid:
                            skipped_valid += 1
                            records.append({
                                "global_index": global_index, "split": split, "video_id": video_id,
                                "status": "skipped_valid_after_claim", "output": str(output),
                                "claim": str(claim_path), "source_duration_s": source_duration_s,
                            })
                        else:
                            started_video = time.monotonic()
                            streams, effective_batch, peak_memory_mb, had_oom = extract_with_batch_fallback(
                                video, model, device=device,
                                image_size=tuple(map(int, context["image_size"])),
                                context_hz=float(context["source_hz"]),
                                motion_hz=float(motion["source_hz"]),
                                motion_grid=tuple(map(int, motion["grid_size"])),
                                requested_batch_size=args.context_batch_size,
                            )
                            context_times, context_values, motion_times, motion_values = streams
                            steps = write_cache(
                                output, context_times=context_times, context_values=context_values,
                                motion_times=motion_times, motion_values=motion_values, config=config,
                                config_sha256=config_sha256, weights_sha256=weights_sha256, video=video,
                            )
                            valid, validation = cache_validation(
                                output, context_dim=int(context["dim"]), motion_dim=int(motion["dim"]),
                                expected_config_sha256=config_sha256, expected_weights_sha256=weights_sha256,
                            )
                            if valid:
                                valid, validation = cache_duration_validation(
                                    output, source_duration_s=source_duration_s
                                )
                            if not valid:
                                raise RuntimeError(f"post-write cache validation failed: {validation}")
                            completed += 1
                            records.append({
                                "global_index": global_index, "split": split, "video_id": video_id,
                                "status": "completed", "output": str(output), "claim": str(claim_path),
                                "timeline_steps": steps, "source_duration_s": source_duration_s,
                                "wall_seconds": time.monotonic() - started_video,
                                "effective_batch": effective_batch, "oom_fallback": had_oom,
                                "peak_cuda_memory_mb": peak_memory_mb,
                            })
                    except FileNotFoundError as error:
                        skipped_missing_media += 1
                        records.append({
                            "global_index": global_index, "split": split, "video_id": video_id,
                            "status": "skipped_missing_media", "output": str(output),
                            "claim": str(claim_path), "previous_cache_reason": reason, "error": str(error),
                        })
                    except Exception as error:
                        failed += 1
                        records.append({
                            "global_index": global_index, "split": split, "video_id": video_id,
                            "status": "failed", "output": str(output), "claim": str(claim_path),
                            "previous_cache_reason": reason, "oom": is_cuda_oom(error),
                            "error": repr(error),
                        })
        processed = completed + skipped_valid + skipped_missing_media + failed
        elapsed = time.monotonic() - started
        active = max(completed, 1)
        eta_seconds = max(shard_total - processed, 0) * elapsed / active
        progress = {**manifest_base, "updated_unix": time.time(), "total": total,
            "completed": completed, "skipped_valid": skipped_valid,
            "skipped_missing_media": skipped_missing_media, "failed": failed,
            "processed": processed, "elapsed_seconds": elapsed, "eta_seconds": eta_seconds,
            "records": records}
        atomic_json(manifest_path, progress)
        print(
            f"progress={processed}/{shard_total} completed={completed} skipped_valid={skipped_valid} "
            f"skipped_missing_media={skipped_missing_media} failed={failed} "
            f"elapsed_s={elapsed:.1f} eta_s={eta_seconds:.1f} last={split}/{video_id}",
            flush=True,
        )
    print(f"manifest={manifest_path}", flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
