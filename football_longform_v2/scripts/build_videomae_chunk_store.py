from __future__ import annotations

"""Cache one-pass VideoMAE tubelet features from non-overlapping 4s chunks."""

import argparse
import hashlib
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_longform_v2.canonical import load_canonical_split  # noqa: E402
from football_longform_v2.config import load_config as load_a0_config  # noqa: E402
from football_longform_v2.schema import read_video_ids  # noqa: E402
from football_videomaev2 import VideoMAEV2BackboneAdapter  # noqa: E402


SCHEMA = "football_longform_v2.videomae_chunk_store.v1"
MEAN = torch.tensor((0.485, 0.456, 0.406)).reshape(1, 1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).reshape(1, 1, 3, 1, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return payload


def build_backbone(a1: dict[str, Any], checkpoint_path: Path) -> VideoMAEV2BackboneAdapter:
    video = a1["video"]
    model = a1["model"]
    vm = model.get("videomaev2", {})
    backbone = VideoMAEV2BackboneAdapter(
        source_root=str(vm["source_root"]),
        architecture=str(model["backbone"]),
        weights=str(model["weights"]),
        image_size=tuple(int(value) for value in video["image_size"]),
        num_frames=int(video["num_frames"]),
        tubelet_size=int(vm.get("tubelet_size", 2)),
        gradient_checkpointing=False,
        drop_path_rate=0.0,
        preserve_global_feature=bool(vm.get("preserve_global_feature", True)),
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    raw_state = checkpoint.get("model", checkpoint)
    state = {
        str(key)[len("backbone.") :]: value
        for key, value in raw_state.items()
        if str(key).startswith("backbone.")
    }
    if not state:
        raise ValueError("A1 checkpoint contains no backbone.* tensors")
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"A1 backbone load mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    return backbone


def video_geometry(path: Path) -> tuple[cv2.VideoCapture, int, float, float]:
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not cap.isOpened() or frame_count <= 0 or fps <= 0:
        cap.release()
        raise RuntimeError(
            f"invalid video geometry: path={path} frames={frame_count} fps={fps}"
        )
    return cap, frame_count, fps, frame_count / fps


class SequentialFrameSampler:
    """Decode monotonically increasing frame indices without repeated seeking."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self.position = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0))
        self.last: np.ndarray | None = None

    def decode(
        self,
        indices: list[int],
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, float]:
        if not indices:
            raise ValueError("indices cannot be empty")
        if any(right < left for left, right in zip(indices, indices[1:])):
            raise ValueError("sequential decoder requires sorted frame indices")
        output = []
        decoded = 0
        height, width = image_size
        for target in indices:
            if self.last is not None and target < self.position:
                # Rounding can map adjacent requested timestamps to one frame.
                frame = self.last
            else:
                ok = True
                while self.position < target:
                    ok = bool(self.cap.grab())
                    self.position += 1
                    if not ok:
                        break
                if ok:
                    ok, frame = self.cap.read()
                    self.position += 1
                else:
                    frame = None
                if ok and frame is not None:
                    self.last = frame
                    decoded += 1
                elif self.last is not None:
                    frame = self.last
                else:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
            output.append(torch.from_numpy(rgb).permute(2, 0, 1))
        return torch.stack(output), decoded / len(indices)


def decode_indices(
    cap: cv2.VideoCapture,
    indices: list[int],
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, float]:
    """Compatibility helper for isolated calls; production uses one sampler/video."""
    if not indices:
        raise ValueError("indices cannot be empty")
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    return SequentialFrameSampler(cap).decode(indices, image_size)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-config")
    parser.add_argument("--a1-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split", choices=("train", "calibration", "inference"), required=True
    )
    parser.add_argument("--video", help="Single input video for --split inference.")
    parser.add_argument("--video-id", help="Stable output ID for --split inference.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.chunk_seconds <= 0 or args.prefetch_batches <= 0:
        raise ValueError("batch size, chunk seconds and prefetch batches must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index/count")

    a1_config_path = Path(args.a1_config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if args.split == "inference":
        if not args.video or not args.video_id:
            raise ValueError("--split inference requires --video and --video-id")
        if args.num_shards != 1 or args.shard_index != 0:
            raise ValueError("single-video inference does not support sharding")
        source_video = Path(args.video).expanduser().resolve()
        if not source_video.is_file():
            raise FileNotFoundError(f"inference video is missing: {source_video}")
        all_ids = (str(args.video_id),)
        source_by_video_id = {str(args.video_id): source_video}
    else:
        if not args.canonical_config:
            raise ValueError("train/calibration extraction requires --canonical-config")
        if args.video or args.video_id:
            raise ValueError("--video/--video-id are only valid for --split inference")
        canonical_config_path = Path(args.canonical_config).expanduser().resolve()
        a0 = load_a0_config(str(canonical_config_path))
        a0_root = Path(a0["_project_root"])
        canonical = load_canonical_split(
            resolve(a0_root, str(a0["paths"]["canonical_manifest"])), args.split
        )
        id_key = "train_ids" if args.split == "train" else "calibration_ids"
        all_ids = read_video_ids(resolve(a0_root, str(a0["paths"][id_key])))
        if tuple(all_ids) != tuple(canonical.media_ids):
            raise RuntimeError("canonical manifest and split ID list disagree")
        source_by_video_id = {
            video_id: Path(canonical.source_video_by_media_id[video_id])
            for video_id in all_ids
        }
    video_ids = all_ids[args.shard_index :: args.num_shards]
    a1 = load_yaml(a1_config_path)
    frame_count = int(a1["video"]["num_frames"])
    tubelet_size = int(a1["model"]["videomaev2"].get("tubelet_size", 2))
    if frame_count % tubelet_size:
        raise ValueError("num_frames must be divisible by tubelet_size")
    image_size = tuple(int(value) for value in a1["video"]["image_size"])
    backbone = build_backbone(a1, checkpoint_path)
    device = torch.device(args.device)
    backbone.to(device).eval()
    mean = MEAN.to(device)
    std = STD.to(device)
    checkpoint_sha = sha256_file(checkpoint_path)
    config_sha = sha256_file(a1_config_path)
    output_split = output_root / args.split
    output_split.mkdir(parents=True, exist_ok=True)
    shard_rows = []

    for video_id in video_ids:
        source_video = source_by_video_id[video_id]
        source_stat = source_video.stat()
        video_output = output_split / video_id
        metadata_path = video_output / "metadata.json"
        feature_path = video_output / "features.npy"
        timestamp_path = video_output / "timestamps.npy"
        if metadata_path.is_file() and not args.overwrite:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                cached.get("schema") == SCHEMA
                and cached.get("checkpoint_sha256") == checkpoint_sha
                and cached.get("config_sha256") == config_sha
                and cached.get("source_video") == str(source_video)
                and int(cached.get("source_size_bytes", -1)) == source_stat.st_size
                and int(cached.get("source_mtime_ns", -1)) == source_stat.st_mtime_ns
                and feature_path.is_file()
                and timestamp_path.is_file()
            ):
                print(f"skip_ready video={video_id}", flush=True)
                continue
            raise RuntimeError(f"stale cache exists; use --overwrite: {video_output}")
        cap, source_frames, fps, duration = video_geometry(source_video)
        started = time.time()
        chunks = max(int(math.ceil(duration / args.chunk_seconds)), 1)
        feature_parts: list[np.ndarray] = []
        time_parts: list[np.ndarray] = []
        valid_parts: list[np.ndarray] = []
        decode_quality = []
        decoded_queue: queue.Queue = queue.Queue(maxsize=args.prefetch_batches)
        stop_decode = threading.Event()

        def queue_item(item: object) -> bool:
            while not stop_decode.is_set():
                try:
                    decoded_queue.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def decode_producer() -> None:
            sampler = SequentialFrameSampler(cap)
            pending_frames: list[torch.Tensor] = []
            pending_times: list[list[float]] = []
            pending_valid: list[list[bool]] = []
            pending_quality: list[float] = []

            def emit() -> bool:
                if not pending_frames:
                    return True
                item = (
                    "batch",
                    torch.stack(pending_frames),
                    list(pending_times),
                    list(pending_valid),
                    list(pending_quality),
                )
                pending_frames.clear()
                pending_times.clear()
                pending_valid.clear()
                pending_quality.clear()
                return queue_item(item)

            try:
                for chunk_index in range(chunks):
                    if stop_decode.is_set():
                        return
                    start = chunk_index * args.chunk_seconds
                    frame_times = [
                        start + (index + 0.5) * args.chunk_seconds / frame_count
                        for index in range(frame_count)
                    ]
                    frame_valid = [timestamp <= duration for timestamp in frame_times]
                    indices = [
                        min(max(int(round(timestamp * fps)), 0), source_frames - 1)
                        for timestamp in frame_times
                    ]
                    frames, quality = sampler.decode(indices, image_size)
                    pending_frames.append(frames)
                    pending_times.append(frame_times)
                    pending_valid.append(frame_valid)
                    pending_quality.append(quality)
                    if len(pending_frames) >= args.batch_size and not emit():
                        return
                if not emit():
                    return
            except BaseException as error:
                queue_item(("error", error))
            finally:
                queue_item(("done",))

        producer = threading.Thread(
            target=decode_producer,
            name=f"decode-{video_id}",
            daemon=True,
        )
        producer.start()
        try:
            while True:
                item = decoded_queue.get()
                kind = item[0]
                if kind == "done":
                    break
                if kind == "error":
                    raise RuntimeError(f"sequential decode failed for {video_id}") from item[1]
                _, decoded_frames, batch_times, batch_valid, batch_quality = item
                decode_quality.extend(batch_quality)
                inputs = decoded_frames.to(device, non_blocking=True)
                inputs = (inputs.float().div_(255.0) - mean) / std
                frame_features = backbone.forward_temporal_features(
                    inputs, output_frames=frame_count
                )
                batch = frame_features.shape[0]
                tubelets = frame_count // tubelet_size
                tubelet_features = frame_features.reshape(
                    batch, tubelets, tubelet_size, -1
                ).mean(dim=2)
                tubelet_times = torch.tensor(batch_times, dtype=torch.float32).reshape(
                    batch, tubelets, tubelet_size
                ).mean(dim=2)
                tubelet_valid = torch.tensor(batch_valid, dtype=torch.bool).reshape(
                    batch, tubelets, tubelet_size
                ).any(dim=2)
                feature_parts.append(tubelet_features.half().cpu().numpy())
                time_parts.append(tubelet_times.numpy())
                valid_parts.append(tubelet_valid.numpy())
        finally:
            stop_decode.set()
            producer.join(timeout=10.0)
            cap.release()
        if producer.is_alive():
            raise RuntimeError(f"decode producer did not stop for {video_id}")

        features = np.concatenate(feature_parts, axis=0).reshape(-1, backbone.output_feature_dim)
        timestamps = np.concatenate(time_parts, axis=0).reshape(-1)
        valid = np.concatenate(valid_parts, axis=0).reshape(-1)
        features = features[valid]
        timestamps = timestamps[valid]
        if timestamps.size == 0 or np.any(np.diff(timestamps) <= 0):
            raise RuntimeError(f"invalid stitched tubelet timeline for {video_id}")
        video_output.mkdir(parents=True, exist_ok=True)
        # Uncompressed .npy arrays are intentional: the temporal trainer can
        # mmap arbitrary 120s blocks without repeatedly decompressing a whole
        # match into every DataLoader worker.
        np.save(feature_path, features.astype(np.float16), allow_pickle=False)
        np.save(timestamp_path, timestamps.astype(np.float32), allow_pickle=False)
        metadata = {
            "schema": SCHEMA,
            "video_id": video_id,
            "split": args.split,
            "source_video": str(source_video),
            "source_size_bytes": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "config": str(a1_config_path),
            "config_sha256": config_sha,
            "chunk_seconds": args.chunk_seconds,
            "prefetch_batches": args.prefetch_batches,
            "tubelet_seconds": args.chunk_seconds / (frame_count // tubelet_size),
            "tubelet_count": int(timestamps.size),
            "feature_dim": int(features.shape[-1]),
            "feature_dtype": "float16",
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        row = {
            "video_id": video_id,
            "source_video": str(source_video),
            "duration_seconds": duration,
            "tubelet_count": int(timestamps.size),
            "decode_quality_mean": float(np.mean(decode_quality)),
            "elapsed_seconds": time.time() - started,
            "output": str(video_output),
        }
        shard_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    shard_manifest = {
        "schema": SCHEMA,
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "config": str(a1_config_path),
        "config_sha256": config_sha,
        "chunk_seconds": args.chunk_seconds,
        "prefetch_batches": args.prefetch_batches,
        "videos": shard_rows,
    }
    manifest_path = output_root / f"manifest_{args.split}_shard{args.shard_index}.json"
    manifest_path.write_text(
        json.dumps(shard_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
