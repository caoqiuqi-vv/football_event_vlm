from __future__ import annotations

"""Build a model-free 2 FPS JPEG-LMDB and aligned log-mel store once/video."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import librosa
import lmdb
import numpy as np
import torch
import yaml


SCHEMA = "football_e2e_spotter.pixel_audio_store.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def letterbox(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(int(round(source_width * scale)), 1)
    resized_height = max(int(round(source_height * scale)), 1)
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    output[y:y + resized_height, x:x + resized_width] = resized
    return output


def extract_log_mel(
    video: Path,
    *,
    frame_count: int,
    sample_fps: float,
    sample_rate: int,
    mel_bins: int,
) -> np.ndarray:
    samples_per_step = int(round(sample_rate / sample_fps))
    if samples_per_step <= 0:
        raise ValueError("audio samples per output step must be positive")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("ffmpeg did not expose stdout")
    mel_filter = torch.from_numpy(librosa.filters.mel(
        sr=sample_rate, n_fft=512, n_mels=mel_bins, fmin=40.0,
        fmax=min(7600.0, 0.5 * sample_rate),
    ).astype(np.float32))
    window = torch.hann_window(512)
    rows: list[np.ndarray] = []
    batch_steps = 128
    try:
        for start in range(0, frame_count, batch_steps):
            count = min(batch_steps, frame_count - start)
            raw = read_exact(process.stdout, count * samples_per_step * 2)
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            expected = count * samples_per_step
            if samples.size < expected:
                samples = np.pad(samples, (0, expected - samples.size))
            samples = torch.from_numpy(samples[:expected].reshape(count, samples_per_step))
            spectrum = torch.stft(
                samples, n_fft=512, hop_length=160, win_length=512,
                window=window, center=True, return_complex=True,
            ).abs().square()
            mel = torch.matmul(mel_filter, spectrum).mean(dim=-1)
            rows.append(torch.log1p(mel).numpy().astype(np.float16))
    finally:
        if process.stdout is not None:
            process.stdout.close()
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg audio decode failed ({return_code}): {stderr[-1000:]}")
    output = np.concatenate(rows, axis=0) if rows else np.zeros((0, mel_bins), np.float16)
    if output.shape != (frame_count, mel_bins):
        raise RuntimeError(f"audio feature shape mismatch: {output.shape}")
    return output


def build_one(
    item: dict,
    *,
    split: str,
    output_root: str,
    sample_fps: float,
    image_size: tuple[int, int],
    jpeg_quality: int,
    audio_sample_rate: int,
    audio_mel_bins: int,
    config_sha256: str,
) -> dict:
    video_id = str(item["media_id"])
    source = Path(item["source_video"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_stat = source.stat()
    destination = Path(output_root).resolve() / split / video_id
    metadata_path = destination / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ready = (
            metadata.get("schema") == SCHEMA
            and metadata.get("config_sha256") == config_sha256
            and metadata.get("source_video") == str(source)
            and int(metadata.get("source_size_bytes", -1)) == source_stat.st_size
            and int(metadata.get("source_mtime_ns", -1)) == source_stat.st_mtime_ns
            and (destination / "frames.lmdb" / "data.mdb").is_file()
            and (destination / "audio_logmel.npy").is_file()
        )
        if ready:
            return {"video_id": video_id, "status": "skip_ready", **metadata}
        raise RuntimeError(f"stale pixel/audio cache exists: {destination}")
    if destination.exists():
        raise RuntimeError(f"incomplete destination exists and will not be overwritten: {destination}")

    started = time.time()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(source_fps) or source_fps <= 0 or source_frames <= 0:
        cap.release()
        raise RuntimeError(f"invalid video geometry: {source}")
    duration = source_frames / source_fps
    frame_count = max(int(math.floor(duration * sample_fps)), 1)
    temp = destination.with_name(f".{video_id}.{os.getpid()}.tmp")
    temp.mkdir(parents=True, exist_ok=False)
    lmdb_path = temp / "frames.lmdb"
    # map_size is a virtual address ceiling; actual disk use follows JPEG bytes.
    map_size = max(frame_count * 160_000 + 64 * 1024 * 1024, 128 * 1024 * 1024)
    environment = lmdb.open(str(lmdb_path), map_size=map_size, subdir=True, lock=True)
    position = 0
    last_frame: np.ndarray | None = None
    decoded = 0
    transaction = environment.begin(write=True)
    try:
        for index in range(frame_count):
            timestamp = (index + 0.5) / sample_fps
            target = min(max(int(round(timestamp * source_fps)), 0), source_frames - 1)
            ok = True
            while position < target:
                ok = bool(cap.grab())
                position += 1
                if not ok:
                    break
            if ok:
                ok, frame = cap.read()
                position += 1
            else:
                frame = None
            if ok and frame is not None:
                last_frame = frame
                decoded += 1
            elif last_frame is not None:
                frame = last_frame
            else:
                frame = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            frame = letterbox(frame, image_size[0], image_size[1])
            encoded, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            )
            if not encoded:
                raise RuntimeError(f"JPEG encode failed for {video_id} frame {index}")
            transaction.put(f"f/{index:08d}".encode(), buffer.tobytes())
            if (index + 1) % 512 == 0:
                transaction.commit()
                transaction = environment.begin(write=True)
        transaction.commit()
    except BaseException:
        transaction.abort()
        raise
    finally:
        environment.sync()
        environment.close()
        cap.release()

    audio = extract_log_mel(
        source, frame_count=frame_count, sample_fps=sample_fps,
        sample_rate=audio_sample_rate, mel_bins=audio_mel_bins,
    )
    np.save(temp / "audio_logmel.npy", audio, allow_pickle=False)
    metadata = {
        "schema": SCHEMA,
        "video_id": video_id,
        "annotation_id": str(item["annotation_id"]),
        "split": split,
        "source_video": str(source),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "duration_seconds": duration,
        "source_fps": source_fps,
        "source_frames": source_frames,
        "sample_fps": sample_fps,
        "frame_count": frame_count,
        "timestamp_rule": "(frame_index+0.5)/sample_fps",
        "image_size": list(image_size),
        "jpeg_quality": jpeg_quality,
        "audio_sample_rate": audio_sample_rate,
        "audio_mel_bins": audio_mel_bins,
        "audio_shape": list(audio.shape),
        "decode_quality": decoded / max(frame_count, 1),
        "config_sha256": config_sha256,
        "model_features_cached": False,
        "detection_or_tracking_used": False,
    }
    (temp / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temp.rename(destination)
    return {
        "video_id": video_id, "status": "built", "elapsed_seconds": time.time() - started,
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "calibration"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.workers <= 0 or args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid workers or shard")
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    manifest_path = Path(data["canonical_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = (config_path.parents[2] / manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = list(manifest[args.split])[args.shard_index::args.num_shards]
    if args.max_videos is not None:
        items = items[:max(int(args.max_videos), 0)]
    preprocessing_contract = {
        "schema": SCHEMA,
        "sample_fps": float(data["sample_fps"]),
        "image_size": [int(value) for value in data["image_size"]],
        "jpeg_quality": int(data["jpeg_quality"]),
        "audio_sample_rate": int(data["audio_sample_rate"]),
        "audio_mel_bins": int(data["audio_mel_bins"]),
        "timestamp_rule": "(frame_index+0.5)/sample_fps",
        "canonical_manifest_sha256": sha256_file(manifest_path),
    }
    preprocessing_sha256 = hashlib.sha256(json.dumps(
        preprocessing_contract, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    common = {
        "split": args.split,
        "output_root": str(Path(data["frame_store"]).expanduser().resolve()),
        "sample_fps": float(data["sample_fps"]),
        "image_size": tuple(int(value) for value in data["image_size"]),
        "jpeg_quality": int(data["jpeg_quality"]),
        "audio_sample_rate": int(data["audio_sample_rate"]),
        "audio_mel_bins": int(data["audio_mel_bins"]),
        # Only preprocessing fields participate in cache identity. Changing an
        # optimizer or model hyperparameter must not invalidate pixel data.
        "config_sha256": preprocessing_sha256,
    }
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_one, item, **common) for item in items]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({
                key: row[key] for key in (
                    "video_id", "status", "frame_count", "duration_seconds", "decode_quality"
                ) if key in row
            }, ensure_ascii=False), flush=True)
    output_root = Path(common["output_root"])
    report = {
        "schema": SCHEMA,
        "split": args.split,
        "config": str(config_path),
        "preprocessing_contract": preprocessing_contract,
        "preprocessing_sha256": common["config_sha256"],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "requested": len(items),
        "completed": len(rows),
        "rows": sorted(rows, key=lambda row: row["video_id"]),
        "sealed_test_media_opened": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"build_{args.split}_shard{args.shard_index}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
