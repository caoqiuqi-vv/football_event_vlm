from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT / "src", WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from football_longform_v2.backbones import BackboneError, build_plain_backbone  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.feature_store import align_feature_stream  # noqa: E402

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV")
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def resolve(project_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_video(root: Path, video_id: str) -> Path:
    for suffix in VIDEO_SUFFIXES:
        candidate = root / f"{video_id}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(
        path for path in root.glob(f"{video_id}*")
        if path.is_file() and path.suffix in VIDEO_SUFFIXES
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one video for {video_id}, found {matches}")
    return matches[0]


def preprocess(frame_bgr: np.ndarray, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (width, height)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0)
    return (tensor - MEAN) / STD


def timeline_metadata(
    *,
    arch: str,
    backbone_id: str,
    weights: Path,
    video: Path,
    max_seconds: float | None,
    config_sha256: str | None = None,
    weights_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    """Return serializable provenance fields for a timeline cache."""
    metadata = {
        "backbone_arch": np.asarray(arch),
        "backbone_id": np.asarray(backbone_id),
        "backbone_weights": np.asarray(str(weights)),
        "source_video": np.asarray(str(video)),
        "max_seconds": np.asarray(float(max_seconds) if max_seconds is not None else np.nan),
    }
    if config_sha256 is not None:
        metadata["config_sha256"] = np.asarray(config_sha256)
    if weights_sha256 is not None:
        metadata["weights_sha256"] = np.asarray(weights_sha256)
    return metadata


@torch.inference_mode()
def infer_context(
    model: torch.nn.Module, frames: list[torch.Tensor], device: torch.device
) -> np.ndarray:
    inputs = torch.stack(frames).to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model.forward_features(inputs)
        features = torch.cat(
            [output["x_norm_clstoken"], output["x_norm_patchtokens"].mean(dim=1)], dim=-1
        )
    return features.float().cpu().numpy()


def extract_streams(
    video: Path,
    model: torch.nn.Module,
    *,
    device: torch.device,
    image_size: tuple[int, int],
    context_hz: float,
    motion_hz: float,
    motion_grid: tuple[int, int],
    context_batch_size: int,
    max_seconds: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid FPS for {video}: {fps}")
    context_period, motion_period = 1.0 / context_hz, 1.0 / motion_hz
    next_motion_time = next_context_time = 0.0
    frame_index = 0
    previous_gray: np.ndarray | None = None
    context_times: list[float] = []
    context_batches: list[np.ndarray] = []
    pending_frames: list[torch.Tensor] = []
    motion_times: list[float] = []
    motion_features: list[np.ndarray] = []

    def flush_context() -> None:
        if pending_frames:
            context_batches.append(infer_context(model, pending_frames, device))
            pending_frames.clear()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_index / fps
        frame_index += 1
        if max_seconds is not None and timestamp > max_seconds:
            break
        if timestamp + 0.5 / fps < next_motion_time:
            continue
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 64))
        if previous_gray is None:
            descriptor = np.zeros(motion_grid, dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))
            descriptor = np.log1p(cv2.resize(
                magnitude, (motion_grid[1], motion_grid[0]), interpolation=cv2.INTER_AREA
            )).astype(np.float32)
        previous_gray = gray
        motion_times.append(timestamp)
        motion_features.append(descriptor.reshape(-1))
        next_motion_time += motion_period
        if timestamp + 0.5 * motion_period >= next_context_time:
            context_times.append(timestamp)
            pending_frames.append(preprocess(frame, image_size))
            next_context_time += context_period
            if len(pending_frames) >= context_batch_size:
                flush_context()
    capture.release()
    flush_context()
    if not context_batches or not motion_features:
        raise RuntimeError(f"no samples extracted from {video}")
    return (
        np.asarray(context_times, dtype=np.float32),
        np.concatenate(context_batches).astype(np.float32),
        np.asarray(motion_times, dtype=np.float32),
        np.stack(motion_features).astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-batch-size", type=int, default=8)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])
    features = config["features"]
    context, motion = features["context"], features["motion"]
    video = find_video(Path(args.video_root).resolve(), args.video_id)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else resolve(project_root, config["paths"]["feature_store"])
        / args.video_id / "timeline.npz"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    device = torch.device(args.device)
    arch = str(context.get("arch", ""))
    weights = resolve(project_root, str(context["weights"]))
    try:
        model = build_plain_backbone(arch, weights).to(device).eval()
    except BackboneError as error:
        raise RuntimeError(f"cannot build configured context backbone: {error}") from error
    context_times, context_values, motion_times, motion_values = extract_streams(
        video, model, device=device,
        image_size=tuple(map(int, context["image_size"])),
        context_hz=float(context["source_hz"]),
        motion_hz=float(motion["source_hz"]),
        motion_grid=tuple(map(int, motion["grid_size"])),
        context_batch_size=args.context_batch_size,
        max_seconds=args.max_seconds,
    )
    del model
    end_time = max(float(context_times[-1]), float(motion_times[-1]))
    timeline_hz = float(features["timeline_hz"])
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
    np.savez_compressed(
        output,
        timestamps=target_times.numpy().astype(np.float32),
        context=aligned_context.numpy().astype(np.float16),
        motion=aligned_motion.numpy().astype(np.float16),
        context_valid=context_valid.numpy(),
        motion_valid=motion_valid.numpy(),
        **timeline_metadata(
            arch=arch,
            backbone_id=str(context.get("backbone_id", weights.stem)),
            weights=weights,
            video=video,
            max_seconds=args.max_seconds,
            config_sha256=hashlib.sha256(Path(config["_config_path"]).read_bytes()).hexdigest(),
            weights_sha256=sha256_file(weights),
        ),
    )
    print(
        f"video={video} output={output} timeline_steps={target_times.numel()} "
        f"context={context_values.shape} motion={motion_values.shape} backbone={context.get('backbone_id', weights.stem)}", flush=True
    )


if __name__ == "__main__":
    main()
