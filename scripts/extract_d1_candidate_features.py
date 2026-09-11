#!/usr/bin/env python
"""Extract frozen DINOv3 ViT-L/16 features around D1 candidates (GPU; queued).

Spec: docs/football_tail_first_fullchain_plan_20260910.md §5 (D1 候选点高帧率核验器).

Per candidate: 48 frames at 8 fps over peak_time ± 3s (bin centers,
t_i = peak + (i - 23.5)/8 — identical formula to build_d1_examiner_manifest.py),
720p native resolution with the E16 preprocessing of train_football_events.py
(BGR->RGB, cv2.resize INTER_CUBIC to 720x1280, /255, ImageNet mean/std; the
/255+normalize runs on GPU — E16 config uses preprocessing.normalize_on_device=true).

Decoding keeps the dense-evaluation-line semantics (cv2.VideoCapture FFMPEG
backend, one seek then monotonic grab-walk, same as
train_football_events._decode_frames_single_seek) but is STRICTLY STREAMING —
v1 materialized the whole union of wanted frames as raw BGR in a Python list
(~44k frames x 2.76MB ~ 122GB for the largest video) which caused swap thrash
and OOM SIGKILLs. v2 pipeline per video:

  unique frame indices (sorted; candidate windows share frame indices exactly
  because all peaks sit on the 5s grid — decoded/forwarded once per video)
    -> K decoder threads, each with its own capture walking a contiguous chunk
       (one seek per chunk), decode + cvtColor + resize to uint8 RGB
    -> bounded queue (--queue-frames, uint8 2.76MB/frame)
    -> main thread stacks --batch-size frames, uint8 H2D copy, normalize on GPU,
       frozen DINO forward, fp16 features scattered into a preallocated
       [n_unique, D] array (n_unique*4KB ~ <=0.3GB/video)
    -> after the walk: nearest-valid fallback for failed frames, scatter to
       candidates [N,48,D] fp16, atomic npz write.

Resident memory is bounded by: queue + one batch + feats_unique + model +
CUDA context (~5GB total at defaults; hard guard via --max-rss-gb, exit code 3).

Multi-GPU: `--gpu-ids 0,2,3` wraps the backbone in nn.DataParallel (single
process; decode stays on CPU). For larger fleets prefer launching one process
per GPU sharded by `--video-ids`.

Logging goes through the logging module (StreamHandler flushes every record —
no PYTHONUNBUFFERED needed) with per-video progress/ETA/RSS.

`--decode-only` runs seek-precision QA on a few candidates without the GPU;
`--av-crosscheck` additionally validates timestamps with an independent PyAV seek.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import resource
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

NUM_FRAMES = 48
SAMPLE_FPS = 8.0
EMBED_DIM = 1024
IMAGE_SIZE = (720, 1280)  # (H, W), E16 配置

DEFAULT_WEIGHTS = "checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

LOG = logging.getLogger("d1_extract")
LOG.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
LOG.addHandler(_handler)
LOG.propagate = False

_SENTINEL = object()


def candidate_sample_times(peak_time_sec: float) -> np.ndarray:
    """48 bin centers of [peak-3s, peak+3s); symmetric around the peak."""
    offsets = (np.arange(NUM_FRAMES, dtype=np.float64) - (NUM_FRAMES - 1) / 2.0) / SAMPLE_FPS
    return float(peak_time_sec) + offsets


def times_to_frame_indices(times: np.ndarray, fps: float, total_frames: int) -> np.ndarray:
    indices = np.round(times * fps).astype(np.int64)
    return np.clip(indices, 0, max(total_frames - 1, 0))


def rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


class DinoFeatureExtractor:
    """Frozen DINOv3 ViT-L/16 -> per-frame [CLS || mean(patch)] or CLS features."""

    def __init__(self, weights: str, gpu_ids: list[int], feature_mode: str, batch_size: int):
        import torch
        from torch import nn

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available; run with --decode-only for CPU-only checks"
            )
        from dinov3.hub.backbones import dinov3_vitl16
        from train_football_events import IMAGENET_MEAN, IMAGENET_STD

        self.torch = torch
        self.feature_mode = feature_mode
        self.batch_size = batch_size
        self.device = torch.device(f"cuda:{gpu_ids[0]}")
        backbone = dinov3_vitl16(pretrained=True, weights=weights)
        backbone.eval()
        backbone.requires_grad_(False)

        class _Head(nn.Module):
            def __init__(self, backbone_module: nn.Module, feature_mode: str) -> None:
                super().__init__()
                # must be a registered submodule so .to(device) / DataParallel
                # move/replicate it (a closure capture stays on CPU)
                self.backbone = backbone_module
                self.feature_mode = feature_mode
                # normalize on device, same math as E16 normalize_on_device=true
                self.register_buffer("in_mean", IMAGENET_MEAN.reshape(1, 3, 1, 1).clone(), persistent=False)
                self.register_buffer("in_std", IMAGENET_STD.reshape(1, 3, 1, 1).clone(), persistent=False)

            def forward(self, x_u8):
                # x_u8: uint8 [B, H, W, 3] straight from cv2 (RGB)
                x = x_u8.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
                x = (x - self.in_mean) / self.in_std
                out = self.backbone.forward_features(x)
                cls = out["x_norm_clstoken"]
                if self.feature_mode == "cls":
                    return cls
                patch_mean = out["x_norm_patchtokens"].mean(dim=1)
                return torch.cat([cls, patch_mean], dim=-1)

        model: nn.Module = _Head(backbone, feature_mode).to(self.device)
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
        self.model = model
        self.out_dim = EMBED_DIM if feature_mode == "cls" else 2 * EMBED_DIM
        param = next(self.model.parameters())
        assert param.device.type == "cuda", f"backbone not on GPU: {param.device}"
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
        self.model = model
        self.out_dim = EMBED_DIM if feature_mode == "cls" else 2 * EMBED_DIM

    def forward_uint8(self, frames_u8: np.ndarray) -> np.ndarray:
        """frames_u8: uint8 numpy [B,H,W,3] (RGB, 720x1280, cv2 layout).
        Returns fp16 [B,D]."""
        torch = self.torch
        feats: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, frames_u8.shape[0], self.batch_size):
                chunk = torch.from_numpy(frames_u8[start : start + self.batch_size])
                chunk = chunk.pin_memory().to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self.model(chunk)
                feats.append(out.float().cpu().numpy().astype(np.float16))
        return np.concatenate(feats, axis=0)


def iter_chunk_frames(cap, chunk_indices: np.ndarray):
    """Streaming variant of train_football_events._decode_frames_single_seek:
    one seek to the first wanted index, then a monotonic grab-walk.
    Yields (local_pos, frame|None) — frames are never accumulated."""
    import cv2

    if len(chunk_indices) == 0:
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(chunk_indices[0]))
    current = int(chunk_indices[0])
    last_frame = None
    for local_pos, raw_index in enumerate(chunk_indices):
        index = int(raw_index)
        if last_frame is not None and index < current:
            yield local_pos, last_frame.copy()
            continue
        ok = True
        while current < index:
            try:
                ok = cap.grab()
            except cv2.error:
                ok = False
            current += 1
            if not ok:
                break
        if not ok:
            yield local_pos, None
            continue
        try:
            ok, frame = cap.read()
        except cv2.error:
            ok, frame = False, None
        current += 1
        if ok and frame is not None:
            last_frame = frame
            yield local_pos, frame
        else:
            yield local_pos, None


def _decoder_worker(
    video_path: str,
    chunk_indices: np.ndarray,
    chunk_offset: int,
    out_queue: "queue.Queue",
    stop_event: threading.Event,
) -> None:
    import cv2

    from train_football_events import _open_video_capture

    cap = _open_video_capture(video_path)
    try:
        for local_pos, frame in iter_chunk_frames(cap, chunk_indices):
            if stop_event.is_set():
                break
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(
                    frame, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_CUBIC
                )
            out_queue.put((chunk_offset + local_pos, frame))
    finally:
        cap.release()
        out_queue.put(_SENTINEL)


def decode_candidate_frames(cap, unique_indices: np.ndarray) -> list:
    """Materializing walk used ONLY by --decode-only QA (never by extraction)."""
    from train_football_events import _decode_frames_single_seek

    return _decode_frames_single_seek(cap, [int(i) for i in unique_indices])


def stream_video_features(
    video_id: str,
    video_path: str,
    unique_indices: np.ndarray,
    extractor: DinoFeatureExtractor,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decode+forward the union of wanted frames, streaming. Returns
    (feats_unique [n_unique, D] fp16, valid_mask, stats)."""
    torch = extractor.torch
    n_unique = len(unique_indices)
    feats_unique = np.zeros((n_unique, extractor.out_dim), dtype=np.float16)
    valid_mask = np.zeros(n_unique, dtype=bool)

    chunks = [c for c in np.array_split(unique_indices, max(args.decode_workers, 1)) if len(c)]
    out_queue: queue.Queue = queue.Queue(maxsize=args.queue_frames)
    stop_event = threading.Event()
    offset = 0
    threads = []
    for chunk in chunks:
        t = threading.Thread(
            target=_decoder_worker,
            args=(video_path, chunk, offset, out_queue, stop_event),
            daemon=True,
        )
        t.start()
        threads.append(t)
        offset += len(chunk)

    max_rss = float(args.max_rss_gb) * 2**30
    positions: list[int] = []
    frames: list[np.ndarray] = []
    processed = 0
    n_decode_fail = 0
    finished = 0
    t0 = time.monotonic()
    t_log = t0
    n_log = 0

    def flush_batch() -> None:
        nonlocal positions, frames, processed
        if not frames:
            return
        arr = np.stack(frames)  # uint8 [B,3,H,W]
        feats = extractor.forward_uint8(arr)
        pos_arr = np.asarray(positions, dtype=np.int64)
        feats_unique[pos_arr] = feats
        valid_mask[pos_arr] = True
        processed += len(frames)
        positions, frames = [], []

    while finished < len(threads):
        item = out_queue.get()
        if item is _SENTINEL:
            finished += 1
            continue
        pos, frame = item
        if frame is None:
            n_decode_fail += 1
            continue
        positions.append(pos)
        frames.append(frame)
        if len(frames) >= args.batch_size:
            flush_batch()
            if rss_gb() * 2**30 > max_rss:
                stop_event.set()
                LOG.critical(
                    "%s: RSS %.1fGB exceeded --max-rss-gb %.1f, exiting(3)",
                    video_id, rss_gb(), args.max_rss_gb,
                )
                sys.exit(3)
            now = time.monotonic()
            if processed - n_log >= args.log_every_frames:
                rate = processed / max(now - t0, 1e-9)
                eta = (n_unique - processed) / max(rate, 1e-9)
                LOG.info(
                    "%s: %d/%d frames (%.1f f/s, ETA %.0fs) rss=%.1fGB",
                    video_id, processed, n_unique, rate, eta, rss_gb(),
                )
                t_log = now
                n_log = processed
    flush_batch()
    for t in threads:
        t.join()

    # last-resort fallback for failed frames: reuse nearest valid unique frame
    n_valid = int(valid_mask.sum())
    if n_valid and n_valid < n_unique:
        valid_positions = np.flatnonzero(valid_mask)
        for pos in np.flatnonzero(~valid_mask):
            nearest = valid_positions[np.argmin(np.abs(valid_positions - pos))]
            feats_unique[pos] = feats_unique[nearest]
        valid_mask[~valid_mask] = True
    stats = {
        "n_unique_frames": int(n_unique),
        "n_decoded_valid": n_valid,
        "n_decode_fail": n_decode_fail,
        "decode_forward_sec": round(time.monotonic() - t0, 2),
        "frames_per_sec": round(n_unique / max(time.monotonic() - t0, 1e-9), 1),
    }
    return feats_unique, valid_mask, stats


def process_video(
    video_id: str,
    video_path: str,
    group: pd.DataFrame,
    extractor: DinoFeatureExtractor | None,
    output_dir: Path,
    *,
    decode_only: bool,
    decode_probe_candidates: int,
    av_crosscheck: bool,
    args: argparse.Namespace,
) -> dict:
    import cv2

    from train_football_events import _open_video_capture

    t_start = time.monotonic()
    cap = _open_video_capture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"invalid video metadata for {video_path}: fps={fps} frames={total_frames}")

    candidate_idx = group["candidate_idx"].to_numpy(dtype=np.int64)
    peaks = group["peak_time_sec"].to_numpy(dtype=np.float64)
    per_cand_indices = np.stack(
        [times_to_frame_indices(candidate_sample_times(p), fps, total_frames) for p in peaks]
    )  # [N, 48]
    per_cand_times = np.stack([candidate_sample_times(p) for p in peaks])  # [N, 48]

    unique_indices, inverse = np.unique(per_cand_indices.reshape(-1), return_inverse=True)
    inverse = inverse.reshape(per_cand_indices.shape)

    if decode_only:
        # CPU smoke mode: probe seek precision on up to --decode-probe-candidates
        # candidates using the real extraction decode walk (single_seek), then
        # fresh per-index seeks for timestamp evidence. No union decode, no GPU.
        report = {
            "video_id": video_id,
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "n_candidates": len(group),
            "n_unique_frames_full_video": int(len(unique_indices)),
        }
        probe_rows = []
        n_probe = min(decode_probe_candidates, len(group))
        for ci in range(n_probe):
            want = [int(i) for i in per_cand_indices[ci]]
            # 1) extraction-path walk: single seek + sequential grab/read
            t0 = time.monotonic()
            walk = decode_candidate_frames(cap, np.asarray(want))
            walk_sec = time.monotonic() - t0
            pos_frames_end = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            walk_ok = sum(f is not None for f in walk)
            # 2) fresh seek to the window's first frame -> timestamp evidence
            first = want[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            ok, _frame = cap.read()
            pos_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            expected = first / fps
            err_now = pos_msec / 1000.0 - expected
            err_next = pos_msec / 1000.0 - (first + 1) / fps  # some builds report next-frame pos
            probe_rows.append(
                {
                    "candidate_idx": int(candidate_idx[ci]),
                    "peak_time_sec": float(peaks[ci]),
                    "first_frame_index": first,
                    "first_frame_expected_time_sec": round(expected, 4),
                    "walk_decoded_ok": f"{walk_ok}/{NUM_FRAMES}",
                    "walk_sec": round(walk_sec, 3),
                    "pos_frames_after_walk": pos_frames_end,
                    "pos_frames_expect_after_walk": want[-1] + 1,
                    "fresh_seek_ok": bool(ok),
                    "pos_frames_after_fresh_read": pos_frames,
                    "pos_msec_after_fresh_read": round(pos_msec, 1),
                    "err_if_pts_of_read_sec": round(err_now, 4),
                    "err_if_pts_of_next_sec": round(err_next, 4),
                }
            )
        report["seek_probe"] = probe_rows
        errs = [
            min(abs(r["err_if_pts_of_read_sec"]), abs(r["err_if_pts_of_next_sec"]))
            for r in probe_rows
            if r["fresh_seek_ok"]
        ]
        report["seek_abs_err_max_sec"] = max(errs) if errs else None
        report["seek_abs_err_mean_sec"] = float(np.mean(errs)) if errs else None
        if av_crosscheck:
            report["av_crosscheck"] = av_timestamp_crosscheck(
                video_path, fps, per_cand_times[0, [0, NUM_FRAMES // 2, -1]]
            )
        cap.release()
        return report

    assert extractor is not None
    LOG.info(
        "%s: %d candidates, %d unique frames (fps=%.3f, %d total)",
        video_id, len(group), len(unique_indices), fps, total_frames,
    )
    cap.release()  # streaming decode uses per-worker captures
    feats_unique, _valid_mask, stats = stream_video_features(
        video_id, video_path, unique_indices, extractor, args
    )

    feats_cand = feats_unique[inverse]  # [N, 48, D] fp16
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}.npz"
    tmp_path = output_dir / f".{video_id}.npz.tmp"
    with tmp_path.open("wb") as handle:  # file handle: np.savez won't append .npz
        np.savez(
            handle,
            candidate_idx=candidate_idx,
            frame_indices=per_cand_indices.astype(np.int32),
            feats=feats_cand.astype(np.float16),
        )
    os.replace(tmp_path, out_path)
    return {
        "video_id": video_id,
        "n_candidates": len(group),
        **stats,
        "elapsed_sec": round(time.monotonic() - t_start, 2),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "npz": str(out_path),
        "npz_bytes": out_path.stat().st_size,
    }


def av_timestamp_crosscheck(video_path: str, fps: float, times_sec: np.ndarray) -> list[dict]:
    """Independent PyAV seek used only by --decode-only QA."""
    import av

    rows = []
    container = av.open(video_path)
    stream = container.streams.video[0]
    for t in times_sec:
        container.seek(max(int((float(t) - 1.0) * av.time_base), 0), any_frame=False, backward=True, stream=stream)
        got = None
        for frame in container.decode(stream):
            ts = float(frame.pts * stream.time_base) if frame.pts is not None else None
            if ts is None:
                continue
            if ts >= float(t) - 0.5 / fps:
                got = ts
                break
        rows.append(
            {
                "target_time_sec": round(float(t), 4),
                "av_frame_time_sec": None if got is None else round(got, 4),
                "abs_err_sec": None if got is None else round(abs(got - float(t)), 4),
            }
        )
    container.close()
    return rows


def video_complete(path: Path, expected_candidate_ids: set[int]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            if "candidate_idx" not in data or "feats" not in data:
                return False
            have = set(int(i) for i in data["candidate_idx"])
            return expected_candidate_ids.issubset(have)
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/football_d1_examiner/manifest_20260911.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/football_d1_examiner/features"))
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--gpu-ids", default="0", help="comma-separated; >1 id -> DataParallel")
    parser.add_argument("--feature-mode", choices=["cls_patchmean", "cls"], default="cls_patchmean")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-workers", type=int, default=3, help="decoder threads per video, each walks a contiguous chunk")
    parser.add_argument("--queue-frames", type=int, default=192, help="bounded decode queue (uint8 frames, ~2.8MB each)")
    parser.add_argument("--max-rss-gb", type=float, default=20.0, help="graceful exit(3) above this RSS")
    parser.add_argument("--log-every-frames", type=int, default=4000)
    parser.add_argument("--cv2-threads", type=int, default=2)
    parser.add_argument("--limit-videos", type=int, default=0, help="debug: only first N videos")
    parser.add_argument("--video-ids", default="", help="comma-separated video_id filter")
    parser.add_argument("--decode-only", action="store_true", help="CPU decode QA; no GPU, no features")
    parser.add_argument("--decode-probe-candidates", type=int, default=2, help="candidates probed per video in --decode-only")
    parser.add_argument("--av-crosscheck", action="store_true", help="with --decode-only: PyAV timestamp crosscheck")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2

    cv2.setNumThreads(max(int(args.cv2_threads), 0))
    manifest = pd.read_parquet(args.manifest)
    if args.video_ids:
        keep = set(args.video_ids.split(","))
        manifest = manifest[manifest["video_id"].isin(keep)]
    if manifest.empty:
        raise ValueError("manifest filter produced zero candidates")

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    extractor = None
    if not args.decode_only:
        t0 = time.monotonic()
        extractor = DinoFeatureExtractor(args.weights, gpu_ids, args.feature_mode, args.batch_size)
        LOG.info("backbone loaded in %.1fs (gpu_ids=%s, mode=%s)", time.monotonic() - t0, gpu_ids, args.feature_mode)

    reports: list[dict] = []
    grouped = list(manifest.groupby("video_id", sort=True))
    if args.limit_videos:
        grouped = grouped[: args.limit_videos]
    LOG.info("videos to process: %d (resume=%s)", len(grouped), args.resume)
    run_t0 = time.monotonic()
    done_frames = 0
    for i, (video_id, group) in enumerate(grouped, 1):
        group = group.sort_values("candidate_idx")
        video_path = str(group["video_path"].iloc[0])
        out_path = args.output_dir / f"{video_id}.npz"
        expected = set(int(j) for j in group["candidate_idx"])
        if not args.decode_only and args.resume and video_complete(out_path, expected):
            LOG.info("[%d/%d] %s: complete, skip", i, len(grouped), video_id)
            continue
        t0 = time.monotonic()
        report = process_video(
            video_id,
            video_path,
            group,
            extractor,
            args.output_dir,
            decode_only=args.decode_only,
            decode_probe_candidates=args.decode_probe_candidates,
            av_crosscheck=args.av_crosscheck,
            args=args,
        )
        reports.append(report)
        if args.decode_only:
            LOG.info(
                "[%d/%d] %s: cand=%d seek_err_max=%ss %.1fs",
                i, len(grouped), video_id, report["n_candidates"],
                report["seek_abs_err_max_sec"], time.monotonic() - t0,
            )
            LOG.info("\n%s", json.dumps(report, ensure_ascii=False, indent=2))
        else:
            done_frames += report["n_unique_frames"]
            elapsed = time.monotonic() - run_t0
            LOG.info(
                "[%d/%d] %s: cand=%d uniq=%d fail=%d %.1f f/s, video %.1fs, peak_rss=%.1fGB, "
                "npz=%.1fMB | run elapsed %.0fs",
                i, len(grouped), video_id, report["n_candidates"], report["n_unique_frames"],
                report["n_decode_fail"], report["frames_per_sec"], report["elapsed_sec"],
                report["peak_rss_gb"], report["npz_bytes"] / 2**20, elapsed,
            )

    summary_path = args.output_dir / (
        "extract_decode_only_report.json" if args.decode_only else "extract_run_report.json"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # don't clobber a previous report with an empty one on resume-skip-only runs
    if reports or not summary_path.exists():
        summary_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not reports:
        LOG.info("nothing new processed; kept existing %s", summary_path)
    else:
        LOG.info("wrote %s", summary_path)


if __name__ == "__main__":
    main()
