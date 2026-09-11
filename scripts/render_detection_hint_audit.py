#!/usr/bin/env python
"""Render random recovered-ball frames as original/hint side-by-side images."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_detection_aware import DetectionHintRenderer  # noqa: E402


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-ball", type=int, default=3)
    return parser.parse_args()


def video_path(videos_dir: Path, video_id: str) -> Path:
    for extension in VIDEO_EXTENSIONS:
        candidate = videos_dir / f"{video_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing video for {video_id} under {videos_dir}")


def recovered_frames(path: Path) -> list[int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sources = payload.get("object_sources")
    if sources is None:
        return []
    frame_ids = payload["frame_ids"].numpy()
    offsets = payload["frame_offsets"].numpy()
    source_values = sources.numpy()
    result = []
    for position, frame_id in enumerate(frame_ids):
        lo, hi = int(offsets[position]), int(offsets[position + 1])
        if bool((source_values[lo:hi] > 0).any()):
            result.append(int(frame_id))
    return result


def reservoir_samples(
    index_root: Path, num_samples: int, seed: int
) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    reservoir: list[tuple[str, int]] = []
    seen = 0
    for path in sorted(index_root.glob("*.pt")):
        for frame_id in recovered_frames(path):
            item = (path.stem, frame_id)
            seen += 1
            if len(reservoir) < num_samples:
                reservoir.append(item)
            else:
                replacement = rng.randrange(seen)
                if replacement < num_samples:
                    reservoir[replacement] = item
    rng.shuffle(reservoir)
    return reservoir


def read_frame(path: Path, frame_id: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode {path} frame={frame_id}")
    return frame


def labelled_panel(frame: np.ndarray, label: str, target_width: int = 800) -> np.ndarray:
    scale = target_width / max(frame.shape[1], 1)
    panel = cv2.resize(
        frame,
        (target_width, max(int(round(frame.shape[0] * scale)), 1)),
        interpolation=cv2.INTER_AREA,
    )
    cv2.rectangle(panel, (0, 0), (target_width, 38), (0, 0, 0), -1)
    cv2.putText(
        panel,
        label,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    args = parse_args()
    index_root = Path(args.index_root)
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = reservoir_samples(index_root, args.num_samples, args.seed)
    if not samples:
        raise RuntimeError(f"No recovered detections found under {index_root}")
    renderer = DetectionHintRenderer(
        {
            "index_root": str(index_root),
            "topk_ball": args.topk_ball,
            "topk_goal": 1,
            "raw_rgb_prob": 0.0,
            "ball_dropout_prob": 0.0,
            "goal_dropout_prob": 0.0,
        }
    )
    rows = []
    panels = []
    for index, (video_id, frame_id) in enumerate(samples):
        source = read_frame(video_path(videos_dir, video_id), frame_id)
        hinted = renderer.render(
            source, video_id, frame_id, is_train=False
        )
        pair = np.concatenate(
            [
                labelled_panel(source, "original"),
                labelled_panel(hinted, "detector hint: top-3 ball + goal"),
            ],
            axis=1,
        )
        name = f"{index:03d}_{video_id}_f{frame_id}.jpg"
        cv2.imwrite(str(output_dir / name), pair)
        panels.append(cv2.resize(pair, (1200, 338), interpolation=cv2.INTER_AREA))
        rows.append({"video_id": video_id, "frame_id": frame_id, "file": name})
    contact_sheet = np.concatenate(panels, axis=0)
    cv2.imwrite(str(output_dir / "contact_sheet.jpg"), contact_sheet)
    (output_dir / "samples.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )
    print(f"rendered={len(rows)} contact_sheet={output_dir / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
