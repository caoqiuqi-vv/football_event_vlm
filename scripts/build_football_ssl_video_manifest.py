#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def iter_video_paths(root: Path):
    return (
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def load_expected_names(list_paths: list[str]) -> set[str]:
    names: set[str] = set()
    for list_text in list_paths:
        list_path = Path(list_text)
        if not list_path.exists():
            raise FileNotFoundError(list_path)
        for line in list_path.read_text().splitlines():
            value = line.strip()
            if value:
                names.add(Path(value).name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated video manifest for football SSL.")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[],
        help="Stable video roots that are scanned directly.",
    )
    parser.add_argument(
        "--name-lists",
        nargs="*",
        default=[],
        help="Text files defining the expected video names for migrating datasets.",
    )
    parser.add_argument(
        "--lookup-roots",
        nargs="*",
        default=[],
        help="Roots searched by name in priority order; e.g. new encoded root then legacy root.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--stride-sec", type=float, default=2.0)
    parser.add_argument("--min-bytes", type=int, default=1_000_000)
    args = parser.parse_args()

    stable_paths: dict[str, Path] = {}
    for root_text in args.roots:
        root = Path(root_text)
        if not root.exists():
            raise FileNotFoundError(root)
        for path in iter_video_paths(root):
            stable_paths[str(path)] = path

    lookup_roots = [Path(value).resolve() for value in args.lookup_roots]
    for root in lookup_roots:
        if not root.exists():
            raise FileNotFoundError(root)

    expected_names = load_expected_names(args.name_lists)
    # Include files already present even if a migration task list omitted an early pilot item.
    for root in lookup_roots:
        expected_names.update(path.name for path in iter_video_paths(root))

    records: list[tuple[Path, list[str], str]] = [
        (path, [str(path)], path.name) for path in sorted(stable_paths.values())
    ]
    unresolved: list[dict[str, object]] = []
    resolved_by_root = {str(root): 0 for root in lookup_roots}
    for name in sorted(expected_names):
        candidates = [root / name for root in lookup_roots]
        chosen = next(
            (
                path
                for path in candidates
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
                and path.stat().st_size >= args.min_bytes
            ),
            None,
        )
        if chosen is None:
            unresolved.append({"name": name, "candidates": [str(path) for path in candidates]})
            continue
        resolved_by_root[str(chosen.parent)] += 1
        records.append((chosen, [str(path) for path in candidates], name))

    valid = []
    invalid = []
    total_duration = 0.0
    total_samples = 0
    for index, (path, fallback_paths, name) in enumerate(records, start=1):
        metadata = None
        probe_errors = []
        for candidate_text in fallback_paths:
            candidate = Path(candidate_text)
            try:
                if not candidate.is_file() or candidate.stat().st_size < args.min_bytes:
                    continue
                capture = cv2.VideoCapture(str(candidate))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                capture.release()
                duration = frames / fps if fps > 0 else 0.0
                if width > 0 and height > 0 and fps > 0 and frames > 0 and duration > 1.0:
                    path = candidate
                    metadata = (width, height, fps, frames, duration)
                    break
                probe_errors.append(
                    {
                        "path": str(candidate),
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "frames": frames,
                    }
                )
            except (FileNotFoundError, OSError) as exception:
                probe_errors.append({"path": str(candidate), "error": repr(exception)})
        if metadata is None:
            invalid.append(
                {
                    "name": name,
                    "candidates": fallback_paths,
                    "probe_errors": probe_errors,
                    "reason": "no_decodable_candidate",
                }
            )
            continue
        width, height, fps, frames, duration = metadata
        parent_text = str(path.parent)
        if parent_text in resolved_by_root:
            resolved_by_root[parent_text] += 1
        item = {
            "name": name,
            "path": str(path),
            "fallback_paths": fallback_paths,
            "duration_sec": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "frames": frames,
        }
        valid.append(item)
        total_duration += duration
        total_samples += max(int(max(duration - 2.0, 0.0) // args.stride_sec), 1)
        if index % 100 == 0 or index == len(records):
            print(
                f"manifest progress={index}/{len(records)} valid={len(valid)} "
                f"invalid={len(invalid)} unresolved={len(unresolved)}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for item in valid:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "roots": args.roots,
        "name_lists": args.name_lists,
        "lookup_roots": [str(root) for root in lookup_roots],
        "expected_lookup_names": len(expected_names),
        "resolved_by_root_at_build": resolved_by_root,
        "discovered_records": len(records),
        "valid": len(valid),
        "invalid": invalid,
        "unresolved": unresolved,
        "duration_hours": total_duration / 3600.0,
        "stride_sec": args.stride_sec,
        "virtual_samples": total_samples,
        "manifest": str(output.resolve()),
        "runtime_resolution": "fallback_paths are checked in priority order for every sample",
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
