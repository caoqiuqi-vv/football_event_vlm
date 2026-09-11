from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.annotations import load_event_annotations  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.schema import assert_disjoint_splits, read_video_ids  # noqa: E402

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV")


def resolve(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def find_video(video_root: Path, video_id: str) -> tuple[Path | None, str]:
    exact = [video_root / f"{video_id}{suffix}" for suffix in VIDEO_SUFFIXES]
    exact = [path for path in exact if path.is_file()]
    if len(exact) == 1:
        return exact[0], "exact"
    matches = sorted(
        path for path in video_root.glob(f"{video_id}*")
        if path.is_file() and path.suffix in VIDEO_SUFFIXES
    )
    if len(matches) == 1:
        return matches[0], "prefix"
    if not matches:
        return None, "missing"
    return None, "ambiguous"


def video_metadata(path: Path) -> dict[str, float | int | None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"fps": None, "frame_count": None, "duration_seconds": None, "width": None, "height": None}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "fps": fps if fps > 0 else None,
        "frame_count": frame_count if frame_count > 0 else None,
        "duration_seconds": frame_count / fps if fps > 0 and frame_count > 0 else None,
        "width": width if width > 0 else None,
        "height": height if height > 0 else None,
    }


def digest_ids(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze media/annotation inventory for LF-A0 official.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--report", default="experiments/lf_a0_official_fullscale/inventory.json")
    parser.add_argument(
        "--source-train-ids",
        default="../configs/football/splits/thirdparty18_holdout_all_rest_train/train_fit_video_ids.txt",
    )
    parser.add_argument(
        "--source-calibration-ids",
        default="../configs/football/splits/thirdparty18_holdout_all_rest_train/calibration_20_video_ids.txt",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])
    paths = config["paths"]
    annotations = resolve(project_root, paths["annotations"])
    video_root = Path(args.video_root).expanduser().resolve()
    splits = {
        "train": read_video_ids(resolve(project_root, args.source_train_ids)),
        "calibration": read_video_ids(resolve(project_root, args.source_calibration_ids)),
        "thirdparty_test": read_video_ids(resolve(project_root, paths["thirdparty_test_ids"])),
    }
    assert_disjoint_splits(*splits.values())

    all_rows: list[dict] = []
    summary: dict[str, dict] = {}
    seen_paths: Counter[str] = Counter()
    for split, ids in splits.items():
        if split == "thirdparty_test":
            summary[split] = {
                "declared_count": len(ids),
                "id_sha256": digest_ids(ids),
                "blind_holdout": True,
                "media_and_annotations_not_opened": True,
            }
            continue
        rows: list[dict] = []
        for video_id in ids:
            annotation = annotations / f"{video_id}.json"
            video, match_kind = find_video(video_root, video_id)
            event_counts: Counter[str] = Counter()
            rejected_event_counts: Counter[str] = Counter()
            annotation_error: str | None = None
            if annotation.is_file():
                try:
                    reviewed = load_event_annotations(annotation)
                    event_counts.update(event.label for event in reviewed.accepted)
                    rejected_event_counts.update(event.label for event in reviewed.rejected)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    annotation_error = str(error)
            row = {
                "video_id": video_id,
                "split": split,
                "annotation_path": str(annotation),
                "annotation_exists": annotation.is_file(),
                "annotation_error": annotation_error,
                "event_count": int(sum(event_counts.values())),
                "event_counts": dict(sorted(event_counts.items())),
                "rejected_event_count": int(sum(rejected_event_counts.values())),
                "rejected_event_counts": dict(sorted(rejected_event_counts.items())),
                "video_path": str(video) if video else None,
                "video_match": match_kind,
                "video_metadata": video_metadata(video) if video else None,
            }
            if video:
                seen_paths[str(video)] += 1
            rows.append(row)
            all_rows.append(row)
        media_ready = [row for row in rows if row["video_path"] and row["annotation_exists"]]
        accepted_counts: Counter[str] = Counter()
        rejected_counts: Counter[str] = Counter()
        videos_with_accepted: Counter[str] = Counter()
        for row in rows:
            accepted_counts.update(row["event_counts"])
            rejected_counts.update(row["rejected_event_counts"])
            videos_with_accepted.update(
                label for label, count in row["event_counts"].items() if count > 0
            )
        summary[split] = {
            "declared_count": len(rows),
            "id_sha256": digest_ids(ids),
            "media_and_annotation_ready_count": len(media_ready),
            "missing_video_ids": [row["video_id"] for row in rows if row["video_match"] == "missing"],
            "ambiguous_video_ids": [row["video_id"] for row in rows if row["video_match"] == "ambiguous"],
            "prefix_resolved_ids": [row["video_id"] for row in rows if row["video_match"] == "prefix"],
            "missing_annotation_ids": [row["video_id"] for row in rows if not row["annotation_exists"]],
            "invalid_or_zero_duration_ids": [
                row["video_id"] for row in rows
                if row["video_metadata"] is not None
                and not (row["video_metadata"]["duration_seconds"] or 0.0) > 0.0
            ],
            "accepted_event_counts": dict(sorted(accepted_counts.items())),
            "rejected_event_counts": dict(sorted(rejected_counts.items())),
            "videos_with_accepted_events": dict(sorted(videos_with_accepted.items())),
        }
    duplicate_media = sorted(path for path, count in seen_paths.items() if count > 1)
    report = {
        "experiment_id": config["experiment_id"],
        "inventory_scope": "train+calibration media and annotations only; thirdparty18 is ID-digest-only and never opened",
        "video_root": str(video_root),
        "annotation_root": str(annotations),
        "split_summary": summary,
        "declared_train_plus_calibration": len(splits["train"]) + len(splits["calibration"]),
        "ready_train_plus_calibration": summary["train"]["media_and_annotation_ready_count"] + summary["calibration"]["media_and_annotation_ready_count"],
        "declared_all_splits": sum(len(ids) for ids in splits.values()),
        "duplicate_resolved_media_paths": duplicate_media,
        "rows": all_rows,
    }
    report_path = resolve(project_root, args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "declared_train_plus_calibration", "ready_train_plus_calibration", "duplicate_resolved_media_paths", "split_summary"
    )}, ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
