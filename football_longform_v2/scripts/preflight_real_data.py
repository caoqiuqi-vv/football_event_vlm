from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from football_longform_v2.config import load_config
from football_longform_v2.schema import assert_disjoint_splits, read_video_ids


LABEL_MAP = {
    "射门": "shot",
    "其他射门类型": "shot",
    "扑救": "save",
    "角球": "corner",
    "点球": "penalty",
    "任意球": "freekick",
    "中圈开球": "kickoff",
}
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV")
def annotation_items(payload: object) -> list[dict]:
    """Normalize both reviewed API exports and legacy flat event lists."""
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("events", payload.get("annotations", [])))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]



def resolve(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def find_video(video_root: Path, video_id: str) -> Path | None:
    for suffix in VIDEO_SUFFIXES:
        candidate = video_root / f"{video_id}{suffix}"
        if candidate.is_file():
            return candidate
    # A legacy cohort uses shortened IDs while media filenames retain the full ID.
    prefix_matches = sorted(
        path
        for path in video_root.glob(f"{video_id}*")
        if path.is_file() and path.suffix in VIDEO_SUFFIXES
    )
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


def audit_split(
    ids: tuple[str, ...], annotations: Path, video_root: Path, feature_store: Path
) -> dict:
    counts: Counter[str] = Counter()
    missing_annotations: list[str] = []
    missing_videos: list[str] = []
    ready_timelines: list[str] = []
    for video_id in ids:
        annotation_path = annotations / f"{video_id}.json"
        if not annotation_path.is_file():
            missing_annotations.append(video_id)
        else:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            for event in annotation_items(payload):
                label = LABEL_MAP.get(str(event.get("label", "")))
                # Repair exports only attach label_correct to manually changed
                # rows. Missing means accepted; only explicit false is rejected.
                if label and event.get("label_correct") is not False:
                    counts[label] += 1
        if find_video(video_root, video_id) is None:
            missing_videos.append(video_id)
        if (feature_store / video_id / "timeline.npz").is_file():
            ready_timelines.append(video_id)
    return {
        "video_count": len(ids),
        "accepted_events": dict(sorted(counts.items())),
        "missing_annotation_count": len(missing_annotations),
        "missing_annotation_ids": missing_annotations,
        "missing_video_count": len(missing_videos),
        "missing_video_ids": missing_videos,
        "ready_timeline_count": len(ready_timelines),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--report", default="experiments/lf_a0_preflight/report.json")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["_project_root"])
    paths = config["paths"]
    annotations = resolve(root, paths["annotations"])
    feature_store = resolve(root, paths["feature_store"])
    video_root = Path(args.video_root).expanduser().resolve()
    split_ids = {
        "train": read_video_ids(resolve(root, paths["train_ids"])),
        "calibration": read_video_ids(resolve(root, paths["calibration_ids"])),
        "thirdparty_test": read_video_ids(resolve(root, paths["thirdparty_test_ids"])),
    }
    assert_disjoint_splits(*split_ids.values())
    report = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "video_root": str(video_root),
        "annotations": str(annotations),
        "feature_store": str(feature_store),
        "rgb_only": True,
        "entity_enabled": False,
        "splits": {
            name: audit_split(ids, annotations, video_root, feature_store)
            for name, ids in split_ids.items()
        },
    }
    report_path = resolve(root, args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()

