from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.annotations import load_event_annotations  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.schema import assert_disjoint_splits, read_video_ids  # noqa: E402


def resolve(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def supervision_summary(entries: list[dict]) -> dict:
    accepted: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    videos: Counter[str] = Counter()
    for entry in entries:
        reviewed = load_event_annotations(entry["annotation_path"])
        per_video = Counter(event.label for event in reviewed.accepted)
        accepted.update(per_video)
        rejected.update(event.label for event in reviewed.rejected)
        videos.update(label for label, count in per_video.items() if count > 0)
    return {
        "accepted_event_counts": dict(sorted(accepted.items())),
        "rejected_event_counts": dict(sorted(rejected.items())),
        "videos_with_accepted_events": dict(sorted(videos.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the reviewed-media canonical LF-A0 manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory", default="experiments/lf_a0_official_fullscale/inventory.json")
    parser.add_argument("--output", default="experiments/lf_a0_official_fullscale/canonical_manifest.json")
    parser.add_argument("--supervision-report")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["_project_root"])
    inventory_path = resolve(root, args.inventory)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = inventory["rows"]
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    train_ready = [row for row in by_split["train"] if row["video_path"] and row["annotation_exists"]]
    by_media = defaultdict(list)
    for row in train_ready:
        by_media[row["video_path"]].append(row)
    train: list[dict] = []
    aliases: list[dict] = []
    for media_path, views in sorted(by_media.items()):
        if len(views) == 1:
            view = views[0]
            train.append({
                "media_id": view["video_id"], "annotation_id": view["video_id"],
                "source_video": media_path, "annotation_path": view["annotation_path"],
            })
            continue
        if len(views) != 2:
            raise RuntimeError(f"unsupported {len(views)} annotation views for {media_path}")
        short = next((view for view in views if len(view["video_id"]) < 18), None)
        full = next((view for view in views if len(view["video_id"]) >= 18), None)
        if short is None or full is None:
            raise RuntimeError(f"cannot select short reviewed view and full media id for {media_path}")
        train.append({
            "media_id": full["video_id"], "annotation_id": short["video_id"],
            "source_video": media_path, "annotation_path": short["annotation_path"],
        })
        aliases.append({
            "media_id": full["video_id"], "reviewed_annotation_id": short["video_id"],
            "excluded_annotation_id": full["video_id"], "source_video": media_path,
        })
    calibration = [
        {
            "media_id": row["video_id"], "annotation_id": row["video_id"],
            "source_video": row["video_path"], "annotation_path": row["annotation_path"],
        }
        for row in by_split["calibration"] if row["video_path"] and row["annotation_exists"]
    ]
    if len(aliases) != 8 or len(train) != 135 or len(calibration) != 18:
        raise RuntimeError(
            f"unexpected canonical counts aliases={len(aliases)} train={len(train)} calibration={len(calibration)}"
        )
    paths = config["paths"]
    protected = read_video_ids(resolve(root, paths["thirdparty_test_ids"]))
    legacy_path = resolve(root, paths["legacy_long6_ids"])
    legacy = read_video_ids(legacy_path)
    selected = [item["media_id"] for item in train] + [item["media_id"] for item in calibration]
    if set(selected) & set(protected):
        raise RuntimeError("canonical selection intersects thirdparty18")
    if set(selected) & set(legacy):
        raise RuntimeError("canonical selection intersects legacy long6 holdout")
    assert_disjoint_splits(
        [item["media_id"] for item in train],
        [item["media_id"] for item in calibration],
        protected,
    )
    output = resolve(root, args.output)
    train_ids_path = output.with_name("canonical_train_media_ids.txt")
    calibration_ids_path = output.with_name("canonical_calibration_media_ids.txt")
    write_ids(train_ids_path, [item["media_id"] for item in train])
    write_ids(calibration_ids_path, [item["media_id"] for item in calibration])
    manifest = {
        "schema": "football_longform_v2.canonical_media.v1",
        "purpose": "Official-A fullscale frozen feature extraction and locator gate.",
        "policy": {
            "one_media_one_timeline": True,
            "reviewed_short_id_overrides_full_id_annotation": True,
            "excluded_full_annotation_views_are_not_sampled": True,
            "forbidden_splits": ["thirdparty18", "legacy_long6_holdout"],
        },
        "source_inventory": str(inventory_path),
        "train": train,
        "calibration": calibration,
        "aliases": aliases,
        "excluded_missing_media": {
            "train": [row["video_id"] for row in by_split["train"] if not row["video_path"]],
            "calibration": [row["video_id"] for row in by_split["calibration"] if not row["video_path"]],
        },
        "counts": {"train_unique_media": len(train), "calibration_unique_media": len(calibration)},
        "supervision_summary": {
            "train": supervision_summary(train),
            "calibration": supervision_summary(calibration),
        },
        "canonical_train_ids": str(train_ids_path),
        "canonical_calibration_ids": str(calibration_ids_path),
        "protected_thirdparty18_count": len(protected),
        "protected_legacy_long6_count": len(legacy),
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.supervision_report:
        supervision_report = resolve(root, args.supervision_report)
        supervision_report.parent.mkdir(parents=True, exist_ok=True)
        supervision_report.write_text(
            json.dumps(manifest["supervision_summary"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(f"manifest={output}")


if __name__ == "__main__":
    main()
