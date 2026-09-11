from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.annotations import load_events  # noqa: E402
from football_longform_v2.canonical import load_canonical_split  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.schema import read_video_ids  # noqa: E402


def resolve(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_balanced_folds(
    rows: list[tuple[str, int]], *, fold_count: int, seed: int
) -> list[list[str]]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if len(rows) < fold_count:
        raise ValueError("fewer videos than folds")
    if len({video_id for video_id, _ in rows}) != len(rows):
        raise ValueError("duplicate video IDs")
    rng = random.Random(seed)
    positives = [row for row in rows if row[1] > 0]
    negatives = [row for row in rows if row[1] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positives.sort(key=lambda row: -row[1])
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    event_totals = [0] * fold_count
    positive_totals = [0] * fold_count
    for video_id, event_count in positives:
        index = min(
            range(fold_count),
            key=lambda one: (event_totals[one], positive_totals[one], len(folds[one]), one),
        )
        folds[index].append(video_id)
        event_totals[index] += event_count
        positive_totals[index] += 1
    for video_id, _ in negatives:
        index = min(range(fold_count), key=lambda one: (len(folds[one]), one))
        folds[index].append(video_id)
    return [sorted(fold) for fold in folds]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic video-grouped OOF folds for penalty calibration."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["_project_root"])
    paths = config["paths"]
    canonical_path = resolve(root, paths["canonical_manifest"])
    canonical = load_canonical_split(canonical_path, "train")
    calibration = set(load_canonical_split(canonical_path, "calibration").media_ids)
    fixed_test = set(read_video_ids(resolve(root, paths["thirdparty_test_ids"])))
    annotations = resolve(root, paths["annotations"])
    rows: list[tuple[str, int]] = []
    for video_id in canonical.media_ids:
        annotation_id = canonical.annotation_id_by_media_id[video_id]
        events = load_events(annotations / f"{annotation_id}.json")
        rows.append((video_id, sum(event.label == "penalty" for event in events)))
    folds = build_balanced_folds(rows, fold_count=args.folds, seed=args.seed)
    train_set = set(canonical.media_ids)
    flattened = [video_id for fold in folds for video_id in fold]
    if len(flattened) != len(train_set) or set(flattened) != train_set:
        raise RuntimeError("OOF folds do not form an exact partition of canonical train videos")
    if train_set & calibration or train_set & fixed_test or calibration & fixed_test:
        raise RuntimeError("train/calibration/fixed-test split overlap detected")
    count_by_id = dict(rows)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else resolve(root, paths["output_dir"]) / "penalty_oof_folds.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fold_records = []
    for index, validation_ids in enumerate(folds):
        validation = set(validation_ids)
        train_ids = sorted(train_set - validation)
        positive_ids = [video_id for video_id in validation_ids if count_by_id[video_id] > 0]
        train_path = output.with_name(f"penalty_oof_fold_{index}_train_ids.txt")
        validation_path = output.with_name(f"penalty_oof_fold_{index}_validation_ids.txt")
        train_path.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
        validation_path.write_text("\n".join(validation_ids) + "\n", encoding="utf-8")
        fold_records.append({
            "fold": index,
            "train_id_file": str(train_path),
            "train_id_file_sha256": sha256_file(train_path),
            "validation_id_file": str(validation_path),
            "validation_id_file_sha256": sha256_file(validation_path),
            "validation_media_ids": validation_ids,
            "train_media_ids": train_ids,
            "validation_video_count": len(validation_ids),
            "validation_penalty_positive_video_count": len(positive_ids),
            "validation_penalty_event_count": sum(count_by_id[video_id] for video_id in validation_ids),
            "validation_penalty_positive_media_ids": positive_ids,
        })
    payload = {
        "schema": "football_longform_v2.penalty_oof_folds.v1",
        "purpose": "development_only_penalty_threshold_calibration",
        "grouping_unit": "canonical_long_video",
        "seed": args.seed,
        "fold_count": args.folds,
        "canonical_manifest": str(canonical_path),
        "canonical_manifest_sha256": sha256_file(canonical_path),
        "source_train_video_count": len(train_set),
        "source_penalty_positive_video_count": sum(count > 0 for _, count in rows),
        "source_penalty_event_count": sum(count for _, count in rows),
        "calibration_overlap_count": 0,
        "fixed_test_overlap_count": 0,
        "fixed_test_labels_or_predictions_used": False,
        "folds": fold_records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "fold_validation_video_counts": [fold["validation_video_count"] for fold in fold_records],
        "fold_penalty_positive_video_counts": [fold["validation_penalty_positive_video_count"] for fold in fold_records],
        "fold_penalty_event_counts": [fold["validation_penalty_event_count"] for fold in fold_records],
        "fixed_test_overlap_count": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
