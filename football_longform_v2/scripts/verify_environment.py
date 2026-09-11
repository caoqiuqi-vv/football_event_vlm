from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_longform_v2.config import load_config
from football_longform_v2.external import RfDetrBallTeacher, YoloStructureProvider
from football_longform_v2.schema import assert_disjoint_splits, read_video_ids


def resolve(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["_project_root"])
    paths = config["paths"]
    train_ids = read_video_ids(resolve(root, paths["train_ids"]))
    calibration_ids = read_video_ids(resolve(root, paths["calibration_ids"]))
    test_ids = read_video_ids(resolve(root, paths["thirdparty_test_ids"]))
    assert_disjoint_splits(train_ids, calibration_ids, test_ids)
    if len(test_ids) != 18:
        raise ValueError(f"thirdparty test must contain exactly 18 videos, got {len(test_ids)}")
    providers = config["external_providers"]
    yolo = YoloStructureProvider.from_config(providers["yolo_structure"], root)
    rfdetr = RfDetrBallTeacher.from_config(providers["rfdetr_ball_teacher"], root)
    yolo.validate()
    rfdetr.validate()
    print(json.dumps({
        "splits": {"train": len(train_ids), "calibration": len(calibration_ids), "test": len(test_ids)},
        "rgb_only": bool(config["acceptance"]["rgb_only_checkpoint_selection"]),
        "entity_enabled": bool(config["features"]["entity"]["enabled"]),
        "yolo_checkpoint": str(yolo.checkpoint),
        "rfdetr_checkpoint": str(rfdetr.checkpoint),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

