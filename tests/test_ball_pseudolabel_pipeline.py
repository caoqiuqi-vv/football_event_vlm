from argparse import Namespace
from pathlib import Path

from scripts.precompute_ball_pseudolabels import (
    balanced_shards,
    box_iou,
    claim_next_video,
    merge_candidates,
    read_video_ids,
    window_starts,
)


def test_window_starts_cover_both_edges() -> None:
    assert window_starts(1280, 640, 0.2) == [0, 512, 640]
    assert window_starts(720, 640, 0.2) == [0, 80]


def test_candidate_merge_preserves_strategy_provenance() -> None:
    candidates = [
        {"bbox_xyxy": [10, 10, 20, 20], "confidence": 0.8, "source": "full"},
        {"bbox_xyxy": [10.5, 10.5, 20.5, 20.5], "confidence": 0.7, "source": "hflip"},
        {"bbox_xyxy": [100, 100, 110, 110], "confidence": 0.4, "source": "tile"},
    ]
    merged = merge_candidates(candidates, 0.45, 20)
    assert len(merged) == 2
    assert merged[0]["sources"] == ["full", "hflip"]
    assert box_iou(merged[0]["bbox_xyxy"], candidates[0]["bbox_xyxy"]) == 1.0


def test_balanced_shards_are_deterministic(tmp_path: Path) -> None:
    paths = []
    for index, size in enumerate((10, 20, 30, 40)):
        path = tmp_path / f"{index}.mp4"
        path.write_bytes(b"x" * size)
        paths.append(path)
    first = balanced_shards(paths, 2)
    second = balanced_shards(list(reversed(paths)), 2)
    assert [[p.name for p in group] for group in first] == [
        [p.name for p in group] for group in second
    ]
    assert sorted(p.name for group in first for p in group) == sorted(p.name for p in paths)


def test_read_video_ids_accepts_ids_filenames_and_comments(tmp_path: Path) -> None:
    split = tmp_path / "test_ids.txt"
    split.write_text("# held out\n123\n/path/to/456.mp4\n\n", encoding="utf-8")
    assert read_video_ids([str(split)]) == {"123", "456"}


def test_dynamic_claims_are_unique(tmp_path: Path) -> None:
    videos = []
    for name, size in (("large.mp4", 20), ("small.mp4", 10)):
        path = tmp_path / name
        path.write_bytes(b"x" * size)
        videos.append(path)
    output = tmp_path / "output"
    args = Namespace(device="0", shard_index=0)
    first = claim_next_video(videos, output, args)
    second = claim_next_video(videos, output, args)
    assert first is not None and second is not None
    assert first[0].name == "large.mp4"
    assert second[0].name == "small.mp4"
