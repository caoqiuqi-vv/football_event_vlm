from __future__ import annotations

import torch
from PIL import Image, ImageDraw

from dinov3.data.datasets.football_ssl_webdataset import FootballSSLWebDataset
from dinov3.data.datasets.football_video_ssl import motion_local_crop
from dinov3.data.football_ssl import FootballDataAugmentationDINO, FootballSSLViews


def config() -> dict:
    return {
        "global_size": [224, 400],
        "global_scale": [0.85, 1.0],
        "global_ratio": [1.65, 1.90],
        "local_size": 128,
        "local_crops_number": 4,
        "local_scale": [0.6, 1.0],
        "horizontal_flips": True,
        "photometric": {"jpeg_probability": 0.0},
    }


def test_motion_crop_focuses_changed_region() -> None:
    first = Image.new("RGB", (640, 360), "green")
    second = first.copy()
    ImageDraw.Draw(second).rectangle((280, 130, 390, 260), fill="white")
    crop = motion_local_crop(first, second)
    assert crop is not None
    assert crop.width == crop.height
    assert 100 <= crop.width <= 220


def test_football_augmentation_shapes_with_detector_fallback() -> None:
    anchor = Image.new("RGB", (640, 360), "green")
    neighbor = Image.new("RGB", (640, 360), "darkgreen")
    views = FootballSSLViews(
        anchor=anchor,
        neighbor=neighbor,
        detector_local=None,
        motion_local=anchor.crop((200, 60, 500, 360)),
        video_id="sample",
        time_sec=1.0,
        detector_valid=False,
        detector_reason="missing_index",
    )
    output = FootballDataAugmentationDINO(
        config(),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )(views)
    assert [tuple(tensor.shape) for tensor in output["global_crops"]] == [(3, 224, 400)] * 2
    assert [tuple(tensor.shape) for tensor in output["local_crops"]] == [(3, 128, 128)] * 4
    assert all(tensor.dtype == torch.float32 for tensor in output["local_crops"])


def test_webdataset_pairs_only_adjacent_frames_before_shuffle() -> None:
    rows = [
        (Image.new("RGB", (8, 8), color), Image.new("RGB", (4, 4)), metadata)
        for color, metadata in [
            ("red", {"video_id": "a", "time_sec": 0.0}),
            ("blue", {"video_id": "a", "time_sec": 2.0}),
            ("green", {"video_id": "b", "time_sec": 0.0}),
        ]
    ]
    dataset = object.__new__(FootballSSLWebDataset)
    dataset._pipeline = lambda: iter(rows)
    pairs = list(dataset._temporal_pairs())
    assert len(pairs) == 2
    assert pairs[0][1].getpixel((0, 0)) == (0, 0, 255)
    # A video boundary must never create a false temporal pair.
    assert pairs[1][1].getpixel((0, 0)) == (0, 0, 255)
