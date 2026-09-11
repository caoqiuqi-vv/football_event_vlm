from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import v2

from dinov3.data.transforms import GaussianBlur, make_normalize_transform


@dataclass(frozen=True)
class FootballSSLViews:
    """Decoded views for one football SSL sample.

    Geometry is selected in the dataset, before photometric augmentation.  This
    makes it possible to audit detector/motion fallback decisions independently
    from the stochastic DINO transforms.
    """

    anchor: Image.Image
    neighbor: Image.Image
    detector_local: Image.Image | None
    motion_local: Image.Image | None
    video_id: str
    time_sec: float
    detector_valid: bool
    detector_reason: str


def load_football_ssl_data_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Football SSL data config does not exist: {config_path}")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"Football SSL data config must be a mapping: {config_path}")
    if config.get("kind") != "football_detector_local_v2":
        raise ValueError(f"Unsupported football SSL data config kind: {config.get('kind')!r}")
    return config


class RandomJPEGCompression:
    def __init__(self, probability: float = 0.25, quality: tuple[int, int] = (55, 95)) -> None:
        self.probability = float(probability)
        self.quality = (int(quality[0]), int(quality[1]))

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*self.quality))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class FootballDataAugmentationDINO:
    """DINO multi-crop augmentation that preserves football-specific evidence.

    The two global views come from nearby frames and keep a wide aspect ratio.
    Local views mix detector, motion and unconstrained crops, so detector misses
    cannot remove a video from training or become a shortcut.
    """

    def __init__(self, config: dict[str, Any], *, mean, std) -> None:
        global_size = tuple(int(value) for value in config["global_size"])
        local_size = int(config["local_size"])
        self.local_crops_number = int(config.get("local_crops_number", 4))
        self.gram_teacher_views = bool(config.get("gram_teacher_views", False))
        if len(global_size) != 2 or any(value <= 0 for value in global_size):
            raise ValueError(f"global_size must be [height, width], got {global_size}")
        if self.local_crops_number < 2:
            raise ValueError("football SSL requires at least two local crops")

        global_scale = tuple(float(value) for value in config.get("global_scale", [0.82, 1.0]))
        global_ratio = tuple(float(value) for value in config.get("global_ratio", [1.65, 1.90]))
        local_scale = tuple(float(value) for value in config.get("local_scale", [0.60, 1.0]))
        horizontal_flips = bool(config.get("horizontal_flips", True))
        photometric = config.get("photometric", {})

        self.global_geometry = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_size,
                    scale=global_scale,
                    ratio=global_ratio,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )
        self.local_geometry = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_size,
                    scale=local_scale,
                    ratio=(0.85, 1.15),
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        color = v2.Compose(
            [
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=float(photometric.get("brightness", 0.25)),
                            contrast=float(photometric.get("contrast", 0.25)),
                            saturation=float(photometric.get("saturation", 0.15)),
                            hue=float(photometric.get("hue", 0.05)),
                        )
                    ],
                    p=float(photometric.get("color_jitter_probability", 0.8)),
                ),
                v2.RandomGrayscale(p=float(photometric.get("grayscale_probability", 0.05))),
                RandomJPEGCompression(
                    probability=float(photometric.get("jpeg_probability", 0.25)),
                    quality=tuple(photometric.get("jpeg_quality", [55, 95])),
                ),
            ]
        )
        # torchvision requires an explicit dtype on the server version.
        import torch

        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean, std=std),
            ]
        )
        self.global_1 = v2.Compose([color, GaussianBlur(p=0.7), self.normalize])
        self.global_2 = v2.Compose([color, GaussianBlur(p=0.15), self.normalize])
        self.local = v2.Compose([color, GaussianBlur(p=0.35), self.normalize])

    def __call__(self, views: FootballSSLViews) -> dict[str, Any]:
        if not isinstance(views, FootballSSLViews):
            raise TypeError(f"Expected FootballSSLViews, got {type(views).__name__}")

        global_base_1 = self.global_geometry(views.anchor)
        global_base_2 = self.global_geometry(views.neighbor)
        global_crop_1 = self.global_1(global_base_1)
        global_crop_2 = self.global_2(global_base_2)

        # Ordered sources make every batch contain both a focused view and an
        # unconstrained view.  Repetition only occurs when >4 locals are asked.
        focused = views.detector_local or views.motion_local or views.anchor
        secondary = views.motion_local if views.detector_local is not None and views.motion_local is not None else views.anchor
        local_sources = [focused, secondary, views.anchor, views.neighbor]
        locals_out = [
            self.local(self.local_geometry(local_sources[index % len(local_sources)]))
            for index in range(self.local_crops_number)
        ]
        output = {
            "weak_flag": True,
            "global_crops": [global_crop_1, global_crop_2],
            "global_crops_teacher": [global_crop_1, global_crop_2],
            "local_crops": locals_out,
            "offsets": (),
        }
        if self.gram_teacher_views:
            output["gram_teacher_crops"] = [
                self.normalize(global_base_1),
                self.normalize(global_base_2),
            ]
        return output

