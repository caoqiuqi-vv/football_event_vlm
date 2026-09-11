from __future__ import annotations

import unittest
from pathlib import Path

from football_longform_v2.config import load_config
from football_longform_v2.external import RfDetrBallTeacher, YoloStructureProvider


ROOT = Path(__file__).resolve().parents[1]


class ExternalProviderTest(unittest.TestCase):
    def test_external_assets_and_commands(self) -> None:
        config = load_config(ROOT / "configs/lf_a0_rgb_only.yaml")
        raw = config["external_providers"]
        yolo = YoloStructureProvider.from_config(raw["yolo_structure"], ROOT)
        rfdetr = RfDetrBallTeacher.from_config(raw["rfdetr_ball_teacher"], ROOT)
        yolo.validate()
        rfdetr.validate()
        yolo_cmd = yolo.command(Path("/tmp/input.mp4"), Path("/tmp/yolo"))
        rfdetr_cmd = rfdetr.command(Path("/tmp/input.mp4"), Path("/tmp/rfdetr"))
        self.assertIn("--ball-cls", yolo_cmd.argv)
        self.assertIn("-1", yolo_cmd.argv)
        self.assertIn("--checkpoint", rfdetr_cmd.argv)
        self.assertFalse(config["features"]["entity"]["enabled"])


if __name__ == "__main__":
    unittest.main()
