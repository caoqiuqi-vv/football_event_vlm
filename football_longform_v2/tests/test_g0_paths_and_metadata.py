from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from football_longform_v2.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def load_builder_module():
    path = ROOT / "scripts/build_rgb_timeline.py"
    spec = importlib.util.spec_from_file_location("g0_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class G0PathAndMetadataTest(unittest.TestCase):
    def test_g0_paths_are_project_local_and_supervised_artifact_exists(self) -> None:
        for config_name in ("g0_official.yaml", "g0_supervised_ema.yaml"):
            config = load_config(ROOT / "configs" / config_name)
            project_root = Path(config["_project_root"])
            for key in ("output_dir", "feature_store"):
                resolved = (project_root / config["paths"][key]).resolve()
                self.assertTrue(str(resolved).startswith(str(project_root / "experiments")))
        supervised = load_config(ROOT / "configs/g0_supervised_ema.yaml")
        weights = (Path(supervised["_project_root"]) / supervised["features"]["context"]["weights"]).resolve()
        self.assertTrue(weights.is_file())
        self.assertTrue(str(weights).startswith(str(ROOT / "experiments/g0_backbone_ablation/artifacts")))

    def test_metadata_serializes_max_seconds_without_model_loading(self) -> None:
        builder = load_builder_module()
        metadata = builder.timeline_metadata(
            arch="dinov3_vitl16",
            backbone_id="unit_test",
            weights=Path("/tmp/weights.pth"),
            video=Path("/tmp/video.mp4"),
            max_seconds=120.0,
            config_sha256="config-hash",
            weights_sha256="weights-hash",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "timeline.npz"
            np.savez(output, timestamps=np.array([0.0], dtype=np.float32), **metadata)
            with np.load(output, allow_pickle=False) as payload:
                self.assertEqual(payload["backbone_id"].item(), "unit_test")
                self.assertEqual(float(payload["max_seconds"].item()), 120.0)
                self.assertEqual(payload["config_sha256"].item(), "config-hash")
                self.assertEqual(payload["weights_sha256"].item(), "weights-hash")


if __name__ == "__main__":
    unittest.main()
