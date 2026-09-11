from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import torch

from football_longform_v2.feature_store import (
    align_feature_stream,
    assert_timeline_provenance,
    read_timeline_provenance,
)


class FeatureAlignmentTest(unittest.TestCase):
    def test_linear_timestamp_alignment(self) -> None:
        source_times = torch.tensor([0.0, 1.0, 2.0])
        features = torch.tensor([[0.0], [2.0], [4.0]])
        target_times = torch.tensor([0.0, 0.5, 1.5, 2.0])
        aligned, valid = align_feature_stream(source_times, features, target_times)
        self.assertTrue(torch.allclose(aligned[:, 0], torch.tensor([0.0, 1.0, 3.0, 4.0])))
        self.assertTrue(valid.all())


    def test_timeline_provenance_requires_exact_source_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "video.mp4"
            source.touch()
            cache = root / "timeline.npz"
            np.savez_compressed(
                cache,
                backbone_arch=np.asarray("dinov3_vitl16"),
                backbone_id=np.asarray("official"),
                backbone_weights=np.asarray("weights.pt"),
                source_video=np.asarray(str(source.resolve())),
                config_sha256=np.asarray("config-hash"),
                weights_sha256=np.asarray("weights-hash"),
            )
            self.assertEqual(read_timeline_provenance(cache)["backbone_id"], "official")
            assert_timeline_provenance(
                cache, expected_backbone_arch="dinov3_vitl16", expected_backbone_id="official",
                expected_source_video=source, expected_config_sha256="config-hash",
                expected_weights_sha256="weights-hash",
            )
            with self.assertRaises(ValueError):
                assert_timeline_provenance(
                    cache, expected_backbone_arch="dinov3_vitl16", expected_backbone_id="wrong",
                    expected_source_video=source, expected_config_sha256="config-hash",
                    expected_weights_sha256="weights-hash",
                )


if __name__ == "__main__":
    unittest.main()

