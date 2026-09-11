import unittest

import torch

from football_structured_roi_crop_clip import (
    DualQueryOriginalImageCropper,
    ROICropClipTemporalFusion,
    _gather_class_time,
)


class StructuredROICropClipTest(unittest.TestCase):
    def test_class_time_gather_keeps_matching_class(self):
        values = torch.arange(1 * 4 * 3).reshape(1, 4, 3, 1)
        indices = torch.tensor([[[0, 2], [1, 3], [3, 0]]])
        gathered = _gather_class_time(values, indices).squeeze(-1)
        expected = torch.tensor([[[0, 6], [4, 10], [11, 2]]])
        self.assertTrue(torch.equal(gathered, expected))

    def test_dual_query_crop_shapes(self):
        cropper = DualQueryOriginalImageCropper(
            num_labels=3,
            num_frames=4,
            topk_frames=2,
            exploration_frames=1,
            crop_height=32,
            crop_width=48,
            crop_scale_x=0.5,
            crop_scale_y=0.5,
        )
        inputs = torch.randn(2, 4, 3, 32, 48)
        attention = torch.rand(2, 4, 3, 2, 6)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        frame_logits = torch.randn(2, 4, 3)
        indices = cropper.select_indices(frame_logits)
        crops, params, selected = cropper(inputs, attention, indices)
        self.assertEqual(tuple(indices.shape), (2, 3, 3))
        self.assertEqual(tuple(crops.shape), (2, 18, 3, 32, 48))
        self.assertEqual(tuple(params.shape), (2, 3, 3, 2, 4))
        self.assertEqual(tuple(selected.shape), (2, 3, 3, 2, 6))

    def test_clip_fusion_has_no_frame_output(self):
        fusion = ROICropClipTemporalFusion(
            hidden_dim=16,
            num_labels=3,
            num_frames=4,
            slots=3,
            queries=2,
            num_heads=4,
            layers=1,
            dropout=0.0,
            gate_init=0.02,
            max_delta=1.0,
        )
        tokens = torch.randn(2, 18, 16)
        indices = torch.tensor(
            [
                [[0, 1, 3], [1, 2, 3], [0, 2, 3]],
                [[0, 2, 3], [0, 1, 2], [1, 2, 3]],
            ]
        )
        global_logits = torch.randn(2, 3)
        outputs = fusion(
            crop_tokens=tokens,
            indices=indices,
            global_logits=global_logits,
        )
        self.assertEqual(tuple(outputs["logits"].shape), (2, 3))
        self.assertNotIn("frame_event_logits", outputs)
        self.assertTrue(
            torch.allclose(outputs["logits"], global_logits, atol=0.0)
        )


if __name__ == "__main__":
    unittest.main()
