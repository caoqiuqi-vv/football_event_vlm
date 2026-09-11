from __future__ import annotations

import unittest

import torch

from football_longform_v2.backbones import BackboneError, merge_lora_state


class BackboneMergeTest(unittest.TestCase):
    def test_merges_base_and_lora_to_plain_weight(self) -> None:
        state = {
            "blocks.18.attn.qkv.base.weight": torch.ones(3, 2),
            "blocks.18.attn.qkv.base.bias": torch.ones(3),
            "blocks.18.attn.qkv.lora_a": torch.tensor([[1.0, 2.0]]),
            "blocks.18.attn.qkv.lora_b": torch.tensor([[1.0], [2.0], [3.0]]),
            "norm.weight": torch.ones(2),
        }
        merged = merge_lora_state(state, alpha=2.0, rank=1)
        self.assertEqual(set(merged), {"blocks.18.attn.qkv.weight", "blocks.18.attn.qkv.bias", "norm.weight"})
        self.assertTrue(torch.equal(merged["blocks.18.attn.qkv.weight"], torch.tensor([[3.0, 5.0], [5.0, 9.0], [7.0, 13.0]])))

    def test_rejects_missing_lora_pair(self) -> None:
        with self.assertRaises(BackboneError):
            merge_lora_state({"blocks.18.attn.qkv.base.weight": torch.ones(3, 2)}, alpha=2.0, rank=1)


if __name__ == "__main__":
    unittest.main()
