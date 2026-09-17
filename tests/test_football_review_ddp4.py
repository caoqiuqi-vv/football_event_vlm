import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import train_football_events as t


class FourGpuReviewTopology(unittest.TestCase):
    def test_ddp4_preserves_e1_effective_batch_and_learning_rates(self):
        cfg=t.load_config(str(Path(__file__).resolve().parents[1]/'configs/football/review_clean_e2_ddp4_20260917.yaml'),[])
        with patch.object(t,'distributed_training_active',return_value=True):
            topology=t.resolve_runtime_topology(cfg,torch.device('cuda:0'))
        self.assertEqual(topology['world_size'],4)
        self.assertEqual(topology['effective_batch_size'],80)
        self.assertEqual(topology['train_per_gpu_batch_size'],4)
        self.assertEqual(topology['grad_accum_steps'],5)
        self.assertAlmostEqual(topology['lr_global'],5e-5)
        self.assertAlmostEqual(topology['global_backbone_lr_global'],1e-5)
        self.assertFalse(cfg.train.resume['enabled'])
        self.assertEqual(cfg.train.frame_det_loss_weight,.15)


if __name__=='__main__':unittest.main()
