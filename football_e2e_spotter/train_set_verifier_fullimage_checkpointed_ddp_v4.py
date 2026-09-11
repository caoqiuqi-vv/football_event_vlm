#!/usr/bin/env python3
"""DDP-ready full-image verifier trainer with no ROI and safe background batches."""

from __future__ import annotations

import runpy
import sys

import torch
import torch.nn.parallel

import football_e2e_spotter.verifier_fullimage_checkpointed as checkpointed


_base_verifier = checkpointed.CheckpointedFullImageDinoVerifier


class DDPReadyVerifier(_base_verifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Stage-1 exports exactly 64 shared tokens at hidden dim 512.  Materialize
        # the sole LazyLinear before DDP without an expensive DINO forward pass.
        self.shared_projection(torch.zeros(1, 64, 512))


_base_ddp = torch.nn.parallel.DistributedDataParallel


class CandidateSafeDDP(_base_ddp):
    def __init__(self, *args, **kwargs):
        kwargs["find_unused_parameters"] = True
        super().__init__(*args, **kwargs)


checkpointed.CheckpointedFullImageDinoVerifier = DDPReadyVerifier
torch.nn.parallel.DistributedDataParallel = CandidateSafeDDP
sys.argv[0] = "train_set_verifier_fullimage_checkpointed_ddp_v4.py"
runpy.run_path("football_e2e_spotter/train_set_verifier_fullimage_checkpointed_ddp_v2.py", run_name="__main__")
