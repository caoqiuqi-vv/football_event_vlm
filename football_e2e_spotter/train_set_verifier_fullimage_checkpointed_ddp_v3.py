#!/usr/bin/env python3
"""Run the uncertainty-aware verifier trainer with DDP unused-param support.

Candidate batches are intentionally sampled from deployment-like OOF slots, so
some ranks can receive only background candidates.  The time heads then have
no semantic temporal target on that iteration; DDP detection keeps the run
correct without fabricating a negative-candidate time target.
"""

from __future__ import annotations

import runpy
import sys

import torch.nn.parallel


_base_ddp = torch.nn.parallel.DistributedDataParallel


class CandidateSafeDDP(_base_ddp):
    def __init__(self, *args, **kwargs):
        kwargs["find_unused_parameters"] = True
        super().__init__(*args, **kwargs)


torch.nn.parallel.DistributedDataParallel = CandidateSafeDDP
sys.argv[0] = "train_set_verifier_fullimage_checkpointed_ddp_v3.py"
runpy.run_path("football_e2e_spotter/train_set_verifier_fullimage_checkpointed_ddp_v2.py", run_name="__main__")
