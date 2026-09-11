#!/usr/bin/env python3
"""Run the full-image trainer with activation-checkpointed 25-frame encoding."""

from __future__ import annotations

import runpy
import sys

import football_e2e_spotter.verifier_fullimage as fullimage
from football_e2e_spotter.verifier_fullimage_checkpointed import CheckpointedFullImageDinoVerifier


fullimage.FullImageDinoVerifier = CheckpointedFullImageDinoVerifier
sys.argv[0] = "train_set_verifier_fullimage_checkpointed_ddp.py"
runpy.run_path("football_e2e_spotter/train_set_verifier_fullimage_ddp.py", run_name="__main__")
