#!/usr/bin/env python3
from __future__ import annotations
import runpy
import football_e2e_spotter.verifier as verifier
from football_e2e_spotter.verifier_lora_runtime import configure
verifier.configure_dinov3_vitl16=configure
runpy.run_path('football_e2e_spotter/train_set_verifier.py',run_name='__main__')
