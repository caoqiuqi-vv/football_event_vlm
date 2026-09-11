#!/usr/bin/env python3
from __future__ import annotations
import runpy,sys,types
from football_e2e_spotter import verifier_data
m=types.ModuleType('football_e2e_spotter.train_set_verifier')
for name in ('VerifierCandidateDataset','assign_oof_targets','read_rows'):setattr(m,name,getattr(verifier_data,name))
sys.modules[m.__name__]=m
runpy.run_path('football_e2e_spotter/train_set_verifier_ddp.py',run_name='__main__')
