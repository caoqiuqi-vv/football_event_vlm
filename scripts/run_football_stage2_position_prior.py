#!/usr/bin/env python
"""Compatibility CLI for the current Stage2 experiment; defaults to preflight."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from football_events.stage2.experiment import DEFAULT, prepare, integrity, preflight, pipeline, main
from football_events.stage2.cache import cache, compact, validate_compacted_cache
from football_events.stage2.training import train
from football_events.stage2.reporting import report

if __name__ == "__main__":
    main()
