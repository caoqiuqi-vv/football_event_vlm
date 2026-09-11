"""Compatibility imports; maintained implementation lives in football_events.stage2."""
from football_events.stage2.sampling import height_prior, select_peaks, tokens_at_peaks
from football_events.stage2.model import PositionPriorExtractor, PositionPriorEventModel

__all__ = [
    "height_prior", "select_peaks", "tokens_at_peaks",
    "PositionPriorExtractor", "PositionPriorEventModel",
]
