#!/usr/bin/env python
"""Compatibility entry point for the positive heatmap ROI audit."""

from __future__ import annotations

import scripts.visualize_positive_heatmap_rois as audit


original_load = audit.football.load_model_init_checkpoint
original_make_model = audit.football.make_model


def load_non_strict(*args, **kwargs):
    kwargs["strict"] = False
    return original_load(*args, **kwargs)


def make_with_maps(*args, **kwargs):
    model = original_make_model(*args, **kwargs)
    if model.spatial_attention is None:
        raise RuntimeError("Spatial attention model is required for heatmap audit")
    model.spatial_attention.return_attention_maps = True
    return model


audit.football.load_model_init_checkpoint = load_non_strict
audit.football.make_model = make_with_maps


if __name__ == "__main__":
    audit.main()
