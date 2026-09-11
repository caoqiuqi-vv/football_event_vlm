#!/usr/bin/env python
"""Fail fast if final FeatureMap-R1 config mutations are not visible."""

from __future__ import annotations

import argparse

from train_football_events_featuremap_r1_v6 import load_v6_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_v6_config(args.config, [])
    values = {
        "lr_attr": cfg.train.lr_per_gpu,
        "lr_get": cfg.train.get("lr_per_gpu"),
        "mil_attr": cfg.train.spatial_attention_mil_loss_weight,
        "mil_get": cfg.train.get("spatial_attention_mil_loss_weight"),
        "output_dir": cfg.output_dir,
        "architecture": cfg.model.architecture_version,
    }
    print("FEATUREMAP_R1_V6_CONFIG_OK", values, flush=True)


if __name__ == "__main__":
    main()
