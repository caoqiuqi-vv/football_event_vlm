#!/usr/bin/env python
"""AdaSpot-style joint feature fusion over a frozen global and trainable ROI probe."""

from __future__ import annotations

import torch

import train_football_events as football
import train_football_events_featuremap_r1 as driver
from football_sparsemax_roi_residual import upgrade_roi_only_residual


_load_featuremap_r1 = driver.load_featuremap_r1
_load_config = football.load_config


def load_adaspot_feature_fusion(cfg, device):
    model = _load_featuremap_r1(cfg, device)
    model.spatial_attention = upgrade_roi_only_residual(
        model.spatial_attention, sparsemax_temperature=4.0
    ).to(device)

    roi_checkpoint_path = str(cfg.model.featuremap_r1.roi_init_checkpoint)
    checkpoint = torch.load(
        roi_checkpoint_path, map_location="cpu", weights_only=True
    )
    source = football.strip_module_prefix(checkpoint["model"])
    target = model.state_dict()
    # Reuse the fully learned ROI representation, but never import parameters
    # from a different fusion design into the new alignment modules.
    matched = {
        key: value
        for key, value in source.items()
        if key.startswith("spatial_attention.")
        and key in target
        and tuple(value.shape) == tuple(target[key].shape)
    }
    missing, unexpected = model.load_state_dict(matched, strict=False)
    spatial_matched = len(matched)
    if spatial_matched == 0:
        raise RuntimeError(
            f"No ROI spatial weights matched checkpoint={roi_checkpoint_path}"
        )
    model.spatial_attention_mode = "adaspot_feature_fusion"
    model.freeze_global_parameters()
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    expected_prefixes = (
        "spatial_attention.",
        "spatial_feature_global_align.",
        "spatial_feature_local_align.",
        "spatial_feature_output.",
    )
    unexpected_trainable = [
        name for name in trainable if not name.startswith(expected_prefixes)
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "AdaSpot joint training must only update ROI and feature-fusion modules; "
            f"unexpected={unexpected_trainable[:12]}"
        )
    print(
        f"Loaded AdaSpot ROI checkpoint={roi_checkpoint_path} "
        f"epoch={checkpoint.get('epoch')} matched={len(matched)} "
        f"spatial_matched={spatial_matched} missing={len(missing)} "
        f"unexpected={len(unexpected)} trainable={trainable}",
        flush=True,
    )
    return model.to(device)


def load_adaspot_config(path, overrides):
    cfg = _load_config(path, overrides)
    cfg.model.spatial_attention.mode = "adaspot_feature_fusion"
    cfg.model.freeze_spatial_probe = False
    cfg.model.architecture_version = "featuremap_adaspot_feature_fusion"
    cfg.output_dir = (
        "outputs/football_events/"
        "vitl16_featuremap_adaspot_fusion_from_roi_e4_16f_hr"
    )
    train = cfg.train
    train.negative_teacher_guard_loss_weight = 0.25
    train.negative_teacher_guard_labels = list(football.LABELS)
    train.negative_teacher_guard_margin = 0.0
    train.negative_teacher_guard_branch = "fused"
    # Fusion starts from an exact identity mapping and can learn quickly. The
    # epoch-4 ROI probe adapts conservatively so early fusion noise does not
    # overwrite its already useful localization.
    train.lr_per_gpu = 2.0e-5
    train.spatial_fusion_lr_per_gpu = 2.0e-5
    train.spatial_probe_lr_per_gpu = 5.0e-6
    train.clip_loss_weight = 1.0
    train.spatial_clip_loss_weight = 0.5
    train.spatial_attention_mil_loss_weight = 0.15
    train.spatial_attention_query_diversity_loss_weight = 0.01
    train.spatial_attention_concentration_loss_weight = 0.03
    train.spatial_attention_overlap_loss_weight = 0.03
    train.spatial_attention_entropy_target = 0.38
    train.positive_retention_loss_weight = 0.25
    train.positive_retention_branch = "fused"
    return cfg


if __name__ == "__main__":
    football.load_config = load_adaspot_config
    driver.load_featuremap_r1 = load_adaspot_feature_fusion
    driver.main()
