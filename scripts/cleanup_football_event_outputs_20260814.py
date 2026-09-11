#!/usr/bin/env python3
"""Remove superseded football-event runs and duplicate checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path("outputs/football_events")

# These runs are failed, debug-only, empty, or conclusively worse than retained
# baselines. Their metrics and reasons are recorded in CLEANUP_REVIEW_20260814.md.
DELETE_DIRS = {
    "dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_lora_d3_debug",
    "temporal_head_comparison",
    "vitb16_lvd1689m_long_freeze_temporal",
    "vith16plus_single_hr_e1_frame_det",
    "vitl16_clean_negative_margin_raw_16f_hr",
    "vitl16_clean_negative_margin_raw_16f_hr_init_eval",
    "vitl16_e1_clip_rank_clean_lora_last8",
    "vitl16_e1_clip_rank_clean_lora_last8_bsz8",
    "vitl16_e1_dual_class_query_fusion_d7c",
    "vitl16_e1_dynamic_roi_quality_clean_d6b",
    "vitl16_e1_precision_focal_neg8_from_e1_5090",
    "vitl16_e1_precision_focal_neg8_from_e1_5090_fast_b8_g3467",
    "vitl16_e1_precision_focal_neg8_from_e1_5090_g4567",
    "vitl16_e1_precision_focal_neg8_from_e1_5090_g4567_run2",
    "vitl16_e1_precision_focal_neg8_probe",
    "vitl16_er1_multilayer_readout_16f_hr",
    "vitl16_er1_multilayer_readout_16f_hr_from_scratch_20ep",
    "vitl16_featuremap_sparse_roi_q2k64_fp32head_16f_hr",
    "vitl16_featuremap_sparsemax_roi_nobypass_lr1e4_16f_hr",
    "vitl16_featuremap_sparsemax_roi_t4_q2_16f_hr",
    "vitl16_featuremap_structured_roi_temporal_16f_hr_failed_nan_20260813_2016",
    "vitl16_global_conditioned_spatial_residual_raw_16f_hr",
    "vitl16_r1_detector_heatmap_dual_roi_16f_hr",
    "vitl16_robust_dual_16f_e1_dynamic_roi_d1",
    "vitl16_robust_dual_16f_e1_dynamic_roi_decoupled_lora_d4",
    "vitl16_robust_dual_16f_e1_dynamic_roi_lora_d3",
    "vitl16_robust_dual_16f_e1_frame_det",
    "vitl16_robust_dual_16f_e1_frame_det_hard_neg_exp1",
    "vitl16_robust_dual_16f_exp4_lora_debug",
    "vitl16_robust_dual_16f_tcn",
    "vitl16_robust_dual_16f_tcn_hardneg",
    "vitl16_spatial_attn_multilayer_readout_r16_16f_hr",
    "vitl16_spatial_attn_residual_multilayer_16f_hr_warmup",
    "vitl16_spatial_probe_p1_16f_hr",
    "vitl16_stage2_single_hr_hardneg_v2_temporal",
    "vitl16_temporal_spatial_attention_16f_hr",
    "vitl16_temporal_spatial_attention_16f_hr_warmup",
    "vitl16_temporal_spatial_attention_v2_region_only_16f_hr_warmup",
    "vitl16_top_crop_fixed10_raw_16f_hr",
    "vitl16_uniform_event_duration_ohem_refine",
    "vitl16_uniform_event_duration_online_rank_refine",
    "vitl16_uniform_event_epoch1_hardneg_refine",
    "vitl16_uniform_event_online_ohem_refine",
    "vitl16_uniform_event_teacher_guard_hr16f_refine",
    "vitl16_weekend_attn_pool_frame_det",
    "vitl16_weekend_class_query_frame_det",
    "vitl16_weekend_event_anchor_512x896_neighborhood_st",
    "vitl16_weekend_event_topk_512x896_class_union_st",
    "vitl16_weekend_event_topk_640x1120_b2_class_union_st",
    "vitl16_weekend_event_topk_class_union_st",
    "vitl16_weekend_hardneg_last4",
    "vitl16_weekend_uniform_event_dual_512x896_st",
    "vitl16_weekend_uniform_event_dual_e1backbone_512x896_st",
}

# Preserve configs, logs, and metrics for useful comparisons, but remove all
# top-level checkpoint files other than those named here.
PRUNE_KEEP = {
    "vitl16_c4a_roi_evidence_tokens_from_A_best": {"best.pt"},
    "vitl16_dual_roi_gain_safe_neg_16f_hr": {"best.pt"},
    "vitl16_dual_roi_shared_16f_hr": {"best.pt"},
    "vitl16_dual_roi_temporal_memory_safe_neg_16f_hr": {"best.pt"},
    "vitl16_e1_dual_class_query_fusion_d7c_frame0_20260805": {"best.pt"},
    "vitl16_e1_dual_cross_attention_d7b": {"best.pt"},
    "vitl16_e1_dynamic_roi_quality_joint_d6a": {"best.pt"},
    "vitl16_featuremap_adaspot_fusion_from_roi_e4_16f_hr": {"best.pt"},
    "vitl16_featuremap_time_causal_from_fusion_e2_16f_hr": {"best.pt"},
    "vitl16_robust_dual_16f_e1_frame_det_exp2_from_vitl16_robust_dual_16f_exp4_hr": {
        "best.pt"
    },
    "vitl16_spatial_attn_multilayer_readout_16f_hr": {"best.pt"},
    "vitl16_spatial_attn_residual_multilayer_16f_hr": {"best.pt", "epoch_3.pt"},
    "vitl16_strong_lora12_mlp_16f_hr_e1": {"best.pt"},
    "vitl16_temporal_spatial_attention_v2_region_only_16f_hr": {"best.pt"},
    "vitl16_top_crop_aspect10_raw_16f_hr": {"best.pt"},
    "vitl16_weekend_event_anchor_512x896_sym20_st": {"best.pt"},
    "vitl16_weekend_uniform_event_dual_e1backbone_512x896_frozen_ref": {
        "epoch_1.pt"
    },
}

PROTECTED_DIRS = {
    "vitl16_featuremap_dual_roi_pr_correction_from_structured_e2_16f_hr",
    "vitl16_featuremap_dual_roi_corrected_frame_aux_from_pr_e2_16f_hr",
    "vitl16_featuremap_structured_roi_temporal_16f_hr",
    "vitl16_robust_dual_16f_exp4",
}


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = ROOT.resolve()
    if root.name != "football_events" or root.parent.name != "outputs":
        raise RuntimeError(f"Refusing unexpected cleanup root: {root}")
    if DELETE_DIRS & PROTECTED_DIRS or set(PRUNE_KEEP) & PROTECTED_DIRS:
        raise RuntimeError("Protected directory appears in cleanup sets")

    delete_paths = [root / name for name in sorted(DELETE_DIRS) if (root / name).is_dir()]
    prune_paths = []
    for name, keep_names in sorted(PRUNE_KEEP.items()):
        directory = root / name
        if not directory.is_dir():
            continue
        prune_paths.extend(
            path
            for path in sorted(directory.glob("*.pt"))
            if path.name not in keep_names
        )

    delete_bytes = sum(directory_size(path) for path in delete_paths)
    prune_bytes = sum(path.stat().st_size for path in prune_paths)
    print(
        f"delete_dirs={len(delete_paths)} prune_checkpoints={len(prune_paths)} "
        f"reclaim_gib={(delete_bytes + prune_bytes) / 1024**3:.2f}"
    )
    if not args.execute:
        print("Dry run only. Pass --execute to delete.")
        return

    removed_dirs = []
    removed_checkpoints = []
    for path in delete_paths:
        shutil.rmtree(path)
        removed_dirs.append(path.name)
    for path in prune_paths:
        path.unlink()
        removed_checkpoints.append(str(path.relative_to(root)))

    manifest = {
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        "removed_bytes": delete_bytes + prune_bytes,
        "removed_gib": (delete_bytes + prune_bytes) / 1024**3,
        "removed_directories": removed_dirs,
        "removed_checkpoints": removed_checkpoints,
        "protected_directories": sorted(PROTECTED_DIRS),
    }
    manifest_path = root / "CLEANUP_EXECUTED_20260814.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote={manifest_path}")


if __name__ == "__main__":
    main()
