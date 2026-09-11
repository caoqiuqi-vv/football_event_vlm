"""E1.3 checkpoint + trainability gate (plan 5.1.5/5.1.7, CPU-only).

Loads the real model with the E1.3 config, audits the Epoch0 control
checkpoint for unexpected/shape-mismatched weights, and verifies that
missing weights are exactly the new no-evidence content gate tensors.
Also confirms the trainable parameter split: LoRA in last6/r8 only, legacy
null scalar/value frozen.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/new_users/qiuqi/code/dinov3-main")

import torch

import train_football_events as base

CONFIG_PATH = (
    "/home/new_users/qiuqi/code/dinov3-main/configs/football/"
    "dinov3_vitl16_online_simulation_e13_paired_gate_from_last6r8.yaml"
)


def main() -> int:
    cfg = base.load_config(CONFIG_PATH, [])
    base.configure_label_schema(cfg)
    print("building model (CPU)...", flush=True)
    model = base.make_model(cfg, use_cached_features=False, device=torch.device("cpu"))
    checkpoint_path = cfg.model.init_checkpoint
    print(f"auditing init checkpoint {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = base.strip_module_prefix(raw_state)
    target_state = model.state_dict()

    unexpected: list[str] = []
    skipped: list[str] = []
    matched: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        target_key = key
        if target_key not in target_state:
            if key.endswith(".weight") or key.endswith(".bias"):
                suffix = ".weight" if key.endswith(".weight") else ".bias"
                candidate = key[: -len(suffix)] + ".base" + suffix
                if candidate in target_state:
                    target_key = candidate
        if target_key not in target_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[target_key].shape):
            skipped.append(f"{target_key}: {tuple(value.shape)} != {tuple(target_state[target_key].shape)}")
            continue
        matched[target_key] = value

    missing = [key for key in target_state if key not in matched]
    gate_keys = [key for key in missing if "no_evidence_gate" in key]
    other_missing = [key for key in missing if key not in gate_keys]
    print(f"matched={len(matched)} unexpected={len(unexpected)} skipped_shape={len(skipped)} "
          f"missing={len(missing)} (new gate={len(gate_keys)}, other={len(other_missing)})")
    for key in unexpected[:10]:
        print("  UNEXPECTED:", key)
    for key in skipped[:10]:
        print("  SKIPPED:", key)
    for key in other_missing[:20]:
        print("  OTHER_MISSING:", key)
    for key in gate_keys:
        print("  NEW_GATE:", key)

    errors: list[str] = []
    if unexpected:
        errors.append(f"unexpected weights: {unexpected[:10]}")
    if skipped:
        errors.append(f"shape mismatches: {skipped[:10]}")
    if other_missing:
        errors.append(f"unexplained missing weights: {other_missing[:20]}")
    if len(gate_keys) != 2:
        errors.append(f"expected exactly 2 new gate tensors, got {len(gate_keys)}: {gate_keys}")
    gate_params = sum(target_state[key].numel() for key in gate_keys)
    if gate_params > 1024:
        errors.append(f"new gate too large: {gate_params} params > 1024")
    print(f"new gate params={gate_params}")

    # Trainability split (plan 5.1.7).
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    lora_trainable = [name for name in trainable if "lora_" in name]
    backbone_trainable = [
        name for name in trainable if "backbone" in name and "lora_" not in name
    ]
    null_trainable = [
        name for name, p in model.named_parameters()
        if ("no_evidence_logits" in name or "no_evidence_values" in name) and p.requires_grad
    ]
    print(f"trainable total={len(trainable)} lora={len(lora_trainable)} "
          f"backbone_non_lora={len(backbone_trainable)} null_params={len(null_trainable)}")
    lora_blocks = {
        name.split(".")[1] for name in lora_trainable if len(name.split(".")) > 1
    }
    # LoRA names look like backbone.blocks.N.attn.qkv.lora_A.weight
    lora_block_ids = {
        int(name.split(".")[2]) for name in lora_trainable if name.split(".")[2].isdigit()
    }
    print(f"lora trainable block ids: {sorted(lora_block_ids)}")
    if backbone_trainable:
        errors.append(f"non-LoRA backbone params trainable: {backbone_trainable[:5]}")
    if null_trainable:
        errors.append(f"legacy null params trainable: {null_trainable}")
    # DINOv3 ViT-L has 24 blocks; plan fixes last6.
    if lora_block_ids != {18, 19, 20, 21, 22, 23}:
        errors.append(f"LoRA blocks {sorted(lora_block_ids)} != last6 {{18..23}}")

    if errors:
        print(f"CHECKPOINT_GATE_FAILED with {len(errors)} errors:")
        for line in errors:
            print("  ", line)
        return 1
    print("CHECKPOINT_GATE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
