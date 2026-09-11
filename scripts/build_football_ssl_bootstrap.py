#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import OmegaConf
from torch import nn

from dinov3.configs import get_default_config
from dinov3.layers.dino_head import DINOHead
from dinov3.models import build_model
from dinov3.train.lora import LoRALinear, inject_lora, reset_lora_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a full SSL student bootstrap from official DINO weights.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone-weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(args.config))
    student, _teacher, embed_dim = build_model(
        cfg.student,
        only_teacher=False,
        img_size=int(cfg.crops.global_crops_size),
        device="cpu",
    )
    injected = inject_lora(student, cfg.lora) if bool(cfg.lora.enabled) else 0
    student.init_weights()
    if bool(cfg.lora.enabled):
        reset_lora_parameters(student)
    official = torch.load(args.backbone_weights, map_location="cpu", weights_only=False)
    lora_prefixes = {
        name for name, module in student.named_modules() if isinstance(module, LoRALinear)
    }
    mapped = {}
    for key, value in official.items():
        mapped_key = key
        for prefix in lora_prefixes:
            if key == prefix or key.startswith(prefix + "."):
                mapped_key = prefix + ".base" + key[len(prefix):]
                break
        mapped[mapped_key] = value
    missing, unexpected = student.load_state_dict(mapped, strict=False)
    expected_missing = {
        f"{prefix}.{suffix}" for prefix in lora_prefixes for suffix in ("lora_a", "lora_b")
    }
    unexpected_missing = sorted(set(missing) - expected_missing)
    if unexpected or unexpected_missing:
        raise RuntimeError(
            f"Backbone bootstrap mismatch missing={unexpected_missing[:20]} unexpected={unexpected[:20]}"
        )

    dino_head = DINOHead(
        in_dim=embed_dim,
        out_dim=int(cfg.dino.head_n_prototypes),
        hidden_dim=int(cfg.dino.head_hidden_dim),
        bottleneck_dim=int(cfg.dino.head_bottleneck_dim),
        nlayers=int(cfg.dino.head_nlayers),
    )
    ibot_head = DINOHead(
        in_dim=embed_dim,
        out_dim=int(cfg.ibot.head_n_prototypes),
        hidden_dim=int(cfg.ibot.head_hidden_dim),
        bottleneck_dim=int(cfg.ibot.head_bottleneck_dim),
        nlayers=int(cfg.ibot.head_nlayers),
    )
    dino_head.init_weights()
    ibot_head.init_weights()
    module = nn.ModuleDict({"backbone": student, "dino_head": dino_head, "ibot_head": ibot_head})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher": module.state_dict()}, output)
    metadata = {
        "config": str(Path(args.config).resolve()),
        "backbone_weights": str(Path(args.backbone_weights).resolve()),
        "output": str(output.resolve()),
        "lora_modules": injected,
        "lora_rank": int(cfg.lora.rank) if bool(cfg.lora.enabled) else 0,
        "trainable_backbone_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "total_backbone_parameters": sum(p.numel() for p in student.parameters()),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
