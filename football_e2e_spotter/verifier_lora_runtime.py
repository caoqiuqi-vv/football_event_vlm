"""Load the local LoRA file without importing the DINO training package."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from torch import nn
def configure(backbone: nn.Module, *, warmup: bool, lora_rank: int = 16) -> int:
 path=Path(__file__).resolve().parents[1]/'dinov3/train/lora.py'; spec=importlib.util.spec_from_file_location('_set_spotter_lora',path); module=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
 for p in backbone.parameters():p.requires_grad_(False)
 if warmup:return 0
 n=module.inject_lora(backbone,{'rank':lora_rank,'target_last_blocks':20,'train_norm':True});module.reset_lora_parameters(backbone)
 for block in list(backbone.blocks)[-4:]:
  for p in block.parameters():p.requires_grad_(True)
 return n
