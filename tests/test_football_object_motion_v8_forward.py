"""Exercise the production motion forward with a tiny substituted backbone.

AST-loading this exact function avoids importing the unrelated monolithic data
trainer and its video dependencies; the fusion function itself is not copied.
This is an integration test, not a real-DINO CUDA memory smoke.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from football_object_motion.ball_backbone import BallLoRALinear, ball_lora_enabled
from football_object_motion.model import ObjectMotionEvidenceAdapter, interpolate_motion_residual


def load_forward(extractor):
    path = Path(__file__).resolve().parents[1] / 'football_object_motion/train.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_motion_forward')
    scope = dict(torch=torch, nn=nn, Tensor=Tensor, Any=Any,
                 ball_lora_enabled=ball_lora_enabled,
                 extract_intermediate_patch_layers=extractor,
                 interpolate_motion_residual=interpolate_motion_residual)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
    return scope['_motion_forward']


class TinyModel(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.backbone = BallLoRALinear(nn.Linear(8, 8), rank=2, alpha=2)
        self.backbone.enabled = True
        self.event_head = nn.Linear(8, 3)
        self.global_projection = nn.Linear(8, 16)
        for module in (self.event_head, self.global_projection):
            for p in module.parameters():
                p.requires_grad = False
        self.object_motion_adapter = ObjectMotionEvidenceAdapter(
            patch_dim=8, hidden_dim=16, num_labels=3, num_heads=4,
            temporal_layers=1, dropout=0, object_cross_attention_enabled=True,
            heatmap_upsample_factor=2, ball_fpn_dim=8,
            event_relation_grad_enabled=True, event_context_enabled=True,
            event_frame_fusion_enabled=True,
        )
        self.__dict__['_object_motion_original_forward'] = self.anchor
        motion_cfg = dict(shared_anchor_ball_lora=True, object_cross_attention_enabled=True,
                          legacy_frame_residual_enabled=False, ball_feature_layers=[0, 1, 2, 3])
        self.__dict__['_object_motion_runtime_cfg'] = SimpleNamespace(
            model=SimpleNamespace(object_motion=motion_cfg),
            train={'object_motion_event_residual_scale': alpha},
        )
        self.__dict__['_object_motion_backbone_patch_size'] = 1

    def anchor(self, images, return_aux=False):
        features = self.backbone(images.mean(dim=(-1, -2)))
        frame_logits = self.event_head(features)
        return dict(logits=frame_logits.mean(1), frame_event_logits=frame_logits,
                    global_temporal=self.global_projection(features.mean(1)))


def extractor(model, frames, *, layers, **kwargs):
    tokens = frames.permute(0, 1, 3, 4, 2).flatten(2, 3)
    tokens = model.backbone(tokens)
    return tokens.unsqueeze(2).expand(-1, -1, len(layers), -1, -1)


@pytest.mark.parametrize('alpha', [0.0, 0.25, 1.0])
def test_production_forward_frame_scale_and_shared_lora_gradient(alpha):
    torch.manual_seed(31)
    model = TinyModel(alpha)
    fusion = model.object_motion_adapter.object_cross_attention
    with torch.no_grad():
        nn.init.normal_(fusion.delta_head[-1].weight, std=0.05)
        nn.init.normal_(fusion.frame_delta_head[-1].weight, std=0.05)
    images = torch.randn(1, 5, 8, 3, 3)
    times = torch.arange(5).float().unsqueeze(0)
    model.__dict__['_object_motion_runtime_inputs'] = images
    model.__dict__['_object_motion_runtime_times'] = times
    model.__dict__['_object_motion_runtime_global_times'] = times
    reference = model.anchor(images)
    result = load_forward(extractor)(model, images, return_aux=True)
    torch.testing.assert_close(result['object_motion_frame_residual'],
                               alpha * result['object_motion_adapter_frame_residual'])
    torch.testing.assert_close(result['frame_event_logits'],
                               reference['frame_event_logits'] + result['object_motion_frame_residual'])
    torch.testing.assert_close(result['logits'],
                               reference['logits'] + alpha * result['object_motion_adapter_clip_residual'])
    objective = result['logits'].square().mean() + result['frame_event_logits'].square().mean()
    objective.backward()
    assert model.backbone.ball_lora_b.grad is not None
    assert model.backbone.ball_lora_b.grad.abs().sum() > 0
    temporal_grad = sum(float(p.grad.abs().sum()) for p in model.object_motion_adapter.temporal.parameters() if p.grad is not None)
    assert (temporal_grad > 0) == (alpha > 0)
    assert not any(p.grad is not None for p in model.object_motion_adapter.ball_layer_heads.parameters())
