"""Frozen localization guidance with original-event patches and trainable temporal fusion.

The legacy shared backbone/reader classes are retained to preserve checkpoint
and frozen-teacher behavior; spatial sampling is maintained in this package.
"""
import torch
from torch import nn
from football_stage2_optional import FrozenStage2Extractor, candidate_features
from football_stage2_joint import JointTemporalReader, JointTemporalEventModel
from .sampling import select_peaks, tokens_at_peaks

class PositionPriorExtractor(FrozenStage2Extractor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prior = cfg['position_prior']

    @torch.no_grad()
    def extract_frames(self, frames):
        assert frames.shape[-2:] == (720, 1280)
        x = (frames.float() / 255 - self.rgb_mean) / self.rgb_std
        x, (h, w) = self.backbone.prepare_tokens_with_masks(x)
        rope = self.backbone.rope_embed(H=h, W=w)
        for block in self.backbone.blocks[:self.start]:
            x = block(x, rope)
        t, s = (x, x)
        for block in self.teacher_tail:
            t = block(t, rope)
        for block in self.backbone.blocks[self.start:]:
            s = block(s, rope)
        t, s = (self.norm_tokens(t), self.norm_tokens(s))
        n = 1 + self.backbone.n_storage_tokens
        original, adapted = (t[:, n:], s[:, n:])
        logits = self.adapt_head(adapted)[:, :, 0]
        peaks = select_peaks(logits, self.prior)
        result = {'stage1_' + k: v for k, v in tokens_at_peaks(original, logits, peaks).items()}
        result['global_features'] = torch.cat([t[:, 0], original.mean(1)], -1)
        result['legacy_tokens'], result['legacy_descriptors'] = candidate_features(adapted, logits)
        result['selected_peaks'] = peaks
        result['unweighted_peaks'] = select_peaks(logits, {**self.prior, 'mode': 'none'})
        return result

    @torch.no_grad()
    def extract_clip(self, frames, chunk=8):
        parts = [self.extract_frames(frames[i:i + chunk]) for i in range(0, len(frames), chunk)]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

class PositionPriorEventModel(JointTemporalEventModel):
    """Raw 720P inference uses exactly the same selector and FP16 cache rounding."""

    def __init__(self, checkpoint):
        nn.Module.__init__(self)
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        self.cfg = state['config']
        self.enabled = state['best']['enabled']
        self.extractor = PositionPriorExtractor(self.cfg)
        self.reader = JointTemporalReader(self.cfg, joint=True)
        self.reader.load_learned(state['learned'])
        self.reader.set_epoch(state['epoch'])
        self.register_buffer('normal_thresholds', torch.tensor(state['best']['thresholds']))
        self.register_buffer('baseline_thresholds', torch.tensor(state['baseline_thresholds']))
        self.eval()
