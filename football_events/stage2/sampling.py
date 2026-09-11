"""Two ball candidates and exact original-patch core/context gathering.

Candidate zero is unrestricted. A positive height prior only ranks the second
candidate. Saliency descriptors always use the original unmodified heatmap.
"""
import math
import torch
from torchvision.ops import roi_align

def height_prior(config, device):
    y = (torch.arange(45, device=device, dtype=torch.float32) + 0.5) / 45
    floor = float(config['top_weight'])
    low = float(config['top_end'])
    high = float(config['full_weight_start'])
    if not (0 < floor <= 1 and 0 <= low < high <= 1):
        raise ValueError('Invalid positive soft height prior')
    weight = floor + (1 - floor) * ((y - low) / (high - low)).clamp(0, 1)
    if config['mode'] == 'none':
        weight = torch.ones_like(weight)
    elif config['mode'] != 'soft_top':
        raise ValueError('Unknown prior mode')
    return weight[:, None].expand(45, 80).reshape(3600)

def select_peaks(logits, config):
    if logits.ndim != 2 or logits.shape[1] != 3600 or (not torch.isfinite(logits).all()):
        raise ValueError('Expected finite ball logits [frames,3600]')
    raw = logits.float()
    first = raw.argmax(1)
    ids = torch.arange(3600, device=raw.device)
    suppress = ((ids[None] % 80 - first[:, None] % 80).abs() <= 3) & ((ids[None] // 80 - first[:, None] // 80).abs() <= 3)
    score = raw + height_prior(config, raw.device).log()[None]
    second = score.masked_fill(suppress, -torch.inf).argmax(1)
    return torch.stack([first, second], 1)

def tokens_at_peaks(patches, logits, peaks):
    """Two peaks, each nine EXACT patch vectors + four spatial context vectors.

    Out-of-grid core cells are masked, not clamped duplicates. Index 4 in each
    13-token group is always its exact center. No learned preprocessing here.
    Explicit candidate indices; all saliency descriptors use UNMODIFIED logits.
    Descriptor: x,y,dx,dy,scale,kind,relative_salience,entropy,peakp,gap,candidate.
    """
    b, n, d = patches.shape
    assert n == 3600 and logits.shape == (b, n) and (peaks.shape == (b, 2))
    assert ((peaks >= 0) & (peaks < 3600)).all()
    device = patches.device
    prob = logits.float().softmax(1)
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum(1) / math.log(n)
    top = prob.topk(2, dim=1).values
    dy, dx = torch.meshgrid(torch.arange(-1, 2, device=device), torch.arange(-1, 2, device=device), indexing='ij')
    dx, dy = (dx.flatten(), dy.flatten())
    features = []
    descs = []
    masks = []
    indices = []
    fmap = patches.transpose(1, 2).reshape(b, d, 45, 80).float()
    for candidate in range(2):
        peak = peaks[:, candidate]
        cx, cy = (peak % 80, peak // 80)
        gx, gy = (cx[:, None] + dx, cy[:, None] + dy)
        valid = (gx >= 0) & (gx < 80) & (gy >= 0) & (gy < 45)
        ix = (gy.clamp(0, 44) * 80 + gx.clamp(0, 79)).long()
        raw = patches.gather(1, ix[..., None].expand(-1, -1, d))
        raw = torch.where(valid[..., None], raw, torch.zeros_like(raw))
        center = torch.stack([cx + 0.5, cy + 0.5], -1).float()
        lo = (center - 5.5).clamp_min(0)
        hi = torch.minimum(center + 5.5, center.new_tensor([80.0, 45.0]))
        rois = torch.cat([torch.arange(b, device=device)[:, None], lo, hi], -1)
        context = roi_align(fmap, rois, (2, 2), spatial_scale=1.0, sampling_ratio=0, aligned=True).flatten(2).transpose(1, 2)
        qx, qy = torch.meshgrid(torch.tensor([0.25, 0.75], device=device), torch.tensor([0.25, 0.75], device=device), indexing='xy')
        xycontext = lo[:, None] + torch.stack([qx.flatten(), qy.flatten()], -1)[None] * (hi - lo)[:, None]
        xycore = torch.stack([gx + 0.5, gy + 0.5], -1).float()
        xy = torch.cat([xycore, xycontext], 1)
        relative = (xy - center[:, None]) / 11
        pos = xy / xy.new_tensor([80.0, 45.0])
        scale = torch.cat([torch.full((b, 9, 1), 1 / 11, device=device), torch.ones(b, 4, 1, device=device)], 1)
        kind = torch.cat([torch.zeros(b, 9, 1, device=device), torch.ones(b, 4, 1, device=device)], 1)
        salience = torch.cat([
            torch.log1p(3600 * prob.gather(1, ix)),
            torch.log1p(3600 * prob.gather(1, (cy * 80 + cx)[:, None])).expand(-1, 4),
        ], 1)[..., None]
        stats = torch.stack([entropy, top[:, 0], top[:, 0] - top[:, 1]], -1)[:, None].expand(-1, 13, -1)
        desc = torch.cat([pos, relative, scale, kind, salience, stats, torch.full((b, 13, 1), float(candidate), device=device)], -1)
        features.append(torch.cat([raw.float(), context], 1))
        descs.append(desc)
        masks.append(torch.cat([valid, torch.ones(b, 4, device=device, dtype=torch.bool)], 1))
        indices.append(torch.cat([torch.where(valid, ix, -torch.ones_like(ix)), torch.full((b, 4), -1, device=device, dtype=torch.long)], 1))
    return {
        'tokens': torch.cat(features, 1),
        'descriptors': torch.cat(descs, 1),
        'valid': torch.cat(masks, 1),
        'patch_indices': torch.cat(indices, 1),
    }
