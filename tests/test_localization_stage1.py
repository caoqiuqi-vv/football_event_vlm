import torch
from scripts.train_football_localization_stage1 import feature_kl, spatial_targets, localization_loss


def test_feature_kl_identity_and_gradient():
    torch.manual_seed(4)
    t = torch.randn(2, 10, 32, requires_grad=True)
    s = t.detach().clone().requires_grad_(True)
    assert abs(float(feature_kl(s, t))) < 1e-6
    s = (t.detach()+.2*torch.randn_like(t)).requires_grad_(True)
    loss = feature_kl(s, t)
    assert loss > 0
    loss.backward()
    assert s.grad.norm() > 0 and t.grad is None


def test_spatial_targets_unknown_and_geometry():
    boxes = torch.zeros(2,2,4,4)
    boxes[0,0,0] = torch.tensor([.49,.49,.51,.51])
    boxes[1,1,0] = torch.tensor([.1,.2,.3,.5])
    valid = torch.tensor([[True,False],[False,True]])
    targets = spatial_targets(boxes, valid)
    assert targets.shape == (2,3600,2)
    assert torch.allclose(targets.sum(1), valid.float(), atol=1e-6)
    logits = torch.randn_like(targets, requires_grad=True)
    loss = localization_loss(logits, targets, valid.float())
    loss.backward()
    assert torch.count_nonzero(logits.grad[0,:,1]) == 0
    assert torch.count_nonzero(logits.grad[1,:,0]) == 0
    assert torch.count_nonzero(logits.grad[0,:,0]) > 0
