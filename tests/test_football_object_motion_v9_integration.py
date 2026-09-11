"""Real DataLoader/queue regressions; no video decode or DINO weights required."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from football_object_motion.sampling import DensePairBatchSampler, annotate_stream_item, resolve_stream_row
from football_object_motion.losses import (
    AdjacentPairResidualQueue, _validated_pairwise_ranking, reset_pair_residual_queues,
    _queued_pairwise_ranking, _PAIR_RESIDUAL_QUEUES,
    motion_pair_ranking,
)
from football_object_motion.teacher import OnlineObjectMotionTeacher
from test_football_object_motion_v9_sampling import Config, load_definitions


def pair_meta(role, usable=True):
    return dict(pair_id='p', pair_role=role, relation_class_index=0, source='s', video_id='v',
                reviewed_negative=role == 'negative', full_clean_window=True, review_manifest='fixture.json',
                nearest_same_class_gt_gap=10, pair_usable=usable)


def test_real_queue_skips_failed_pair_and_recovers_with_gradients():
    queue = AdjacentPairResidualQueue()
    for usable in (False, True):
        negative = torch.tensor([[0.5]], requires_grad=True)
        positive = torch.tensor([[0.0]], requires_grad=True)
        queue.consume(negative, torch.zeros(1, 1), torch.ones(1, 1), pair_meta('negative', usable),
                      margin=.05, temperature=.1, min_gap_sec=5)
        loss, count = queue.consume(positive, torch.ones(1, 1), torch.ones(1, 1), pair_meta('positive'),
                                   margin=.05, temperature=.1, min_gap_sec=5)
        assert count == int(usable)
        loss.backward()
        assert (positive.grad.abs().sum() > 0).item() == usable
        assert queue.pending is None


def test_real_batched_pair_skips_unusable_supervision():
    scores = torch.tensor([[0.], [.5]], requires_grad=True)
    targets = torch.tensor([[1.], [0.]])
    masks = torch.tensor([[0.], [1.]])
    loss, count = _validated_pairwise_ranking(scores, targets, masks,
        [pair_meta('positive'), pair_meta('negative')], margin=.05, temperature=.1, min_gap_sec=5)
    assert count == 0
    loss.backward()
    assert torch.count_nonzero(scores.grad) == 0


class BaseRows(Dataset):
    is_train = True
    events_by_video = {}
    def __init__(self):
        self.records = [SimpleNamespace(sample_id=f'motion_dense_train_v_w{i}', source='s', video_id=f'v{i}')
                        for i in range(4)]
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        record = self.records[index]
        # Force the deterministic, masked decode-failure branch; no video files.
        return dict(meta=dict(vars(record), sampled_clip_start=0., sampled_clip_end=10., decode_failed=True))


def production_dataset():
    rows = BaseRows()
    base = SimpleNamespace(parse_image_size=lambda value: value,
                           frame_label_sigma_seconds=lambda cfg: .5,
                           frame_label_ignore_radius_seconds=lambda cfg: 1., LABELS=('shot', 'save', 'set_piece'))
    pair_rows = [dict(base_index=0, **pair_meta('positive')), dict(base_index=1, **pair_meta('negative'))]
    scope = load_definitions('football_object_motion/train.py', ['ObjectMotionDataset'],
        Dataset=Dataset, base=base, torch=torch, OnlineObjectMotionTeacher=OnlineObjectMotionTeacher,
        build_relation_pair_rows=lambda *args, **kwargs: pair_rows,
        resolve_stream_row=resolve_stream_row, annotate_stream_item=annotate_stream_item)
    cfg = SimpleNamespace(model=SimpleNamespace(object_motion=dict(
        sampling_mode='dense_mixed', require_true_pairs=True, image_size=[16, 16], frames_per_segment=1)))
    dataset = scope.ObjectMotionDataset(rows, cfg)
    return dataset


def collect_meta(batch):
    return [item['meta'] for item in batch]


@pytest.mark.parametrize('workers', [0, 1])
def test_production_dataset_dataloader_reaches_unpaired_rows(workers):
    dataset = production_dataset()
    assert len(dataset) == 6
    assert dataset.relation_pair_indices == [(4, 5)]  # Four natural rows + two aliases.
    sampler = DensePairBatchSampler(dataset.natural_count, dataset.relation_pair_indices,
                                   batches_per_rank=8, on_epoch=reset_pair_residual_queues)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                        persistent_workers=bool(workers), collate_fn=collect_meta)
    for epoch in (1, 2):
        sampler.set_epoch(epoch)
        batches = list(loader)
        natural = [meta for batch in batches for meta in batch if meta['object_motion_sampling_stream'] == 'natural']
        assert {m['video_id'] for m in natural} == {'v0', 'v1', 'v2', 'v3'}
        assert all('pair_id' not in m for m in natural)
        assert all(not m['pair_usable'] for batch in batches for m in batch if m.get('pair_id'))


def test_sampler_epoch_reset_clears_real_pending_queue():
    reset_pair_residual_queues()
    _queued_pairwise_ranking(torch.zeros(1, 1, requires_grad=True), torch.ones(1, 1), torch.ones(1, 1),
        [pair_meta('positive')], margin=.05, temperature=.1, min_gap_sec=5)
    assert _PAIR_RESIDUAL_QUEUES
    sampler = DensePairBatchSampler(4, [], batches_per_rank=4, on_epoch=reset_pair_residual_queues)
    sampler.set_epoch(2)
    assert not _PAIR_RESIDUAL_QUEUES


def test_ordinary_batch_clears_queue_and_never_ranks_random_negatives():
    reset_pair_residual_queues()
    cfg = {'require_true_pairs': True}
    scores = torch.zeros(1, 1, requires_grad=True)
    motion_pair_ranking(scores, torch.ones_like(scores), torch.ones_like(scores), [pair_meta('positive')], cfg)
    assert _PAIR_RESIDUAL_QUEUES
    loss, count = motion_pair_ranking(scores, torch.zeros_like(scores), torch.ones_like(scores), [{}], cfg)
    assert count == 0 and not _PAIR_RESIDUAL_QUEUES
    loss.backward()
    assert torch.count_nonzero(scores.grad) == 0
    with pytest.raises(ValueError, match='separate microbatches'):
        motion_pair_ranking(torch.zeros(2, 1), torch.zeros(2, 1), torch.ones(2, 1),
                            [{}, pair_meta('negative')], cfg)


def test_production_loader_exposes_epoch_sampler_and_fixed_budget():
    dataset = production_dataset()
    base = SimpleNamespace(distributed_training_active=lambda: False)
    cfg = Config(model=Config(object_motion=Config(sampling_mode='dense_mixed', require_true_pairs=True,
                 sampling_batches_per_rank=8, natural_window_fraction=.5)),
                 train=Config(batch_size=1, grad_accum_steps=2), data=Config(num_workers=0, pin_memory=False))
    def unexpected_loader(*args, **kwargs):
        raise AssertionError('mixed training unexpectedly delegated to the legacy loader')
    scope = load_definitions('football_object_motion/train.py', ['_PairLoaderView', 'make_motion_loader'],
        base=base, torch=torch, DataLoader=DataLoader, DensePairBatchSampler=DensePairBatchSampler,
        _motion_cfg=lambda value: value.model.object_motion, motion_collate=collect_meta,
        reset_pair_residual_queues=reset_pair_residual_queues, _ORIGINAL_MAKE_LOADER=unexpected_loader)
    loader = scope.make_motion_loader(dataset, cfg, is_train=True)
    assert isinstance(loader.sampler, DensePairBatchSampler)
    loader.sampler.set_epoch(1)
    assert len(list(loader)) == len(loader) == 8
