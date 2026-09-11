"""Torch-free regressions for coverage, DDP schedules and real grid semantics."""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import importlib.util
import math
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('motion_sampling_under_test', ROOT / 'football_object_motion/sampling.py')
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


def load_definitions(path, names, **scope):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    scope['__name__'] = __name__
    exec(compile(module, str(ROOT / path), 'exec'), scope)
    return SimpleNamespace(**scope)


class Config(dict):
    def __getattr__(self, key):
        return self[key]


def fixture_builders():
    base = load_definitions('train_football_events.py',
                            ['LongVideoRecord', 'FootballEvent', 'labels_for_window',
                             'label_mask_for_sample', 'rejected_set_piece_label_mask'],
                            dataclass=dataclass, np=np, LABELS=('shot', 'save', 'set_piece'),
                            LABEL_TO_INDEX={'shot': 0, 'save': 1, 'set_piece': 2}, ConfigDict=Config)
    e13 = load_definitions('train_football_events_online_simulation_e13.py',
                           ['build_online_eval_records', '_support_interval', '_interval_gap', 'e13_window_weights'],
                           base=base, ConfigDict=Config, replace=replace, math=math)
    e16 = load_definitions('train_football_events_online_simulation_e16.py',
                           ['e16_window_weights', '_mask_ambiguous_cross_label_negatives', '_e16_record'],
                           base=base, e13=e13, ConfigDict=Config, replace=replace)
    cfg = Config(video=Config(clip_duration=10),
                 eval=Config(online_validation=Config(window_stride_sec=5)),
                 data=Config(long_video=Config(online_simulation=Config())))
    return base, e13, e16, cfg


def record(base, video, duration=23, mask=(1., 1., 0.), **overrides):
    values = dict(source='fixture', split='train', video_id=video, sample_id='seed_' + video,
                  video_path=video + '.mp4', annotation_path='fixture.json', anchor_time=7.,
                  base_clip_start=2., base_clip_end=12., video_duration=duration,
                  is_negative=False, labels=(1., 0., 0.), label_mask=mask)
    return base.LongVideoRecord(**{**values, **overrides})


class DenseSamplingTest(unittest.TestCase):
    def test_real_eval_grid_includes_unpaired_background_video_and_tail(self):
        base, e13, e16, cfg = fixture_builders()
        rows = [record(base, 'event'), record(base, 'background')]
        events = {('fixture', 'event'): [base.FootballEvent('fixture', 'event', 'e', 'shot', 'shot', 7, 7, 7, (1., 0., 0.))]}
        reference = e13.build_online_eval_records(rows, events, cfg, online_cfg=cfg.eval.online_validation)
        dense = sampling.build_dense_training_records(rows, events, cfg,
                    grid_builder=e13.build_online_eval_records, record_builder=e16._e16_record)
        geometry = lambda records: [(r.source, r.video_id, r.base_clip_start, r.base_clip_end) for r in records]
        self.assertEqual(geometry(dense), geometry(reference))
        self.assertEqual([r.base_clip_start for r in dense if r.video_id == 'background'], [0., 5., 10., 13.])
        self.assertTrue(all(r.is_negative for r in dense if r.video_id == 'background'))
        self.assertTrue(all(r.label_mask[2] == 0 for r in dense))
        self.assertTrue(all(not r.online_pair_id for r in dense))

    def test_real_context_ignore_masks_are_preserved(self):
        base, e13, e16, cfg = fixture_builders()
        row = record(base, 'context', duration=40, online_pair_id='stale', online_pair_role='edge',
                     online_clip_loss_weights=(0., 0., 0.))
        events = {('fixture', 'context'): [base.FootballEvent('fixture', 'context', 'e', 'shot', 'shot',
                   20, 30, 25, (1., 0., 0.), context_start_time=12, context_end_time=30)]}
        dense = sampling.build_dense_training_records([row], events, cfg,
                    grid_builder=e13.build_online_eval_records, record_builder=e16._e16_record)
        before = next(r for r in dense if r.base_clip_start == 5)
        positive = next(r for r in dense if r.base_clip_start == 20)
        self.assertEqual(before.labels[0], 0)
        self.assertEqual(before.online_clip_loss_weights[0], 0)
        self.assertEqual(before.online_frame_loss_weights[0], 0)
        self.assertEqual(positive.online_clip_loss_weights[0], 1)
        self.assertTrue(all(not r.online_pair_id and not r.online_pair_class_mask for r in dense))

    def test_non_train_inventory_rejected(self):
        base, e13, e16, cfg = fixture_builders()
        with self.assertRaisesRegex(ValueError, 'train split'):
            sampling.build_dense_training_records([record(base, 'heldout', split='val')], {}, cfg,
                grid_builder=e13.build_online_eval_records, record_builder=e16._e16_record)

    def test_production_load_hook_changes_only_dense_training(self):
        base, e13, e16, cfg = fixture_builders()
        cfg['model'] = Config(object_motion=Config(sampling_mode='dense_mixed'))
        rows = [record(base, 'background', duration=23)]
        delegated = []
        online = SimpleNamespace(e13=e13, _e16_record=e16._e16_record,
            _original_load=lambda value, split: (rows, {}),
            online_load=lambda value, split: (delegated.append(split) or [], {}))
        hook = load_definitions('football_object_motion/train.py', ['_motion_cfg', 'load_motion_records'],
            base=base, online_e16=online, build_dense_training_records=sampling.build_dense_training_records,
            json=json)
        dense, _ = hook.load_motion_records(cfg, 'train')
        self.assertEqual(len(dense), 4)
        self.assertEqual(delegated, [])
        hook.load_motion_records(cfg, 'val')
        cfg.model.object_motion['sampling_mode'] = 'legacy_pairs'
        hook.load_motion_records(cfg, 'train')
        self.assertEqual(delegated, ['val', 'train'])

    def test_real_pair_builder_keeps_natural_pool_and_allows_empty_pairs(self):
        base, e13, e16, cfg = fixture_builders()
        rows = [record(base, 'event', duration=40), record(base, 'background', duration=40)]
        events = {('fixture', 'event'): [base.FootballEvent('fixture', 'event', 'e', 'shot', 'shot', 7, 7, 7, (1., 0., 0.))]}
        dense = sampling.build_dense_training_records(rows, events, cfg,
                    grid_builder=e13.build_online_eval_records, record_builder=e16._e16_record)
        original_ids = [r.sample_id for r in dense]
        builder = load_definitions('football_object_motion/train.py',
            ['_motion_cfg', '_load_reviewed_negative_entries', '_usable_relation_negative', 'build_relation_pair_rows'],
            base=base, Path=Path, json=json, replace=replace)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'reviewed.json'
            path.write_text(json.dumps({'hard_negatives': [dict(source='fixture', video_id='event',
                start_sec=22, end_sec=32, center_sec=27, labels=['shot'])]}))
            cfg['model'] = Config(object_motion=Config(reviewed_negative_manifests=[str(path)]))
            dataset = SimpleNamespace(records=dense, events_by_video=events)
            pairs = builder.build_relation_pair_rows(dataset, cfg, allow_empty=True)
            self.assertEqual(len(pairs), 2)
            self.assertEqual([r.sample_id for r in dense[:len(original_ids)]], original_ids)
            self.assertEqual(dense[pairs[1]['base_index']].base_clip_start, 22)
            background = SimpleNamespace(records=[r for r in dense if r.video_id == 'background'], events_by_video={})
            self.assertEqual(builder.build_relation_pair_rows(background, cfg, allow_empty=True), [])

    def test_ordinary_indexing_is_not_restricted_to_pairs(self):
        pairs = [dict(base_index=1, pair_id='a'), dict(base_index=7, pair_id='a')]
        self.assertEqual([sampling.resolve_stream_row(i, 6, pairs, mixed=True)[0] for i in range(8)],
                         [0, 1, 2, 3, 4, 5, 1, 7])
        with self.assertRaises(IndexError):
            sampling.resolve_stream_row(-1, 6, pairs, mixed=True)

    def test_metadata_does_not_leak_and_decode_replacement_is_not_reviewed(self):
        expected = SimpleNamespace(source='s', video_id='v', sample_id='a')
        item = dict(meta=dict(source='s', video_id='other', sample_id='b', sampled_clip_start=0,
                             sampled_clip_end=10, pair_id='stale', reviewed_negative=True))
        pair = dict(base_index=0, pair_id='pair', pair_role='negative', reviewed_negative=True, full_clean_window=True)
        paired = sampling.annotate_stream_item(item, expected, pair, stream='pair')
        self.assertFalse(paired['meta']['pair_usable'])
        self.assertFalse(paired['meta']['reviewed_negative'])
        ordinary = sampling.annotate_stream_item(item, expected, None, stream='natural')
        self.assertNotIn('pair_id', ordinary['meta'])
        self.assertEqual(item['meta']['pair_id'], 'stale')  # No mutation of cached sample.

    def test_ddp_equal_budgets_disjoint_natural_shards_and_complete_pairs(self):
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                all_natural = []
                for rank in range(4):
                    sampler = sampling.DensePairBatchSampler(64, [(64, 65), (66, 67), (68, 69)],
                        batches_per_rank=8, batch_size=batch_size, rank=rank, world_size=4)
                    sampler.set_epoch(1)
                    batches = list(sampler)
                    self.assertEqual(len(batches), 8)
                    self.assertTrue(all(len(b) == batch_size for b in batches))
                    flat = [i for b in batches for i in b]
                    all_natural.extend(i for i in flat if i < 64)
                    for offset in range(0, len(flat), 2):
                        left, right = flat[offset:offset + 2]
                        self.assertEqual(left < 64, right < 64)
                        if left >= 64:
                            self.assertEqual((left - 64) // 2, (right - 64) // 2)
                            if batch_size == 2:
                                self.assertEqual(left % 2, 0)
                self.assertEqual(len(all_natural), len(set(all_natural)))

    def test_epoch_rotation_covers_the_entire_pool_before_repeating(self):
        sampler = sampling.DensePairBatchSampler(10, [], batches_per_rank=4)
        sequence = []
        for epoch in (1, 2, 3):
            sampler.set_epoch(epoch)
            sequence.extend(i for batch in sampler for i in batch)
        self.assertEqual(len(set(sequence[:10])), 10)
        resumed = sampling.DensePairBatchSampler(10, [], batches_per_rank=4)
        resumed.set_epoch(3)
        self.assertEqual([i for batch in resumed for i in batch], sequence[8:])

    def test_pair_order_flips_and_epoch_resets_queue(self):
        calls = []
        sampler = sampling.DensePairBatchSampler(0, [(0, 1)], batches_per_rank=2,
            natural_fraction=0, on_epoch=lambda: calls.append(True))
        sampler.set_epoch(1)
        self.assertEqual(list(sampler), [[1], [0]])
        sampler.set_epoch(2)
        self.assertEqual(list(sampler), [[0], [1]])
        self.assertEqual(len(calls), 2)

    def test_invalid_budget_and_shard_fail_closed(self):
        for extra in (dict(batches_per_rank=3), dict(world_size=0), dict(rank=2, world_size=2),
                      dict(natural_fraction=float('nan')), dict(batch_size=3)):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                sampling.DensePairBatchSampler(10, [(10, 11)], **{'batches_per_rank': 4, **extra})


if __name__ == '__main__':
    unittest.main()
