"""Fixed-budget dense-window and reviewed-pair sampling (stdlib only).

Records stay immutable after worker startup. Epoch changes affect indices only;
this avoids stale record/pair mappings with persistent DataLoader workers.
"""
from __future__ import annotations

import math
import json
import random
from dataclasses import replace
from typing import Callable, Iterator


def build_dense_training_records(records, events, cfg, *, grid_builder, record_builder):
    """Use the evaluation grid but retain E1.6 training ambiguity masks."""
    online_cfg = cfg.eval.get('online_validation', {})
    clip_sec = float(cfg.video.get('clip_duration', 10.0))
    stride = float(online_cfg.get('window_stride_sec', 5.0))
    if not math.isfinite(clip_sec) or clip_sec <= 0 or not math.isfinite(stride) or stride <= 0:
        raise ValueError('dense training requires finite positive clip duration and stride')
    if any(row.split != 'train' for row in records):
        raise ValueError('dense training records must belong to the train split')
    grid = grid_builder(records, events, cfg, online_cfg=online_cfg,
                        sample_prefix='motion_dense_train')
    result = []
    for index, row in enumerate(grid):
        safe = record_builder(
            row, events.get((row.source, row.video_id), ()),
            start=float(row.base_clip_start), clip_sec=clip_sec,
            mode='clean_background', sample_suffix=f'dense_train_{index:09d}', cfg=cfg,
        )
        # Evaluation records may inherit event-pair/cohort fields from their
        # representative. Explicitly discard those on ordinary grid windows.
        result.append(replace(safe, sample_id=row.sample_id,
                              online_pair_id='', online_pair_role='',
                              online_pair_class_mask=(), sample_loss_weight=1.0))
    if not result:
        raise ValueError('dense training grid is empty')
    return result


def resolve_stream_row(index, natural_count, pair_rows, *, mixed):
    """Virtual dataset: ordinary grid rows followed by pair aliases."""
    index = int(index)
    size = natural_count + len(pair_rows) if mixed else len(pair_rows)
    if index < 0 or index >= size:
        raise IndexError(index)
    if mixed and index < natural_count:
        return index, None
    row = pair_rows[index - natural_count if mixed else index]
    return int(row['base_index']), row


def annotate_stream_item(item, expected_record, pair_row, *, stream):
    """Never attach a valid reviewed pair to a decoder's replacement record."""
    item = dict(item)
    meta = dict(item['meta'])
    item['meta'] = meta
    for key in ('pair_id', 'pair_role', 'pair_label', 'relation_class_index',
                'reviewed_negative', 'full_clean_window', 'review_manifest',
                'nearest_same_class_gt_gap', 'pair_usable'):
        meta.pop(key, None)
    meta['object_motion_sampling_stream'] = stream
    if pair_row is not None:
        usable = not meta.get('decode_failed', False) and all(
            meta.get(key) == getattr(expected_record, key)
            for key in ('source', 'video_id', 'sample_id')
        )
        meta.update({key: value for key, value in pair_row.items() if key != 'base_index'})
        meta['pair_usable'] = usable
        if not usable:
            meta['reviewed_negative'] = False
            meta['full_clean_window'] = False
        meta['sampled_clip_center'] = 0.5 * (float(meta['sampled_clip_start']) + float(meta['sampled_clip_end']))
    return item


def cyclic_indices(size: int, count: int, offset: int, seed: int) -> list[int]:
    """Shuffled passes without replacement, continuing across epoch boundaries."""
    if count == 0:
        return []
    if size <= 0 or count < 0 or offset < 0:
        raise ValueError('invalid cyclic sampling size/count/offset')
    result = []
    while len(result) < count:
        cycle, position = divmod(offset, size)
        order = list(range(size))
        random.Random(seed + cycle * 1000003).shuffle(order)
        take = min(size - position, count - len(result))
        result.extend(order[position:position + take])
        offset += take
    return result


class DensePairBatchSampler:
    """Equal DDP budgets; stream-homogeneous batches and adjacent pair halves.

    A unit contains two samples: two microbatches for batch_size=1, or one
    batch for batch_size=2. All ranks use the same stream schedule. Pair sides
    alternate by epoch only for batch_size=1 (batch_size=2 is positive first).
    """
    def __init__(self, natural_count: int, pairs, *, batches_per_rank: int,
                 natural_fraction: float = 0.5, batch_size: int = 1,
                 rank: int = 0, world_size: int = 1, seed: int = 42,
                 on_epoch: Callable[[], None] | None = None, report: bool = False):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError('invalid rank/world_size')
        if batch_size not in (1, 2) or batches_per_rank <= 0:
            raise ValueError('positive batch budget and batch_size 1 or 2 required')
        if batches_per_rank * batch_size % 2:
            raise ValueError('batch_size=1 requires an even batches_per_rank budget')
        if not math.isfinite(natural_fraction) or not 0 <= natural_fraction <= 1 or natural_count < 0:
            raise ValueError('invalid natural count/fraction')
        self.natural_count, self.pairs = int(natural_count), list(pairs)
        if any(len(pair) != 2 or pair[0] == pair[1] or min(pair) < natural_count for pair in self.pairs):
            raise ValueError('pair indices must reference distinct aliases after the natural pool')
        self.batch_size, self.batches_per_rank = int(batch_size), int(batches_per_rank)
        self.rank, self.world_size, self.seed = int(rank), int(world_size), int(seed)
        self.epoch, self.on_epoch = 0, on_epoch
        self.report = bool(report)
        self.units = self.batches_per_rank * self.batch_size // 2
        fraction = float(natural_fraction)
        if not self.pairs and self.natural_count:
            fraction = 1.0  # Videos without eligible reviewed pairs still train.
        self.natural_units = int(math.floor(self.units * fraction + 0.5))
        self.pair_units = self.units - self.natural_units
        if self.natural_units and not self.natural_count:
            raise ValueError('natural sampling requested with an empty pool')
        if self.pair_units and not self.pairs:
            raise ValueError('pair sampling requested with an empty pool')
        if natural_fraction > 0 and self.natural_count and not self.natural_units:
            raise ValueError('batch budget too small for natural sampling')
        self.last_audit = {}

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError('epoch must be nonnegative')
        self.epoch = int(epoch)
        if self.on_epoch is not None:
            self.on_epoch()

    def __len__(self):
        return self.batches_per_rank

    def __iter__(self) -> Iterator[list[int]]:
        # The production trainer starts at epoch 1; epoch 0 previews that pass.
        pass_index = max(self.epoch - 1, 0)
        natural_total = self.natural_units * 2 * self.world_size
        pair_total = self.pair_units * self.world_size
        natural = cyclic_indices(self.natural_count, natural_total,
                                 pass_index * natural_total, self.seed)
        pair_order = cyclic_indices(len(self.pairs), pair_total,
                                    pass_index * pair_total, self.seed + 7919)
        schedule = ['natural'] * self.natural_units + ['pair'] * self.pair_units
        random.Random(self.seed + self.epoch + 104729).shuffle(schedule)
        self.last_audit = dict(
            epoch=self.epoch, batches_per_rank=self.batches_per_rank,
            natural_pool=self.natural_count, reviewed_pairs=len(self.pairs),
            natural_samples_per_rank=self.natural_units * 2,
            pair_samples_per_rank=self.pair_units * 2,
            unique_natural_global=len(set(natural)),
            repeated_natural_global=len(natural) - len(set(natural)),
            unique_pairs_global=len(set(pair_order)),
            epochs_per_natural_pass=(math.ceil(self.natural_count / natural_total) if natural_total else None),
        )
        if self.report:
            print('object_motion_sampling ' + json.dumps(self.last_audit, sort_keys=True), flush=True)
        natural_cursor = pair_cursor = 0
        for stream in schedule:
            if stream == 'natural':
                offset = (natural_cursor * self.world_size + self.rank) * 2
                current = natural[offset:offset + 2]
                natural_cursor += 1
            else:
                index = pair_order[pair_cursor * self.world_size + self.rank]
                current = list(self.pairs[index])
                pair_cursor += 1
                if self.batch_size == 1 and self.epoch % 2:
                    current.reverse()
            if self.batch_size == 1:
                yield [current[0]]
                yield [current[1]]
            else:
                yield current
