"""Online-simulation E1.3 launcher with explicit paired-window semantics.

One training unit contains a 3--7 s central view and an overlapping edge view
of the same annotated event. The sampler keeps this pair in one local batch and
adds trusted background from a different video. Raw accepted context spans and
reviewed-rejected spans are treated as ignore regions rather than negatives.
Model and validation remain in :mod:`train_football_events`.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Sampler

import train_football_events as base

ConfigDict = base.ConfigDict


def build_online_eval_records(
    records: Sequence[base.LongVideoRecord],
    events_by_video: dict[tuple[str, str], list[base.FootballEvent]],
    cfg: Any,
    *,
    online_cfg: Any | None = None,
    sample_prefix: str = "online_eval",
) -> list[base.LongVideoRecord]:
    """Expand videos into the exact fixed-stride online scan grid."""
    if online_cfg is None:
        online_cfg = cfg.get("eval", ConfigDict()).get(
            "external_audit", ConfigDict()
        ).get("online_mode", ConfigDict())
    sample_prefix = str(sample_prefix).strip() or "online_eval"
    clip_sec = float(cfg.video.get("clip_duration", 10.0))
    stride_sec = float(online_cfg.get("window_stride_sec", 5.0))
    if stride_sec <= 0:
        raise ValueError("online evaluation window_stride_sec must be > 0")
    representatives: dict[tuple[str, str], base.LongVideoRecord] = {}
    for row in records:
        key = (row.source, row.video_id)
        previous = representatives.get(key)
        if previous is None or sum(row.label_mask) > sum(previous.label_mask):
            representatives[key] = row
    result: list[base.LongVideoRecord] = []
    for key, representative in sorted(representatives.items()):
        events = [
            event for event in events_by_video.get(key, ()) if not event.is_ignored
        ]
        max_start = max(float(representative.video_duration) - clip_sec, 0.0)
        starts = []
        index = 0
        while index * stride_sec <= max_start + 1e-6:
            starts.append(min(index * stride_sec, max_start))
            index += 1
        if not starts or abs(starts[-1] - max_start) > 1e-6:
            starts.append(max_start)
        for window_index, start in enumerate(starts):
            end = start + clip_sec
            labels = base.labels_for_window(events, start, end)
            anchors = tuple(
                tuple(
                    float(event.anchor_time)
                    for event in events
                    if float(event.labels[label_index]) > 0
                    and start <= event.anchor_time < end
                )
                for label_index in range(len(base.LABELS))
            )
            focus = min(
                (event for event in events if start <= event.anchor_time < end),
                key=lambda event: abs(event.anchor_time - (start + end) * 0.5),
                default=None,
            )
            result.append(
                replace(
                    representative,
                    sample_id=f"{sample_prefix}_{representative.video_id}_w{window_index:06d}",
                    anchor_time=(focus.anchor_time if focus else -1.0),
                    base_clip_start=start,
                    base_clip_end=end,
                    is_negative=not any(float(value) > 0 for value in labels),
                    labels=labels,
                    focus_labels=(focus.labels if focus else ()),
                    online_chunk_id=f"{sample_prefix}_{representative.video_id}",
                    online_chunk_mode="evaluation",
                    online_window_index=window_index,
                    online_negative_kind="none",
                    online_positive_kind="none",
                    online_clean_negative_mask=(),
                    online_label_loss_weights=(),
                    online_gt_anchors=anchors,
                )
            )
    print(
        f"{sample_prefix} videos={len(representatives)} windows={len(result)} "
        f"clip_sec={clip_sec:g} stride_sec={stride_sec:g}",
        flush=True,
    )
    return result


_original_load = base.load_long_video_records
_original_sample_window = base.FootballLongVideoDataset._sample_window
_original_make_loader = base.make_loader
_original_collate = base.football_collate
_train_base_records: list[base.LongVideoRecord] | None = None
_train_events: dict[tuple[str, str], list[base.FootballEvent]] | None = None
_train_cfg: Any | None = None


def online_load(cfg: Any, split: str):
    global _train_base_records, _train_events, _train_cfg
    records, events = _original_load(cfg, split)
    sim = cfg.data.long_video.get("online_simulation", ConfigDict())
    if split == "train" and bool(sim.get("enabled", False)):
        _train_base_records = list(records)
        _train_events = events
        _train_cfg = cfg
        records = build_online_records_e13(records, events, cfg, epoch=0)
    eval_cfg = cfg.get("eval", ConfigDict())
    online_val_cfg = eval_cfg.get("online_validation", ConfigDict())
    online_val_enabled = (
        split == "val" and bool(online_val_cfg.get("enabled", False))
    )
    if online_val_enabled:
        records = build_online_eval_records(
            records,
            events,
            cfg,
            online_cfg=online_val_cfg,
            sample_prefix="online_val",
        )
    external_cfg = cfg.get("eval", ConfigDict()).get(
        "external_audit", ConfigDict()
    )
    online_eval_cfg = external_cfg.get("online_mode", ConfigDict())
    if (
        split == str(external_cfg.get("split", "external"))
        and bool(external_cfg.get("enabled", False))
        and bool(online_eval_cfg.get("enabled", False))
        and not online_val_enabled
    ):
        records = build_online_eval_records(records, events, cfg)
    return records, events


def exact_online_window(self, record, events):
    if record.online_chunk_mode in {
        "paired_event",
        "clean_background",
        "evaluation",
    } or record.sample_id.startswith(("online_eval_", "online_val_")):
        return float(record.base_clip_start), float(record.base_clip_end)
    return _original_sample_window(self, record, events)


def online_collate(items):
    """Apply independent E1.2 clip/frame supervision weights."""
    batch = _original_collate(items)
    raw_clip = [
        tuple(item["meta"].get("online_clip_loss_weights", ()) or ())
        for item in items
    ]
    raw_frame = [
        tuple(item["meta"].get("online_frame_loss_weights", ()) or ())
        for item in items
    ]
    if not any(raw_clip) and not any(raw_frame):
        return batch
    expected = len(base.LABELS)
    if not all(len(weights) == expected for weights in raw_clip):
        raise ValueError("online_clip_loss_weights must have one value per class")
    if not all(len(weights) == expected for weights in raw_frame):
        raise ValueError("online_frame_loss_weights must have one value per class")
    clip_weights = torch.tensor(raw_clip, dtype=batch["label_masks"].dtype)
    frame_weights = torch.tensor(raw_frame, dtype=batch["label_masks"].dtype)
    batch["label_masks"] = batch["label_masks"] * clip_weights
    for key in (
        "frame_target_masks",
        "local_frame_target_masks",
        "highres_pool_target_masks",
    ):
        value = batch.get(key)
        if torch.is_tensor(value):
            batch[key] = value * frame_weights.unsqueeze(1).to(value.dtype)
    return batch


def online_make_loader(dataset, cfg, *, is_train, batch_size=None, distributed=False):
    sim = cfg.data.long_video.get("online_simulation", ConfigDict())
    if not (is_train and bool(sim.get("enabled", False))):
        return _original_make_loader(
            dataset, cfg, is_train=is_train, batch_size=batch_size, distributed=distributed
        )
    local_batch = batch_size or int(cfg.train.batch_size)
    replicas = base.dist.get_world_size() if distributed else 1
    rank = base.dist.get_rank() if distributed else 0
    sampler = OnlinePairDistributedSampler(
        dataset,
        replicas=replicas,
        rank=rank,
        seed=int(cfg.get("seed", 42)),
        drop_last=bool(cfg.train.get("drop_last", True)),
        batch_size=local_batch,
    )
    kwargs = dict(
        dataset=dataset,
        batch_size=local_batch,
        sampler=sampler,
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=bool(cfg.train.get("drop_last", True)),
        collate_fn=online_collate,
        worker_init_fn=base.football_worker_init,
        persistent_workers=False,
        prefetch_factor=int(cfg.data.get("prefetch_factor", 2)),
    )
    if kwargs["num_workers"] == 0:
        kwargs.pop("worker_init_fn")
        kwargs.pop("persistent_workers")
        kwargs.pop("prefetch_factor")
    if rank == 0:
        print(
            f"train_sampler mode=online_simulation_e13_paired pairs={len(sampler.pairs)} "
            f"clean={len(sampler.clean)} samples_per_rank={len(sampler)}",
            flush=True,
        )
    return DataLoader(**kwargs)


def install_e13_hooks() -> None:
    """Install launcher hooks only for an actual E1.3 process."""
    base.load_long_video_records = online_load
    base.FootballLongVideoDataset._sample_window = exact_online_window
    base.make_loader = online_make_loader


def _support_interval(event: base.FootballEvent, min_duration: float) -> tuple[float, float]:
    """Prefer valid raw context spans; otherwise use the reviewed anchor."""
    if event.context_start_time is not None and event.context_end_time is not None:
        lo, hi = sorted((float(event.context_start_time), float(event.context_end_time)))
        if hi - lo >= max(float(min_duration), 0.0):
            return lo, hi
    anchor = float(event.anchor_time)
    return anchor, anchor


def _interval_gap(start: float, end: float, span_start: float, span_end: float) -> float:
    if start <= span_end and end >= span_start:
        return 0.0
    return span_start - end if end < span_start else start - span_end


def e13_window_weights(
    events: Sequence[base.FootballEvent], labels: Sequence[float],
    start: float, end: float, label_index: int, *, mode: str, role: str,
    min_context_duration_sec: float, rejected_ignore_margin_sec: float,
    near_event_ignore_sec: float, near_event_max_sec: float,
    near_event_negative_weight: float, edge_clip_weight: float,
    edge_frame_weight: float,
) -> tuple[float, float, str]:
    """Compute weights without teaching accepted/rejected context as negative."""
    if float(labels[label_index]) > 0:
        if role == "edge":
            return float(edge_clip_weight), float(edge_frame_weight), "positive"
        return 1.0, 1.0, "positive"
    accepted: list[tuple[float, float]] = []
    rejected: list[tuple[float, float]] = []
    for event in events:
        if label_index >= len(event.labels) or float(event.labels[label_index]) <= 0:
            continue
        target = rejected if event.is_ignored else accepted
        target.append(_support_interval(event, min_context_duration_sec))
    if any(_interval_gap(start, end, *span) == 0 for span in accepted):
        return 0.0, 0.0, "accepted_context_ignore"
    margin = max(float(rejected_ignore_margin_sec), 0.0)
    if any(
        _interval_gap(start, end, left - margin, right + margin) == 0
        for left, right in rejected
    ):
        return 0.0, 0.0, "rejected_context_ignore"
    if mode == "paired_event" and accepted:
        distance = min(_interval_gap(start, end, *span) for span in accepted)
        if distance <= near_event_ignore_sec:
            return 0.0, 0.0, "near_event_ignore"
        if distance <= near_event_max_sec:
            weight = float(near_event_negative_weight)
            return weight, weight, "near_event_negative"
    return 1.0, 1.0, "negative"


def _e13_record(
    representative: base.LongVideoRecord,
    events: Sequence[base.FootballEvent], *, start: float, clip_sec: float,
    mode: str, sample_suffix: str, cfg: Any, role: str = "",
    pair_id: str = "", group_id: str = "",
    pair_event: base.FootballEvent | None = None,
) -> base.LongVideoRecord:
    sim = cfg.data.long_video.get("online_simulation", ConfigDict())
    raw_cfg = cfg.get("raw_set_piece_supervision", ConfigDict())
    end = float(start) + float(clip_sec)
    labels = base.labels_for_window(events, start, end)
    clip_weights: list[float] = []
    frame_weights: list[float] = []
    reasons: list[str] = []
    for index in range(len(base.LABELS)):
        clip_weight, frame_weight, reason = e13_window_weights(
            events, labels, start, end, index, mode=mode, role=role,
            min_context_duration_sec=float(raw_cfg.get("context_span_min_duration_sec", 0.5)),
            rejected_ignore_margin_sec=float(
                raw_cfg.get("rejected_ignore_margin_sec", 5.0)
                if bool(raw_cfg.get("rejected_ignore_enabled", True)) else 0.0
            ),
            near_event_ignore_sec=float(sim.get("near_event_ignore_sec", 2.0)),
            near_event_max_sec=float(sim.get("near_event_max_sec", 8.0)),
            near_event_negative_weight=float(sim.get("near_event_negative_weight", 0.5)),
            edge_clip_weight=float(sim.get("edge_clip_loss_weight", 0.35)),
            edge_frame_weight=float(sim.get("edge_frame_loss_weight", 0.5)),
        )
        clip_weights.append(clip_weight)
        frame_weights.append(frame_weight)
        reasons.append(reason)
    positive = any(float(value) > 0 for value in labels)
    clean = mode == "clean_background" and not positive
    clean_mask = tuple(
        float(clean and representative.label_mask[i] > 0 and clip_weights[i] > 0)
        for i in range(len(base.LABELS))
    )
    focus = pair_event or min(
        (event for event in events if not event.is_ignored and start <= event.anchor_time < end),
        key=lambda event: abs(event.anchor_time - 0.5 * (start + end)),
        default=None,
    )
    negative_kind = (
        "clean_background" if clean else
        "context_ignore" if not positive and any("ignore" in reason for reason in reasons) else
        "near_event_context" if not positive and mode == "paired_event" else "none"
    )
    return replace(
        representative,
        sample_id=f"e13_{sample_suffix}",
        anchor_time=float(focus.anchor_time) if focus else -1.0,
        base_clip_start=float(start), base_clip_end=end,
        is_negative=not positive, labels=labels,
        focus_labels=focus.labels if focus else (),
        save_cohort_kind="none", save_cohort_weight=0.0,
        online_chunk_id=pair_id or group_id or f"e13_{sample_suffix}",
        online_chunk_mode=mode, online_window_index=-1,
        online_negative_kind=negative_kind,
        online_positive_kind=role if positive and role else "none",
        online_clean_negative_mask=clean_mask,
        online_clip_loss_weights=tuple(clip_weights),
        online_frame_loss_weights=tuple(frame_weights),
        online_label_loss_weights=tuple(clip_weights),
        online_pair_id=pair_id, online_pair_role=role,
        online_pair_class_mask=(
            tuple(float(value) for value in pair_event.labels)
            if pair_id and pair_event is not None else ()
        ),
    )


def build_online_records_e13(
    records: Sequence[base.LongVideoRecord],
    events_by_video: dict[tuple[str, str], list[base.FootballEvent]],
    cfg: Any, *, epoch: int = 0,
) -> list[base.LongVideoRecord]:
    """Build central/edge event pairs and raw-context-safe backgrounds."""
    sim = cfg.data.long_video.get("online_simulation", ConfigDict())
    clip_sec = float(cfg.video.get("clip_duration", 10.0))
    central_min = float(sim.get("primary_positive_min_sec", 3.0))
    central_max = float(sim.get("primary_positive_max_sec", 7.0))
    edge_min = float(sim.get("edge_position_min_sec", 0.5))
    edge_max = float(sim.get("edge_position_max_sec", 2.0))
    min_shift = float(sim.get("pair_min_shift_sec", 2.0))
    if not 0 <= edge_min < edge_max < central_min < central_max < clip_sec:
        raise ValueError("E1.3 central/edge position ranges are invalid")
    representatives: dict[tuple[str, str], base.LongVideoRecord] = {}
    for row in records:
        key = row.source, row.video_id
        if key not in representatives or sum(row.label_mask) > sum(representatives[key].label_mask):
            representatives[key] = row
    keys = [key for key, row in representatives.items() if row.video_duration >= clip_sec]
    accepted = {
        key: [event for event in events_by_video.get(key, ()) if not event.is_ignored]
        for key in keys
    }
    pools: dict[str, dict[tuple[str, str], list[base.FootballEvent]]] = {
        label: {} for label in base.LABELS
    }
    for key, events in accepted.items():
        for event in events:
            pair_ranges = _pair_start_ranges(
                event, representatives[key].video_duration, clip_sec,
                central_min, central_max, edge_min, edge_max,
            )
            if pair_ranges is None or not _pair_ranges_support_shift(pair_ranges, min_shift):
                continue
            for label, value in zip(base.LABELS, event.labels):
                if value > 0:
                    pools[label].setdefault(key, []).append(event)
    active_labels = [label for label in base.LABELS if pools[label]]
    if not keys or not active_labels:
        raise RuntimeError("E1.3 has no eligible videos/events")
    legacy_windows = max(
        int(math.floor((float(sim.get("chunk_duration_sec", 40.0)) - clip_sec)
                       / float(sim.get("window_stride_sec", 5.0)))) + 1, 1
    )
    chunks = int(sim.get("chunks_per_epoch", 0) or 0)
    chunks = chunks or max(int(math.ceil(len(records) / legacy_windows)), 1)
    pair_count = int(sim.get("event_pairs_per_epoch", 0) or 0)
    pair_count = pair_count or max(int(round(chunks * float(sim.get("event_chunk_fraction", 0.5)))), 1)
    clean_count = max(int(sim.get("clean_background_windows_per_epoch", 0) or 0), pair_count)
    attempts = max(int(sim.get("background_max_attempts", 256)), 1)
    raw_cfg = cfg.get("raw_set_piece_supervision", ConfigDict())
    min_span = float(raw_cfg.get("context_span_min_duration_sec", 0.5))
    rejected_margin = float(raw_cfg.get("rejected_ignore_margin_sec", 5.0))
    near_ignore = float(sim.get("near_event_ignore_sec", 2.0))
    near_gap_min = float(sim.get("near_event_anchor_gap_min_sec", 5.0))
    near_gap_max = float(sim.get("near_event_anchor_gap_max_sec", 15.0))
    if not 0 <= near_ignore < near_gap_min <= near_gap_max:
        raise ValueError("E1.3 near-event gap requires ignore < min <= max")
    rng = random.Random(
        int(cfg.get("seed", 42)) + base.stable_int("online_simulation_e13")
        + int(epoch) * 1_000_003
    )
    result: list[base.LongVideoRecord] = []
    used_events: set[tuple[str, str, str]] = set()
    for pair_index in range(pair_count):
        label = active_labels[pair_index % len(active_labels)]
        videos = list(pools[label])
        available = [
            key for key in videos
            if any((key[0], key[1], str(e.event_id)) not in used_events for e in pools[label][key])
        ]
        key = rng.choice(available or videos)
        choices = [
            e for e in pools[label][key]
            if (key[0], key[1], str(e.event_id)) not in used_events
        ] or list(pools[label][key])
        event = rng.choice(choices)
        used_events.add((key[0], key[1], str(event.event_id)))
        representative = representatives[key]
        pair_starts = _sample_pair_starts(
            event, representative.video_duration, clip_sec,
            central_min, central_max, edge_min, edge_max, min_shift, rng,
        )
        if pair_starts is None:
            raise RuntimeError(f"Cannot construct E1.3 pair for {key}/{event.event_id}")
        central_start, edge_start = pair_starts
        pair_id = f"e13pair:{key[0]}:{key[1]}:{event.event_id}:{pair_index:06d}"
        all_events = list(events_by_video.get(key, ()))
        result.append(_e13_record(
            representative, all_events, start=central_start, clip_sec=clip_sec,
            mode="paired_event", sample_suffix=f"pair_{pair_index:06d}_central",
            cfg=cfg, role="central", pair_id=pair_id, pair_event=event,
        ))
        result.append(_e13_record(
            representative, all_events, start=edge_start, clip_sec=clip_sec,
            mode="paired_event", sample_suffix=f"pair_{pair_index:06d}_edge",
            cfg=cfg, role="edge", pair_id=pair_id, pair_event=event,
        ))

        near_start = _sample_near_event_start(
            event, all_events,
            video_duration=representative.video_duration,
            clip_sec=clip_sec,
            min_anchor_gap_sec=near_gap_min,
            max_anchor_gap_sec=near_gap_max,
            min_context_duration_sec=min_span,
            accepted_margin_sec=near_ignore,
            rejected_margin_sec=rejected_margin,
            attempts=attempts,
            rng=rng,
        )
        if near_start is not None:
            linked_negative = _e13_record(
                representative, all_events, start=near_start, clip_sec=clip_sec,
                mode="paired_event", group_id=pair_id,
                sample_suffix=f"pair_{pair_index:06d}_near", cfg=cfg,
            )
            if not linked_negative.is_negative:
                raise RuntimeError(f"E1.3 near row unexpectedly positive: {pair_id}")
        else:
            fallback = None
            other_keys = [candidate for candidate in keys if candidate != key]
            for _ in range(attempts):
                fallback_key = rng.choice(other_keys)
                fallback_row = representatives[fallback_key]
                max_start = max(fallback_row.video_duration - clip_sec, 0.0)
                fallback_start = rng.uniform(0.0, max_start) if max_start else 0.0
                if not _window_conflicts_support(
                    events_by_video.get(fallback_key, ()), fallback_start,
                    fallback_start + clip_sec,
                    min_context_duration_sec=min_span,
                    rejected_margin_sec=rejected_margin,
                ):
                    fallback = fallback_key, fallback_start
                    break
            if fallback is None:
                raise RuntimeError(f"E1.3 cannot build near fallback for {pair_id}")
            fallback_key, fallback_start = fallback
            linked_negative = _e13_record(
                representatives[fallback_key], events_by_video.get(fallback_key, ()),
                start=fallback_start, clip_sec=clip_sec, mode="clean_background",
                group_id=pair_id,
                sample_suffix=f"pair_{pair_index:06d}_near_fallback", cfg=cfg,
            )
            linked_negative = replace(
                linked_negative, online_negative_kind="near_event_fallback_clean"
            )
        result.append(linked_negative)
    for background_index in range(clean_count):
        chosen = None
        for _ in range(attempts):
            key = rng.choice(keys)
            row = representatives[key]
            max_start = max(row.video_duration - clip_sec, 0.0)
            start = rng.uniform(0.0, max_start) if max_start else 0.0
            end = start + clip_sec
            unsafe = any(
                _interval_gap(
                    start, end,
                    _support_interval(event, min_span)[0] - (rejected_margin if event.is_ignored else 0.0),
                    _support_interval(event, min_span)[1] + (rejected_margin if event.is_ignored else 0.0),
                ) == 0
                for event in events_by_video.get(key, ())
            )
            if not unsafe:
                chosen = key, start
                break
        if chosen is None:
            raise RuntimeError("E1.3 failed to find a raw-context-safe background")
        key, start = chosen
        result.append(_e13_record(
            representatives[key], list(events_by_video.get(key, ())),
            start=start, clip_sec=clip_sec, mode="clean_background",
            sample_suffix=f"clean_{background_index:06d}", cfg=cfg,
        ))
    result = base.apply_dataset_level_save_cohort_weights(result, cfg, "train")
    near_count = sum(row.online_negative_kind == "near_event_context" for row in result)
    near_fallback_count = sum(
        row.online_negative_kind == "near_event_fallback_clean" for row in result
    )
    print(
        f"online_simulation_e13 epoch_manifest={epoch} pairs={pair_count} "
        f"near={near_count} near_fallback={near_fallback_count} "
        f"clean={clean_count} records={len(result)}",
        flush=True,
    )
    return result


class OnlinePairDistributedSampler(Sampler[int]):
    """Keep each pair in one batch and add clean data from another video."""
    def __init__(self, dataset: Any, *, replicas: int, rank: int, seed: int,
                 drop_last: bool, batch_size: int):
        self.dataset = dataset
        self.replicas, self.rank = int(replicas), int(rank)
        self.seed, self.drop_last = int(seed), bool(drop_last)
        self.batch_size = int(batch_size)
        if self.batch_size < 4 or self.batch_size % 4:
            raise ValueError("E1.3 per-rank batch_size must be a multiple of 4")
        self.epoch = 0
        self._refresh_groups()
        self.pairs_per_batch = self.batch_size // 4
        self.num_batches = len(self.pairs) // self.replicas // self.pairs_per_batch
        if self.num_batches <= 0:
            raise ValueError("E1.3 has too few event pairs for this DDP world size")
        self.num_samples = self.num_batches * self.batch_size

    def _refresh_groups(self) -> None:
        pairs: dict[str, dict[str, int]] = {}
        linked: dict[str, list[int]] = {}
        self.clean, self.extras = [], []
        for index, row in enumerate(self.dataset.records):
            if row.online_pair_id:
                roles = pairs.setdefault(row.online_pair_id, {})
                if row.online_pair_role in roles:
                    raise ValueError(f"Duplicate E1.3 role for {row.online_pair_id}")
                roles[row.online_pair_role] = index
            elif row.online_negative_kind in {"near_event_context", "near_event_fallback_clean"}:
                linked.setdefault(row.online_chunk_id, []).append(index)
            elif row.online_negative_kind == "clean_background":
                self.clean.append(index)
            else:
                self.extras.append(index)
        malformed = {key: sorted(value) for key, value in pairs.items()
                     if set(value) != {"central", "edge"}}
        if malformed:
            raise ValueError(f"Malformed E1.3 pairs: {malformed}")
        malformed_linked = {
            key: len(linked.get(key, ())) for key in pairs
            if len(linked.get(key, ())) != 1
        }
        if malformed_linked:
            raise ValueError(f"E1.3 requires one linked near row per pair: {malformed_linked}")
        self.pairs = [
            (key, value["central"], value["edge"], linked[key][0])
            for key, value in sorted(pairs.items())
        ]
        if not self.pairs or not self.clean:
            raise ValueError("E1.3 requires paired positives and clean backgrounds")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if _train_base_records is None or _train_events is None or _train_cfg is None:
            raise RuntimeError("E1.3 dynamic manifest state was not initialized")
        self.dataset.records = build_online_records_e13(
            _train_base_records, _train_events, _train_cfg, epoch=self.epoch
        )
        self._refresh_groups()

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        pairs, clean = list(self.pairs), list(self.clean)
        rng.shuffle(pairs)
        rng.shuffle(clean)
        local_pairs = pairs[self.rank :: self.replicas]
        local_clean = clean[self.rank :: self.replicas]
        if not local_clean:
            local_clean = clean
        batches: list[list[int]] = []
        cursor = 0
        for offset in range(0, len(local_pairs), self.pairs_per_batch):
            group = local_pairs[offset:offset + self.pairs_per_batch]
            if len(group) < self.pairs_per_batch:
                break
            batch: list[int] = []
            videos: list[str] = []
            for _pair_id, central, edge, near in group:
                batch.extend((central, edge, near))
                videos.append(self.dataset.records[central].video_id)
            for video_id in videos:
                selected = next(
                    (local_clean[(cursor + probe) % len(local_clean)]
                     for probe in range(len(local_clean))
                     if self.dataset.records[local_clean[(cursor + probe) % len(local_clean)]].video_id != video_id),
                    None,
                )
                if selected is None:
                    selected = next(
                        (candidate for candidate in clean
                         if self.dataset.records[candidate].video_id != video_id),
                        None,
                    )
                if selected is None:
                    raise RuntimeError(f"No cross-video clean background for {video_id}")
                cursor = (local_clean.index(selected) + 1) % len(local_clean)
                batch.append(selected)
            if len(batch) != self.batch_size:
                raise RuntimeError(f"E1.3 batch assembly produced {len(batch)} rows")
            batches.append(batch)
        rng.shuffle(batches)
        ordered = [index for batch in batches for index in batch]
        if len(ordered) < self.num_samples:
            raise RuntimeError(f"E1.3 rank={self.rank} produced too few samples")
        return iter(ordered[:self.num_samples])


def _valid_start_range(low: float, high: float, max_start: float) -> tuple[float, float] | None:
    low, high = max(float(low), 0.0), min(float(high), float(max_start))
    return (low, high) if low <= high else None


def _pair_start_ranges(
    event: base.FootballEvent, video_duration: float, clip_sec: float,
    central_min: float, central_max: float, edge_min: float, edge_max: float,
) -> tuple[tuple[float, float], list[tuple[float, float]]] | None:
    max_start = max(float(video_duration) - clip_sec, 0.0)
    anchor = float(event.anchor_time)
    central = _valid_start_range(anchor - central_max, anchor - central_min, max_start)
    edge_ranges = [
        _valid_start_range(anchor - edge_max, anchor - edge_min, max_start),
        _valid_start_range(
            anchor - (clip_sec - edge_min),
            anchor - (clip_sec - edge_max),
            max_start,
        ),
    ]
    valid_edges = [value for value in edge_ranges if value is not None]
    if central is None or not valid_edges:
        return None
    return central, valid_edges


def _pair_ranges_support_shift(
    ranges: tuple[tuple[float, float], list[tuple[float, float]]], min_shift: float
) -> bool:
    central, edges = ranges
    return max(
        abs(c_value - e_value)
        for c_value in central
        for edge in edges
        for e_value in edge
    ) >= float(min_shift)


def _sample_pair_starts(
    event: base.FootballEvent, video_duration: float, clip_sec: float,
    central_min: float, central_max: float, edge_min: float, edge_max: float,
    min_shift: float, rng: random.Random,
) -> tuple[float, float] | None:
    ranges = _pair_start_ranges(
        event, video_duration, clip_sec,
        central_min, central_max, edge_min, edge_max,
    )
    if ranges is None:
        return None
    central_range, edge_ranges = ranges
    central = rng.uniform(*central_range)
    candidates = [
        candidate
        for value in edge_ranges
        for candidate in (rng.uniform(*value), value[0], value[1])
    ]
    edge = max(candidates, key=lambda value: abs(value - central))
    if abs(edge - central) >= min_shift:
        return central, edge
    all_pairs = [
        (c_value, e_value)
        for c_value in central_range for value in edge_ranges for e_value in value
    ]
    central, edge = max(all_pairs, key=lambda pair: abs(pair[0] - pair[1]))
    return (central, edge) if abs(edge - central) >= min_shift else None


def _window_conflicts_support(
    events: Sequence[base.FootballEvent], start: float, end: float, *,
    min_context_duration_sec: float, rejected_margin_sec: float,
    accepted_margin_sec: float = 0.0,
) -> bool:
    for event in events:
        left, right = _support_interval(event, min_context_duration_sec)
        margin = rejected_margin_sec if event.is_ignored else accepted_margin_sec
        if _interval_gap(start, end, left - margin, right + margin) == 0:
            return True
    return False


def _sample_near_event_start(
    event: base.FootballEvent,
    all_events: Sequence[base.FootballEvent],
    *,
    video_duration: float,
    clip_sec: float,
    min_anchor_gap_sec: float,
    max_anchor_gap_sec: float,
    min_context_duration_sec: float,
    accepted_margin_sec: float,
    rejected_margin_sec: float,
    attempts: int,
    rng: random.Random,
) -> float | None:
    """Sample a near-event negative without crossing any known support span."""
    max_start = max(float(video_duration) - clip_sec, 0.0)
    if max_start <= 0:
        return None
    sides = [-1, 1]
    for _ in range(max(int(attempts), 1)):
        rng.shuffle(sides)
        gap = rng.uniform(min_anchor_gap_sec, max_anchor_gap_sec)
        for side in sides:
            # Before: window ends ``gap`` before anchor. After: window starts
            # ``gap`` after anchor. This realizes the requested 5--15 s band.
            start = (
                float(event.anchor_time) - gap - clip_sec
                if side < 0 else float(event.anchor_time) + gap
            )
            if start < 0 or start > max_start:
                continue
            end = start + clip_sec
            if _window_conflicts_support(
                all_events, start, end,
                min_context_duration_sec=min_context_duration_sec,
                accepted_margin_sec=accepted_margin_sec,
                rejected_margin_sec=rejected_margin_sec,
            ):
                continue
            if any(base.labels_for_window(all_events, start, end)):
                continue
            return float(start)
    return None



if __name__ == "__main__":
    install_e13_hooks()
    base.main()
