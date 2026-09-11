"""Online-grid-inspired paired-window training, E1.6.

Round-1 supervision fix on top of E1.3.  Every sampled event now trains the
exact windows the fixed-stride online scan will produce for it:

* one canonical 3--7 s window (stable frame localization supervision),
* BOTH covering grid windows of the eval grid (stride 5 s).  The two grid
  windows place the anchor at ``r`` and ``r + stride`` with
  ``r = anchor mod stride``, which closes the [2,3) / [7,8) position blind
  spot of the E1.3 central/edge sampler.  Grid windows keep a full-weight
  clip label; the frame loss is only down-weighted when the anchor sits
  outside the canonical 3--7 s band.
* the E1.3 near-event negative and clean-background machinery is reused
  unchanged (accepted/rejected context stays ignore, never a hard negative).

Roles stay "central"/"edge" so the existing pair-consistency loss treats the
canonical window as the teacher and each grid window as a student.

Model/validation live in :mod:`train_football_events`; the E1.3 grid
evaluation records and collate logic are reused from
:mod:`train_football_events_online_simulation_e13`.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from typing import Any, Iterator, Sequence

from torch.utils.data import DataLoader, Sampler

import train_football_events as base
import train_football_events_online_simulation_e13 as e13

ConfigDict = base.ConfigDict


def _grid_window_starts(
    anchor: float, video_duration: float, clip_sec: float, stride_sec: float,
) -> list[float]:
    """Starts of the fixed-stride grid windows that cover ``anchor``.

    Mirrors ``build_online_eval_records``: grid starts are ``i * stride``
    clamped to ``[0, duration - clip]``; a window covers the anchor when
    ``start <= anchor < start + clip``.
    """
    max_start = max(float(video_duration) - float(clip_sec), 0.0)
    index = int(math.floor(float(anchor) / stride_sec + 1e-9))
    starts: list[float] = []
    for candidate_index in (index, index - 1):
        start = min(candidate_index * stride_sec, max_start)
        start = max(start, 0.0)
        if start <= float(anchor) < start + float(clip_sec):
            if not any(abs(start - existing) < 1e-6 for existing in starts):
                starts.append(start)
    return starts


def e16_window_weights(
    events: Sequence[base.FootballEvent], labels: Sequence[float],
    start: float, end: float, label_index: int, *, mode: str, role: str,
    pair_anchor: float, cfg: Any,
) -> tuple[float, float, str]:
    """Positive weights by role/position; negatives reuse the E1.3 policy."""
    sim = cfg.data.long_video.get("online_simulation", ConfigDict())
    if float(labels[label_index]) > 0:
        if role == "central":
            return 1.0, 1.0, "positive"
        if role == "edge":
            central_min = float(sim.get("primary_positive_min_sec", 3.0))
            central_max = float(sim.get("primary_positive_max_sec", 7.0))
            position = float(pair_anchor) - float(start)
            if central_min <= position <= central_max:
                return 1.0, 1.0, "positive_grid_canonical"
            clip_weight = float(sim.get("grid_clip_loss_weight", 1.0))
            frame_weight = float(sim.get("grid_frame_loss_weight", 0.5))
            return clip_weight, frame_weight, "positive_grid_boundary"
        return 1.0, 1.0, "positive"
    raw_cfg = cfg.get("raw_set_piece_supervision", ConfigDict())
    return e13.e13_window_weights(
        events, labels, start, end, label_index,
        mode=mode, role="",
        min_context_duration_sec=float(
            raw_cfg.get("context_span_min_duration_sec", 0.5)
        ),
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


def _mask_ambiguous_cross_label_negatives(
    labels: Sequence[float],
    clip_weights: list[float],
    frame_weights: list[float],
    reasons: list[str],
    cfg: Any,
) -> None:
    """Mask configured cross-label negatives without inventing positives."""
    sim = cfg.data.long_video.get("online_simulation", ConfigDict())
    mapping = sim.get("ambiguous_negative_mask_by_positive_label", ConfigDict())
    if not mapping:
        return
    for positive_label, masked_labels in mapping.items():
        if positive_label not in base.LABEL_TO_INDEX:
            raise ValueError(
                "Unknown positive label in "
                f"ambiguous_negative_mask_by_positive_label: {positive_label}"
            )
        positive_index = base.LABEL_TO_INDEX[positive_label]
        if float(labels[positive_index]) <= 0:
            continue
        if isinstance(masked_labels, str):
            masked_labels = [masked_labels]
        for masked_label in masked_labels:
            if masked_label not in base.LABEL_TO_INDEX:
                raise ValueError(
                    "Unknown masked label in "
                    f"ambiguous_negative_mask_by_positive_label: {masked_label}"
                )
            masked_index = base.LABEL_TO_INDEX[masked_label]
            if float(labels[masked_index]) > 0:
                continue
            clip_weights[masked_index] = 0.0
            frame_weights[masked_index] = 0.0
            reasons[masked_index] = (
                f"ambiguous_{positive_label}_masks_{masked_label}_negative"
            )


def _e16_record(
    representative: base.LongVideoRecord,
    events: Sequence[base.FootballEvent], *, start: float, clip_sec: float,
    mode: str, sample_suffix: str, cfg: Any, role: str = "",
    pair_id: str = "", group_id: str = "",
    pair_event: base.FootballEvent | None = None,
) -> base.LongVideoRecord:
    sim = cfg.data.long_video.get("online_simulation", ConfigDict())
    end = float(start) + float(clip_sec)
    labels = base.labels_for_window(events, start, end)
    pair_anchor = float(pair_event.anchor_time) if pair_event else -1.0
    clip_weights: list[float] = []
    frame_weights: list[float] = []
    reasons: list[str] = []
    for index in range(len(base.LABELS)):
        clip_weight, frame_weight, reason = e16_window_weights(
            events, labels, start, end, index, mode=mode, role=role,
            pair_anchor=pair_anchor, cfg=cfg,
        )
        clip_weights.append(clip_weight)
        frame_weights.append(frame_weight)
        reasons.append(reason)
    _mask_ambiguous_cross_label_negatives(
        labels, clip_weights, frame_weights, reasons, cfg
    )
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
        sample_id=f"e16_{sample_suffix}",
        anchor_time=float(focus.anchor_time) if focus else -1.0,
        base_clip_start=float(start), base_clip_end=end,
        is_negative=not positive, labels=labels,
        focus_labels=focus.labels if focus else (),
        save_cohort_kind="none", save_cohort_weight=0.0,
        online_chunk_id=pair_id or group_id or f"e16_{sample_suffix}",
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


def _load_hard_negative_entries(
    sim: Any,
    representatives: dict[tuple[str, str], base.LongVideoRecord],
    events_by_video: dict[tuple[str, str], list[base.FootballEvent]],
    cfg: Any,
    rng: random.Random,
) -> list[base.LongVideoRecord]:
    """Optional E1.7 dense mined hard negatives replacing part of the clean pool.

    The manifest is produced by ``scripts/mine_dense_hard_negatives.py`` from a
    full dense scan of the *train* split.  Entries are replayed as
    ``clean_background`` windows so the E1.3/E1.6 safety weighting decides the
    final per-class loss weights; the sampler treats them as regular clean
    rows.  Sampling is epoch-seeded and identical across DDP ranks.
    """
    manifest_path = str(sim.get("hard_negative_manifest", "") or "")
    target = int(sim.get("hard_negative_windows_per_epoch", 0) or 0)
    if not manifest_path or target <= 0:
        return []
    loss_weight = float(sim.get("hard_negative_loss_weight", 1.0))
    if not 0.0 < loss_weight <= 1.0:
        raise ValueError(f"hard_negative_loss_weight must be in (0, 1], got {loss_weight}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)["entries"]
    clip_sec = float(cfg.video.get("clip_duration", 10.0))
    eligible = []
    for entry in entries:
        key = (str(entry["source"]), str(entry["video_id"]))
        row = representatives.get(key)
        if row is None:
            continue
        max_start = max(float(row.video_duration) - clip_sec, 0.0)
        start = min(max(float(entry["start"]), 0.0), max_start)
        eligible.append((key, start))
    rng.shuffle(eligible)
    result: list[base.LongVideoRecord] = []
    skipped_positive = 0
    for hard_index, (key, start) in enumerate(eligible[:target]):
        record = _e16_record(
            representatives[key], list(events_by_video.get(key, ())),
            start=start, clip_sec=clip_sec, mode="clean_background",
            sample_suffix=f"hard_{hard_index:06d}", cfg=cfg,
        )
        if not record.is_negative:
            skipped_positive += 1
            continue
        if loss_weight < 1.0:
            record = replace(
                record,
                online_clip_loss_weights=tuple(
                    float(value) * loss_weight
                    for value in record.online_clip_loss_weights
                ),
                online_frame_loss_weights=tuple(
                    float(value) * loss_weight
                    for value in record.online_frame_loss_weights
                ),
                online_label_loss_weights=tuple(
                    float(value) * loss_weight
                    for value in record.online_label_loss_weights
                ),
            )
        result.append(record)
    print(
        f"online_simulation_e16 hard_negative_manifest={manifest_path} "
        f"requested={target} used={len(result)} skipped_positive={skipped_positive} "
        f"loss_weight={loss_weight:g}",
        flush=True,
    )
    return result


def build_online_records_e16(
    records: Sequence[base.LongVideoRecord],
    events_by_video: dict[tuple[str, str], list[base.FootballEvent]],
    cfg: Any, *, epoch: int = 0,
) -> list[base.LongVideoRecord]:
    """Build canonical + dual-grid event groups and context-safe backgrounds."""
    sim = cfg.data.long_video.get("online_simulation", ConfigDict())
    clip_sec = float(cfg.video.get("clip_duration", 10.0))
    stride_sec = float(sim.get("window_stride_sec", 5.0))
    if stride_sec <= 0:
        raise ValueError("E1.6 requires a positive window_stride_sec")
    central_min = float(sim.get("primary_positive_min_sec", 3.0))
    central_max = float(sim.get("primary_positive_max_sec", 7.0))
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
        duration = representatives[key].video_duration
        for event in events:
            anchor = float(event.anchor_time)
            grid_starts = _grid_window_starts(anchor, duration, clip_sec, stride_sec)
            central_range = e13._valid_start_range(
                anchor - central_max, anchor - central_min,
                max(duration - clip_sec, 0.0),
            )
            if not grid_starts or central_range is None:
                continue
            for label, value in zip(base.LABELS, event.labels):
                if value > 0:
                    pools[label].setdefault(key, []).append(event)
    active_labels = [label for label in base.LABELS if pools[label]]
    if not keys or not active_labels:
        raise RuntimeError("E1.6 has no eligible videos/events")
    event_sampling_mode = str(
        sim.get("event_sampling_mode", "balanced_budget")
    ).strip().lower()
    if event_sampling_mode not in {"balanced_budget", "cover_all_once"}:
        raise ValueError(
            "E1.6 event_sampling_mode must be balanced_budget or cover_all_once"
        )
    configured_pair_count = int(sim.get("event_pairs_per_epoch", 0) or 0)
    if event_sampling_mode == "balanced_budget" and configured_pair_count <= 0:
        raise ValueError("E1.6 requires an explicit event_pairs_per_epoch")
    attempts = max(int(sim.get("background_max_attempts", 256)), 1)
    raw_cfg = cfg.get("raw_set_piece_supervision", ConfigDict())
    min_span = float(raw_cfg.get("context_span_min_duration_sec", 0.5))
    rejected_margin = float(raw_cfg.get("rejected_ignore_margin_sec", 5.0))
    near_ignore = float(sim.get("near_event_ignore_sec", 2.0))
    near_gap_min = float(sim.get("near_event_anchor_gap_min_sec", 5.0))
    near_gap_max = float(sim.get("near_event_anchor_gap_max_sec", 15.0))
    if not 0 <= near_ignore < near_gap_min <= near_gap_max:
        raise ValueError("E1.6 near-event gap requires ignore < min <= max")
    rng = random.Random(
        int(cfg.get("seed", 42)) + base.stable_int("online_simulation_e16")
        + int(epoch) * 1_000_003
    )
    selected_pairs: list[tuple[tuple[str, str], base.FootballEvent]] | None = None
    eligible_unique_total = 0
    if event_sampling_mode == "cover_all_once":
        # A single event may be visible through more than one label pool. Build
        # from event identity so every eligible GT is included exactly once
        # before its central/grid views are expanded.
        unique_events: dict[
            tuple[str, str, str], tuple[tuple[str, str], base.FootballEvent]
        ] = {}
        for label in active_labels:
            for key, label_events in pools[label].items():
                for event in label_events:
                    event_key = (key[0], key[1], str(event.event_id))
                    unique_events.setdefault(event_key, (key, event))
        selected_pairs = list(unique_events.values())
        rng.shuffle(selected_pairs)
        eligible_unique_total = len(selected_pairs)
        pair_count = eligible_unique_total
        if pair_count <= 0:
            raise RuntimeError("E1.6 cover_all_once found no eligible events")
    else:
        pair_count = configured_pair_count
        eligible_unique_total = len({
            (key[0], key[1], str(event.event_id))
            for label in active_labels
            for key, label_events in pools[label].items()
            for event in label_events
        })
    clean_count = max(
        int(sim.get("clean_background_windows_per_epoch", 0) or 0), pair_count
    )
    result: list[base.LongVideoRecord] = []
    used_events: set[tuple[str, str, str]] = set()
    grid_window_total = 0
    grid_boundary_total = 0
    for pair_index in range(pair_count):
        if selected_pairs is not None:
            key, event = selected_pairs[pair_index]
        else:
            label = active_labels[pair_index % len(active_labels)]
            videos = list(pools[label])
            available = [
                key for key in videos
                if any(
                    (key[0], key[1], str(e.event_id)) not in used_events
                    for e in pools[label][key]
                )
            ]
            key = rng.choice(available or videos)
            choices = [
                e for e in pools[label][key]
                if (key[0], key[1], str(e.event_id)) not in used_events
            ] or list(pools[label][key])
            event = rng.choice(choices)
        used_events.add((key[0], key[1], str(event.event_id)))
        representative = representatives[key]
        duration = representative.video_duration
        anchor = float(event.anchor_time)
        max_start = max(duration - clip_sec, 0.0)
        central_start = rng.uniform(
            max(anchor - central_max, 0.0), min(anchor - central_min, max_start)
        )
        pair_id = f"e16pair:{key[0]}:{key[1]}:{event.event_id}:{pair_index:06d}"
        all_events = list(events_by_video.get(key, ()))
        result.append(_e16_record(
            representative, all_events, start=central_start, clip_sec=clip_sec,
            mode="paired_event", sample_suffix=f"pair_{pair_index:06d}_central",
            cfg=cfg, role="central", pair_id=pair_id, pair_event=event,
        ))
        for grid_slot, grid_start in enumerate(
            _grid_window_starts(anchor, duration, clip_sec, stride_sec)
        ):
            position = anchor - grid_start
            grid_window_total += 1
            if not (central_min <= position <= central_max):
                grid_boundary_total += 1
            result.append(_e16_record(
                representative, all_events, start=grid_start, clip_sec=clip_sec,
                mode="paired_event",
                sample_suffix=f"pair_{pair_index:06d}_grid{'ab'[grid_slot]}",
                cfg=cfg, role="edge", pair_id=pair_id, pair_event=event,
            ))

        near_start = e13._sample_near_event_start(
            event, all_events,
            video_duration=duration,
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
            linked_negative = _e16_record(
                representative, all_events, start=near_start, clip_sec=clip_sec,
                mode="paired_event", group_id=pair_id,
                sample_suffix=f"pair_{pair_index:06d}_near", cfg=cfg,
            )
            if not linked_negative.is_negative:
                raise RuntimeError(f"E1.6 near row unexpectedly positive: {pair_id}")
        else:
            fallback = None
            other_keys = [candidate for candidate in keys if candidate != key]
            for _ in range(attempts):
                fallback_key = rng.choice(other_keys)
                fallback_row = representatives[fallback_key]
                fallback_max = max(fallback_row.video_duration - clip_sec, 0.0)
                fallback_start = rng.uniform(0.0, fallback_max) if fallback_max else 0.0
                if not e13._window_conflicts_support(
                    events_by_video.get(fallback_key, ()), fallback_start,
                    fallback_start + clip_sec,
                    min_context_duration_sec=min_span,
                    rejected_margin_sec=rejected_margin,
                ):
                    fallback = fallback_key, fallback_start
                    break
            if fallback is None:
                raise RuntimeError(f"E1.6 cannot build near fallback for {pair_id}")
            fallback_key, fallback_start = fallback
            linked_negative = _e16_record(
                representatives[fallback_key], events_by_video.get(fallback_key, ()),
                start=fallback_start, clip_sec=clip_sec, mode="clean_background",
                group_id=pair_id,
                sample_suffix=f"pair_{pair_index:06d}_near_fallback", cfg=cfg,
            )
            linked_negative = replace(
                linked_negative, online_negative_kind="near_event_fallback_clean"
            )
        result.append(linked_negative)
    hard_records = _load_hard_negative_entries(
        sim, representatives, events_by_video, cfg, rng
    )
    result.extend(hard_records)
    random_clean_count = max(clean_count - len(hard_records), 0)
    for background_index in range(random_clean_count):
        chosen = None
        for _ in range(attempts):
            key = rng.choice(keys)
            row = representatives[key]
            max_start = max(row.video_duration - clip_sec, 0.0)
            start = rng.uniform(0.0, max_start) if max_start else 0.0
            end = start + clip_sec
            unsafe = any(
                e13._interval_gap(
                    start, end,
                    e13._support_interval(event, min_span)[0] - (rejected_margin if event.is_ignored else 0.0),
                    e13._support_interval(event, min_span)[1] + (rejected_margin if event.is_ignored else 0.0),
                ) == 0
                for event in events_by_video.get(key, ())
            )
            if not unsafe:
                chosen = key, start
                break
        if chosen is None:
            raise RuntimeError("E1.6 failed to find a raw-context-safe background")
        key, start = chosen
        result.append(_e16_record(
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
        f"online_simulation_e16 epoch_manifest={epoch} "
        f"event_sampling_mode={event_sampling_mode} "
        f"eligible_unique_events={eligible_unique_total} groups={pair_count} "
        f"grid_windows={grid_window_total} grid_boundary={grid_boundary_total} "
        f"near={near_count} near_fallback={near_fallback_count} "
        f"clean={random_clean_count} hard_negative={len(hard_records)} "
        f"records={len(result)}",
        flush=True,
    )
    return result


_original_load = base.load_long_video_records
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
        records = build_online_records_e16(records, events, cfg, epoch=0)
    eval_cfg = cfg.get("eval", ConfigDict())
    online_val_cfg = eval_cfg.get("online_validation", ConfigDict())
    online_val_enabled = (
        split == "val" and bool(online_val_cfg.get("enabled", False))
    )
    if online_val_enabled:
        records = e13.build_online_eval_records(
            records, events, cfg,
            online_cfg=online_val_cfg, sample_prefix="online_val",
        )
    external_cfg = eval_cfg.get("external_audit", ConfigDict())
    online_eval_cfg = external_cfg.get("online_mode", ConfigDict())
    if (
        split == str(external_cfg.get("split", "external"))
        and bool(external_cfg.get("enabled", False))
        and bool(online_eval_cfg.get("enabled", False))
        and not online_val_enabled
    ):
        records = e13.build_online_eval_records(records, events, cfg)
    return records, events


_original_make_loader = base.make_loader


def online_make_loader(dataset, cfg, *, is_train, batch_size=None, distributed=False):
    sim = cfg.data.long_video.get("online_simulation", ConfigDict())
    if not (is_train and bool(sim.get("enabled", False))):
        return _original_make_loader(
            dataset, cfg, is_train=is_train, batch_size=batch_size, distributed=distributed
        )
    local_batch = batch_size or int(cfg.train.batch_size)
    replicas = base.dist.get_world_size() if distributed else 1
    rank = base.dist.get_rank() if distributed else 0
    sampler = E16GridDistributedSampler(
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
        collate_fn=e13.online_collate,
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
            f"train_sampler mode=online_simulation_e16_grid groups={len(sampler.groups)} "
            f"clean={len(sampler.clean)} samples_per_rank={len(sampler)}",
            flush=True,
        )
    return DataLoader(**kwargs)


class E16GridDistributedSampler(Sampler[int]):
    """Pack whole event groups (canonical + grid windows + near negative) into
    batches.  Each group is padded to 5 rows with cross-video clean
    backgrounds; a batch holds ``batch_size // 5`` whole groups, so
    ``batch_size`` must be a positive multiple of 5."""

    def __init__(self, dataset: Any, *, replicas: int, rank: int, seed: int,
                 drop_last: bool, batch_size: int):
        self.dataset = dataset
        self.replicas, self.rank = int(replicas), int(rank)
        self.seed, self.drop_last = int(seed), bool(drop_last)
        self.batch_size = int(batch_size)
        if self.batch_size < 5 or self.batch_size % 5:
            raise ValueError("E1.6 per-rank batch_size must be a multiple of 5")
        self.groups_per_batch = self.batch_size // 5
        self.epoch = 0
        self._refresh_groups()
        groups_per_global_batch = self.replicas * self.groups_per_batch
        self.num_batches = math.ceil(len(self.groups) / groups_per_global_batch)
        if self.num_batches <= 0:
            raise ValueError("E1.6 has too few event groups for this DDP world size")
        self.num_samples = self.num_batches * self.batch_size

    def _refresh_groups(self) -> None:
        groups: dict[str, dict[str, int]] = {}
        linked: dict[str, list[int]] = {}
        self.clean, self.extras = [], []
        for index, row in enumerate(self.dataset.records):
            if row.online_pair_id:
                role = row.online_pair_role
                if role not in {"central", "edge"}:
                    raise ValueError(f"Invalid E1.6 role {role!r} for {row.online_pair_id}")
                entry = groups.setdefault(row.online_pair_id, {})
                if role == "edge":
                    entry.setdefault(f"grid{len([k for k in entry if k.startswith('grid')])}", index)
                else:
                    if "central" in entry:
                        raise ValueError(f"Duplicate E1.6 central for {row.online_pair_id}")
                    entry["central"] = index
            elif row.online_negative_kind in {"near_event_context", "near_event_fallback_clean"}:
                linked.setdefault(row.online_chunk_id, []).append(index)
            elif row.online_negative_kind == "clean_background":
                self.clean.append(index)
            else:
                self.extras.append(index)
        malformed = {
            key: sorted(value) for key, value in groups.items()
            if "central" not in value
            or not any(name.startswith("grid") for name in value)
        }
        if malformed:
            raise ValueError(f"Malformed E1.6 groups: {malformed}")
        malformed_linked = {
            key: len(linked.get(key, ())) for key in groups
            if len(linked.get(key, ())) != 1
        }
        if malformed_linked:
            raise ValueError(f"E1.6 requires one linked near row per group: {malformed_linked}")
        self.groups = [
            (
                key,
                value["central"],
                sorted(
                    value[name] for name in value if name.startswith("grid")
                ),
                linked[key][0],
            )
            for key, value in sorted(groups.items())
        ]
        if not self.groups or not self.clean:
            raise ValueError("E1.6 requires event groups and clean backgrounds")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if _train_base_records is None or _train_events is None or _train_cfg is None:
            raise RuntimeError("E1.6 dynamic manifest state was not initialized")
        self.dataset.records = build_online_records_e16(
            _train_base_records, _train_events, _train_cfg, epoch=self.epoch
        )
        self._refresh_groups()

    def __len__(self) -> int:
        return self.num_samples

    def _select_clean(self, pool: list[int], video_id: str, cursor: int) -> tuple[int, int]:
        for probe in range(len(pool)):
            candidate = pool[(cursor + probe) % len(pool)]
            if self.dataset.records[candidate].video_id != video_id:
                return candidate, (cursor + probe + 1) % len(pool)
        for candidate in self.clean:
            if self.dataset.records[candidate].video_id != video_id:
                return candidate, cursor
        raise RuntimeError(f"No cross-video clean background for {video_id}")

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        groups, clean = list(self.groups), list(self.clean)
        rng.shuffle(groups)
        rng.shuffle(clean)
        # Pad by repeating at most one global batch minus one groups.  This
        # keeps all real event groups instead of silently dropping the tail to
        # make equally sized DDP batches.
        target_groups = self.num_batches * self.replicas * self.groups_per_batch
        if len(groups) < target_groups:
            groups.extend(groups[: target_groups - len(groups)])
        local_groups = groups[self.rank :: self.replicas]
        local_clean = clean[self.rank :: self.replicas]
        if not local_clean:
            local_clean = clean
        usable = self.num_batches * self.groups_per_batch
        batches: list[list[int]] = []
        clean_cursor = 0
        for offset in range(0, usable, self.groups_per_batch):
            batch: list[int] = []
            for _pair_id, central, grids, near in local_groups[
                offset: offset + self.groups_per_batch
            ]:
                video_id = self.dataset.records[central].video_id
                batch.extend((central, *grids, near))
                selected, clean_cursor = self._select_clean(
                    local_clean, video_id, clean_cursor
                )
                batch.append(selected)
                while len(batch) % 5:
                    selected, clean_cursor = self._select_clean(
                        local_clean, video_id, clean_cursor
                    )
                    batch.append(selected)
            if len(batch) != self.batch_size:
                raise RuntimeError(f"E1.6 batch assembly produced {len(batch)} rows")
            batches.append(batch)
        rng.shuffle(batches)
        ordered = [index for batch in batches for index in batch]
        if len(ordered) < self.num_samples:
            raise RuntimeError(f"E1.6 rank={self.rank} produced too few samples")
        return iter(ordered[: self.num_samples])


def install_e16_hooks() -> None:
    """Install launcher hooks only for an actual E1.6 process."""
    base.load_long_video_records = online_load
    base.FootballLongVideoDataset._sample_window = e13.exact_online_window
    base.make_loader = online_make_loader


if __name__ == "__main__":
    install_e16_hooks()
    base.main()
