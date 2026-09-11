"""Online-simulation E1.3 paired launcher without changing the stable trainer.

Every training micro-batch is a fixed 4-record group: central positive (3-7s),
edge positive (0.5-2s or 8-9.5s, same annotated event, shared pair_id),
linked near-context (same video, 5-15s away, disjoint from every accepted/
rejected support span), and a cross-video clean window. Model, losses and
validation stay in :mod:`train_football_events`.
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


def _event_support_span(
    event: base.FootballEvent,
    *,
    rejected_margin_sec: float,
    span_min_duration_sec: float,
) -> tuple[float, float]:
    """E1.3 2.3: accepted uses the raw valid context span (anchor fallback);
    rejected set-piece spans expand by the ignore margin so near/clean windows
    cannot be sampled over them."""
    raw_start = (
        float(event.context_start_time)
        if event.context_start_time is not None
        else float(event.anchor_time)
    )
    raw_end = (
        float(event.context_end_time)
        if event.context_end_time is not None
        else float(event.anchor_time)
    )
    if event.is_ignored:
        return (raw_start - rejected_margin_sec, raw_end + rejected_margin_sec)
    if raw_end - raw_start < span_min_duration_sec:
        return (float(event.anchor_time), float(event.anchor_time))
    return (raw_start, raw_end)


def _window_clear(
    start: float, end: float, spans: Sequence[tuple[float, float]], *, margin_sec: float
) -> bool:
    for span_start, span_end in spans:
        if end > span_start - margin_sec and start < span_end + margin_sec:
            return False
    return True


def _weights_for_positive_classes(
    num_labels: int, positive_classes: Sequence[int], clip_weight: float, frame_weight: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    clip = [1.0] * num_labels
    frame = [1.0] * num_labels
    for label_index in positive_classes:
        clip[label_index] = clip_weight
        frame[label_index] = frame_weight
    return tuple(clip), tuple(frame)


def build_online_records(
    records: Sequence[base.LongVideoRecord],
    events_by_video: dict[tuple[str, str], list[base.FootballEvent]],
    cfg: Any,
    *,
    epoch: int = 0,
) -> list[base.LongVideoRecord]:
    """E1.3 paired online-simulation manifest.

    Every per-rank micro-batch is a fixed 4-record group:
      1. central positive  (anchor at 3-7s of the 10s window)
      2. edge positive     (anchor at 0.5-2s or 8-9.5s, same event)
      3. linked near-context (same video, 5-15s away, disjoint from every
         accepted/rejected support span; falls back to a cross-video clean row)
      4. cross-video clean (different video, clear of every support span)
    central and edge share a unique ``online_pair_id`` and are guaranteed to
    stay in one local batch by :class:`OnlineChunkDistributedSampler`.
    """
    sim = cfg.data.long_video.get("online_simulation", ConfigDict())
    raw_cfg = cfg.data.get("raw_set_piece_supervision", ConfigDict())
    clip_sec = float(cfg.video.get("clip_duration", 10.0))
    pairs_target = int(sim.get("event_pairs_per_epoch", 0) or 0)
    clean_target = int(sim.get("clean_background_windows_per_epoch", 0) or 0)
    if pairs_target <= 0:
        raise ValueError("E1.3 requires online_simulation.event_pairs_per_epoch > 0")
    if clean_target != pairs_target:
        raise ValueError(
            "E1.3 requires clean_background_windows_per_epoch == event_pairs_per_epoch "
            f"({clean_target} != {pairs_target})"
        )
    if clip_sec <= 0:
        raise ValueError("E1.3 requires video.clip_duration > 0")
    primary_min_sec = base.clamp(
        float(sim.get("primary_positive_min_sec", 3.0)), 0.0, clip_sec
    )
    primary_max_sec = base.clamp(
        float(sim.get("primary_positive_max_sec", 7.0)), primary_min_sec, clip_sec
    )
    edge_min_sec = base.clamp(float(sim.get("edge_position_min_sec", 0.5)), 0.0, clip_sec)
    edge_max_sec = base.clamp(
        float(sim.get("edge_position_max_sec", 2.0)), edge_min_sec, clip_sec
    )
    pair_min_shift_sec = max(float(sim.get("pair_min_shift_sec", 2.0)), 0.0)
    ignore_sec = max(float(sim.get("near_event_ignore_sec", 2.0)), 0.0)
    near_gap_min_sec = max(float(sim.get("near_event_anchor_gap_min_sec", 5.0)), 0.0)
    near_gap_max_sec = max(
        float(sim.get("near_event_anchor_gap_max_sec", 15.0)), near_gap_min_sec
    )
    near_event_negative_weight = float(sim.get("near_event_negative_weight", 0.5))
    edge_clip_loss_weight = float(sim.get("edge_clip_loss_weight", 0.35))
    edge_frame_loss_weight = float(sim.get("edge_frame_loss_weight", 0.5))
    if not 0 < edge_clip_loss_weight <= 1 or not 0 < edge_frame_loss_weight <= 1:
        raise ValueError("edge clip/frame weights must be in (0,1]")
    if not 0 < near_event_negative_weight <= 1:
        raise ValueError("near_event_negative_weight must be in (0,1]")
    attempts = max(int(sim.get("background_max_attempts", 256)), 1)
    rejected_margin_sec = max(
        float(raw_cfg.get("rejected_ignore_margin_sec", 5.0)), 0.0
    )
    span_min_duration_sec = max(
        float(raw_cfg.get("context_span_min_duration_sec", 0.5)), 0.0
    )
    rng = random.Random(
        int(cfg.get("seed", 42))
        + base.stable_int("online_simulation_e13")
        + int(epoch) * 1_000_003
    )

    representatives: dict[tuple[str, str], base.LongVideoRecord] = {}
    for row in records:
        key = (row.source, row.video_id)
        previous = representatives.get(key)
        if previous is None or sum(row.label_mask) > sum(previous.label_mask):
            representatives[key] = row
    keys = [key for key, row in representatives.items() if row.video_duration >= clip_sec]
    all_events = {key: events_by_video.get(key, ()) for key in keys}
    accepted = {
        key: [event for event in events if not event.is_ignored]
        for key, events in all_events.items()
    }
    pools: dict[str, dict[tuple[str, str], list[base.FootballEvent]]] = {
        label: {} for label in base.LABELS
    }
    for key, events in accepted.items():
        for event in events:
            for label, value in zip(base.LABELS, event.labels):
                if float(value) > 0:
                    pools[label].setdefault(key, []).append(event)
    active_labels = [label for label, videos in pools.items() if videos]
    if not keys or not active_labels:
        raise RuntimeError("E1.3 has no eligible videos/events")
    support_spans = {
        key: [
            _event_support_span(
                event,
                rejected_margin_sec=rejected_margin_sec,
                span_min_duration_sec=span_min_duration_sec,
            )
            for event in events
        ]
        for key, events in all_events.items()
    }

    def window_row(
        representative: base.LongVideoRecord,
        key: tuple[str, str],
        start: float,
        end: float,
        *,
        labels: tuple[float, ...],
        sample_id: str,
        chunk_id: str,
        negative_kind: str,
        positive_kind: str,
        clip_weights: tuple[float, ...],
        frame_weights: tuple[float, ...],
        clean_mask: tuple[float, ...],
        pair_id: str = "",
        role: str = "",
        class_mask: tuple[float, ...] = (),
        anchor_time: float = -1.0,
        focus_labels: tuple[float, ...] = (),
    ) -> base.LongVideoRecord:
        return replace(
            representative,
            sample_id=sample_id,
            anchor_time=anchor_time,
            base_clip_start=start,
            base_clip_end=end,
            is_negative=not any(float(value) > 0 for value in labels),
            labels=labels,
            focus_labels=focus_labels,
            save_cohort_kind="none",
            save_cohort_weight=0.0,
            online_chunk_id=chunk_id,
            online_chunk_mode="e13_pair",
            online_window_index=0,
            online_negative_kind=negative_kind,
            online_positive_kind=positive_kind,
            online_clean_negative_mask=clean_mask,
            online_clip_loss_weights=clip_weights,
            online_frame_loss_weights=frame_weights,
            online_label_loss_weights=clip_weights,
            online_pair_id=pair_id,
            online_pair_role=role,
            online_pair_class_mask=class_mask,
        )

    zero_mask = tuple(0.0 for _ in range(len(base.LABELS)))
    result: list[base.LongVideoRecord] = []
    used_events: set[tuple[str, str, str]] = set()
    near_rows = 0
    fallback_clean_rows = 0
    for pair_index in range(pairs_target):
        label = active_labels[pair_index % len(active_labels)]
        videos = list(pools[label])
        eligible = [
            key
            for key in videos
            if any(
                (key[0], key[1], str(event.event_id)) not in used_events
                for event in pools[label][key]
            )
        ]
        key = rng.choice(eligible or videos)
        choices = [
            event
            for event in pools[label][key]
            if (key[0], key[1], str(event.event_id)) not in used_events
        ] or list(pools[label][key])
        event = rng.choice(choices)
        used_events.add((key[0], key[1], str(event.event_id)))
        representative = representatives[key]
        max_start = max(representative.video_duration - clip_sec, 0.0)
        anchor = float(event.anchor_time)
        events = all_events[key]
        spans = support_spans[key]
        class_mask = tuple(
            float(float(value) > 0) for value in event.labels
        )
        target_classes = [
            index for index in range(len(base.LABELS)) if class_mask[index] > 0
        ]
        pair_id = f"e13_pair_{int(epoch):03d}_{pair_index:06d}"
        chunk_id = f"online_chunk_{pair_index:06d}_{label}_{representative.video_id}"

        # 1. central positive: anchor at 3-7s of the window, full weights.
        start_c = base.clamp(anchor - rng.uniform(primary_min_sec, primary_max_sec), 0.0, max_start)
        end_c = start_c + clip_sec
        labels_c = base.labels_for_window(events, start_c, end_c)
        central_clip, central_frame = _weights_for_positive_classes(
            len(base.LABELS), target_classes, 1.0, 1.0
        )
        result.append(
            window_row(
                representative, key, start_c, end_c,
                labels=labels_c,
                sample_id=f"{chunk_id}_central",
                chunk_id=chunk_id,
                negative_kind="none",
                positive_kind="primary",
                clip_weights=central_clip,
                frame_weights=central_frame,
                clean_mask=zero_mask,
                pair_id=pair_id,
                role="central",
                class_mask=class_mask,
                anchor_time=anchor,
                focus_labels=event.labels,
            )
        )

        # 2. edge positive: anchor at 0.5-2s (front) or 8-9.5s (back), same event,
        #    at least pair_min_shift_sec away from the central window.
        edge_start = None
        for _attempt in range(8):
            if rng.random() < 0.5:
                offset = rng.uniform(edge_min_sec, edge_max_sec)
            else:
                offset = rng.uniform(clip_sec - edge_max_sec, clip_sec - edge_min_sec)
            candidate = base.clamp(anchor - offset, 0.0, max_start)
            if abs(candidate - start_c) >= pair_min_shift_sec:
                edge_start = candidate
                break
        if edge_start is None:
            front = base.clamp(anchor - edge_min_sec, 0.0, max_start)
            back = base.clamp(anchor - (clip_sec - edge_min_sec), 0.0, max_start)
            edge_start = front if abs(front - start_c) >= abs(back - start_c) else back
        end_e = edge_start + clip_sec
        labels_e = base.labels_for_window(events, edge_start, end_e)
        edge_clip, edge_frame = _weights_for_positive_classes(
            len(base.LABELS), target_classes, edge_clip_loss_weight, edge_frame_loss_weight
        )
        result.append(
            window_row(
                representative, key, edge_start, end_e,
                labels=labels_e,
                sample_id=f"{chunk_id}_edge",
                chunk_id=chunk_id,
                negative_kind="none",
                positive_kind="boundary",
                clip_weights=edge_clip,
                frame_weights=edge_frame,
                clean_mask=zero_mask,
                pair_id=pair_id,
                role="edge",
                class_mask=class_mask,
                anchor_time=anchor,
                focus_labels=event.labels,
            )
        )

        # 3. linked near-context: same video, anchor gap 5-15s, window disjoint
        #    from every accepted/rejected support span.
        near_start = None
        for _attempt in range(attempts):
            if rng.random() < 0.5:
                start_range = (anchor - near_gap_max_sec, anchor - near_gap_min_sec)
            else:
                start_range = (anchor + near_gap_min_sec, anchor + near_gap_max_sec)
            candidate = base.clamp(
                rng.uniform(start_range[0], start_range[1]), 0.0, max_start
            )
            if (
                _window_clear(candidate, candidate + clip_sec, spans, margin_sec=0.0)
                and not (candidate <= anchor <= candidate + clip_sec)
            ):
                near_start = candidate
                break
        near_clip = (near_event_negative_weight,) * len(base.LABELS)
        near_frame = (near_event_negative_weight,) * len(base.LABELS)
        if near_start is not None:
            result.append(
                window_row(
                    representative, key, near_start, near_start + clip_sec,
                    labels=base.labels_for_window(events, near_start, near_start + clip_sec),
                    sample_id=f"{chunk_id}_near",
                    chunk_id=chunk_id,
                    negative_kind="near_event_context",
                    positive_kind="none",
                    clip_weights=near_clip,
                    frame_weights=near_frame,
                    clean_mask=zero_mask,
                )
            )
            near_rows += 1
        else:
            # Explicit fallback to a cross-video clean row keeps the 4-record
            # group shape while staying honest that no safe near context exists.
            fallback_clean_rows += 1
            clean_key = None
            clean_start = None
            for _attempt in range(attempts):
                other_key = rng.choice(keys)
                if other_key == key:
                    continue
                other_rep = representatives[other_key]
                other_max = max(other_rep.video_duration - clip_sec, 0.0)
                candidate = rng.uniform(0.0, other_max) if other_max else 0.0
                if _window_clear(
                    candidate,
                    candidate + clip_sec,
                    support_spans[other_key],
                    margin_sec=ignore_sec,
                ):
                    clean_key = other_key
                    clean_start = candidate
                    break
            if clean_key is None:
                raise RuntimeError(
                    "E1.3 failed to find a cross-video clean fallback window; "
                    "audit event density before raising background_max_attempts"
                )
            clean_rep = representatives[clean_key]
            clean_mask = tuple(
                float(float(clean_rep.label_mask[index]) > 0)
                for index in range(len(base.LABELS))
            )
            result.append(
                window_row(
                    clean_rep, clean_key, clean_start, clean_start + clip_sec,
                    labels=(0.0,) * len(base.LABELS),
                    sample_id=f"online_chunk_{pair_index:06d}_fallback_{clean_rep.video_id}",
                    chunk_id=f"online_chunk_{pair_index:06d}_fallback_{clean_rep.video_id}",
                    negative_kind="clean_background_fallback",
                    positive_kind="none",
                    clip_weights=(1.0,) * len(base.LABELS),
                    frame_weights=(1.0,) * len(base.LABELS),
                    clean_mask=clean_mask,
                )
            )

        # 4. cross-video clean: a different video, clear of every support span.
        clean_key = None
        clean_start = None
        for _attempt in range(attempts):
            other_key = rng.choice(keys)
            if other_key == key:
                continue
            other_rep = representatives[other_key]
            other_max = max(other_rep.video_duration - clip_sec, 0.0)
            candidate = rng.uniform(0.0, other_max) if other_max else 0.0
            if _window_clear(
                candidate,
                candidate + clip_sec,
                support_spans[other_key],
                margin_sec=ignore_sec,
            ):
                clean_key = other_key
                clean_start = candidate
                break
        if clean_key is None:
            raise RuntimeError(
                "E1.3 failed to find a cross-video clean window; "
                "audit event density before raising background_max_attempts"
            )
        clean_rep = representatives[clean_key]
        clean_mask = tuple(
            float(float(clean_rep.label_mask[index]) > 0)
            for index in range(len(base.LABELS))
        )
        result.append(
            window_row(
                clean_rep, clean_key, clean_start, clean_start + clip_sec,
                labels=(0.0,) * len(base.LABELS),
                sample_id=f"online_chunk_{pair_index:06d}_clean_{clean_rep.video_id}",
                chunk_id=f"online_chunk_{pair_index:06d}_clean_{clean_rep.video_id}",
                negative_kind="clean_background",
                positive_kind="none",
                clip_weights=(1.0,) * len(base.LABELS),
                frame_weights=(1.0,) * len(base.LABELS),
                clean_mask=clean_mask,
            )
        )

    if len(result) != pairs_target * 4:
        raise RuntimeError(
            f"E1.3 manifest size mismatch: {len(result)} != {pairs_target * 4}"
        )
    result = base.apply_dataset_level_save_cohort_weights(result, cfg, "train")
    central_rows = sum(row.online_pair_role == "central" for row in result)
    edge_rows = sum(row.online_pair_role == "edge" for row in result)
    near_rows = sum(row.online_negative_kind == "near_event_context" for row in result)
    clean_rows = sum(row.online_negative_kind == "clean_background" for row in result)
    fallback_rows = sum(
        row.online_negative_kind == "clean_background_fallback" for row in result
    )
    clean_slots = [
        sum(float(row.online_clean_negative_mask[index]) > 0 for row in result)
        for index in range(len(base.LABELS))
    ]
    pair_ids = [row.online_pair_id for row in result if row.online_pair_role]
    if len(set(pair_ids)) != pairs_target:
        raise RuntimeError(f"E1.3 pair id collision: {len(set(pair_ids))} != {pairs_target}")
    print(
        "online_simulation_e13 "
        f"epoch_manifest={epoch} pairs={pairs_target} records={len(result)} "
        f"central_rows={central_rows} edge_rows={edge_rows} "
        f"near_context_rows={near_rows} clean_background_rows={clean_rows} "
        f"near_fallback_clean_rows={fallback_rows} "
        f"clean_slots={dict(zip(base.LABELS, clean_slots))}",
        flush=True,
    )
    return result


def build_online_eval_records(
    records: Sequence[base.LongVideoRecord],
    events_by_video: dict[tuple[str, str], list[base.FootballEvent]],
    cfg: Any,
) -> list[base.LongVideoRecord]:
    """Expand untouched videos into the exact fixed-stride online scan grid."""
    online_cfg = cfg.get("eval", ConfigDict()).get(
        "external_audit", ConfigDict()
    ).get("online_mode", ConfigDict())
    clip_sec = float(cfg.video.get("clip_duration", 10.0))
    stride_sec = float(online_cfg.get("window_stride_sec", 5.0))
    if stride_sec <= 0:
        raise ValueError("eval.external_audit.online_mode.window_stride_sec must be > 0")
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
                    sample_id=f"online_eval_{representative.video_id}_w{window_index:06d}",
                    anchor_time=(focus.anchor_time if focus else -1.0),
                    base_clip_start=start,
                    base_clip_end=end,
                    is_negative=not any(float(value) > 0 for value in labels),
                    labels=labels,
                    focus_labels=(focus.labels if focus else ()),
                    online_chunk_id=f"online_eval_{representative.video_id}",
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
        f"online_external_eval videos={len(representatives)} windows={len(result)} "
        f"clip_sec={clip_sec:g} stride_sec={stride_sec:g}",
        flush=True,
    )
    return result


class OnlineChunkDistributedSampler(Sampler[int]):
    """Assign E1.3 4-record groups; each group is one local micro-batch.

    Groups are produced by :func:`build_online_records` as fixed
    ``[central, edge, near-or-fallback, clean]`` rows. Every rank gets a
    disjoint slice of complete groups, so central/edge pairs always stay in
    the same local batch and the consistency loss can never be silently empty.
    """

    GROUP_SIZE = 4

    def __init__(
        self, dataset: Any, *, replicas: int, rank: int, seed: int,
        drop_last: bool, batch_size: int,
    ):
        self.dataset = dataset
        self.replicas = int(replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if int(batch_size) != self.GROUP_SIZE:
            raise ValueError(
                "E1.3 requires per_gpu_batch_size == 4 (one group per micro-batch); "
                f"got {batch_size}"
            )
        self.batch_size = self.GROUP_SIZE
        self.epoch = 0
        self._refresh_groups()
        groups_per_rank = (
            len(self.groups) // self.replicas
            if self.drop_last
            else int(math.ceil(len(self.groups) / self.replicas))
        )
        self.num_samples = groups_per_rank * self.GROUP_SIZE
        self.total_size = self.num_samples * self.replicas

    def _refresh_groups(self) -> None:
        records = self.dataset.records
        if len(records) % self.GROUP_SIZE != 0:
            raise ValueError(
                f"E1.3 records must be a multiple of {self.GROUP_SIZE}: {len(records)}"
            )
        groups: list[list[int]] = []
        for start in range(0, len(records), self.GROUP_SIZE):
            group = records[start : start + self.GROUP_SIZE]
            roles = [row.online_pair_role for row in group]
            if "central" not in roles or "edge" not in roles:
                raise ValueError(
                    f"E1.3 group {start // self.GROUP_SIZE} lacks a central/edge pair"
                )
            pair_ids = {
                row.online_pair_id
                for row in group
                if row.online_pair_role in ("central", "edge")
            }
            if len(pair_ids) != 1:
                raise ValueError(
                    "E1.3 central/edge rows must share one unique pair_id; "
                    f"group {start // self.GROUP_SIZE} has {pair_ids}"
                )
            groups.append(list(range(start, start + self.GROUP_SIZE)))
        self.groups = groups
        if not self.groups:
            raise ValueError("E1.3 sampler found no groups")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if _train_base_records is None or _train_events is None or _train_cfg is None:
            raise RuntimeError("E1.3 dynamic manifest state was not initialized")
        records = build_online_records(
            _train_base_records, _train_events, _train_cfg, epoch=self.epoch
        )
        if len(records) != len(self.dataset.records):
            raise RuntimeError(
                "E1.3 dynamic manifest changed dataset size: "
                f"{len(self.dataset.records)} -> {len(records)}"
            )
        self.dataset.records = records
        self._refresh_groups()

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        groups = list(self.groups)
        rng.shuffle(groups)
        groups_per_rank = (
            len(groups) // self.replicas
            if self.drop_last
            else int(math.ceil(len(groups) / self.replicas))
        )
        mine = groups[self.rank * groups_per_rank : (self.rank + 1) * groups_per_rank]
        if self.drop_last and len(mine) * self.GROUP_SIZE < self.num_samples:
            raise RuntimeError(
                f"E1.3 sampler rank={self.rank} produced {len(mine) * self.GROUP_SIZE} "
                f"samples; expected {self.num_samples}"
            )
        return iter(index for group in mine for index in group)

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
        records = build_online_records(records, events, cfg, epoch=0)
    external_cfg = cfg.get("eval", ConfigDict()).get(
        "external_audit", ConfigDict()
    )
    online_eval_cfg = external_cfg.get("online_mode", ConfigDict())
    if (
        split == str(external_cfg.get("split", "external"))
        and bool(external_cfg.get("enabled", False))
        and bool(online_eval_cfg.get("enabled", False))
    ):
        records = build_online_eval_records(records, events, cfg)
    return records, events


def exact_online_window(self, record, events):
    if record.sample_id.startswith(("online_chunk_", "online_eval_")):
        return float(record.base_clip_start), float(record.base_clip_end)
    return _original_sample_window(self, record, events)


def online_collate(items):
    """Apply independent E1.3 clip/frame supervision weights."""
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
    sampler = OnlineChunkDistributedSampler(
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
            f"train_sampler mode=online_simulation_e13_paired groups={len(sampler.groups)} "
            f"samples_per_rank={len(sampler)} groups_per_rank={len(sampler) // sampler.GROUP_SIZE}",
            flush=True,
        )
    return DataLoader(**kwargs)


base.load_long_video_records = online_load
base.FootballLongVideoDataset._sample_window = exact_online_window
base.make_loader = online_make_loader


if __name__ == "__main__":
    base.main()
