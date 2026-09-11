"""E1.3 sampler gate: structural assertions on the real 4-record manifest.

Loads the real E1.3 config and train annotations (no video decode), builds one
epoch of paired records, and verifies the plan 2.2/2.3 contract:
  - exactly event_pairs_per_epoch groups of 4 records;
  - central/edge share one unique pair_id, same anchor, same class mask;
  - central anchor in [3,7]s, edge anchor in [0.5,2]s or [8,9.5]s of the window,
    window shift >= pair_min_shift_sec;
  - near-context: same video, anchor gap in [5,15]s, disjoint from every
    accepted/rejected support span;
  - clean: different video, clear of every support span by the ignore margin;
  - per-class weights: central 1.0, edge 0.35/0.5, near 0.5, clean 1.0;
  - distributed sampler assigns disjoint complete groups per rank.
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "/home/new_users/qiuqi/code/dinov3-main")

import train_football_events as base
import train_football_events_online_simulation as sim

CONFIG_PATH = (
    "/home/new_users/qiuqi/code/dinov3-main/configs/football/"
    "dinov3_vitl16_online_simulation_e13_paired_gate_from_last6r8.yaml"
)


def support_spans(events, rejected_margin, span_min_dur):
    spans = []
    for event in events:
        raw_start = float(event.context_start_time) if event.context_start_time is not None else float(event.anchor_time)
        raw_end = float(event.context_end_time) if event.context_end_time is not None else float(event.anchor_time)
        if event.is_ignored:
            spans.append((raw_start - rejected_margin, raw_end + rejected_margin))
        elif raw_end - raw_start < span_min_dur:
            spans.append((float(event.anchor_time), float(event.anchor_time)))
        else:
            spans.append((raw_start, raw_end))
    return spans


def main() -> int:
    cfg = base.load_config(CONFIG_PATH, [])
    base.configure_label_schema(cfg)
    sim_cfg = cfg.data.long_video.online_simulation
    clip_sec = float(cfg.video.clip_duration)
    pairs_target = int(sim_cfg.event_pairs_per_epoch)
    expected = pairs_target * 4
    rejected_margin = float(cfg.data.get("raw_set_piece_supervision", {}).get("rejected_ignore_margin_sec", 5.0))
    span_min_dur = float(cfg.data.get("raw_set_piece_supervision", {}).get("context_span_min_duration_sec", 0.5))
    ignore_sec = float(sim_cfg.get("near_event_ignore_sec", 2.0))
    gap_min = float(sim_cfg.get("near_event_anchor_gap_min_sec", 5.0))
    gap_max = float(sim_cfg.get("near_event_anchor_gap_max_sec", 15.0))
    shift_min = float(sim_cfg.get("pair_min_shift_sec", 2.0))
    edge_clip_w = float(sim_cfg.get("edge_clip_loss_weight", 0.35))
    edge_frame_w = float(sim_cfg.get("edge_frame_loss_weight", 0.5))
    near_w = float(sim_cfg.get("near_event_negative_weight", 0.5))

    records, events_by_video = sim.online_load(cfg, "train")
    assert len(records) == expected, f"manifest size {len(records)} != {expected}"
    print(f"manifest records={len(records)} groups={len(records)//4}")

    all_events = events_by_video
    pair_ids: set[str] = set()
    errors: list[str] = []
    groups = [records[i : i + 4] for i in range(0, len(records), 4)]
    for group_index, group in enumerate(groups):
        roles = [row.online_pair_role for row in group]
        if roles != ["central", "edge", "", ""]:
            errors.append(f"group {group_index}: roles {roles}")
            continue
        central, edge, near, clean = group
        # pair contract
        pid = central.online_pair_id
        if pid != edge.online_pair_id or not pid:
            errors.append(f"group {group_index}: pair id mismatch {pid} vs {edge.online_pair_id}")
        pair_ids.add(pid)
        if abs(central.anchor_time - edge.anchor_time) > 1e-6 or central.anchor_time < 0:
            errors.append(f"group {group_index}: anchor mismatch {central.anchor_time} vs {edge.anchor_time}")
        if central.online_pair_class_mask != edge.online_pair_class_mask:
            errors.append(f"group {group_index}: class mask mismatch")
        # central anchor in [3,7] of its window
        c_offset = central.anchor_time - central.base_clip_start
        if not (3.0 - 1e-6 <= c_offset <= 7.0 + 1e-6):
            errors.append(f"group {group_index}: central anchor offset {c_offset:.2f} not in [3,7]")
        # edge anchor in [0.5,2] or [8,9.5]; shift >= 2s
        e_offset = edge.anchor_time - edge.base_clip_start
        front = 0.5 - 1e-6 <= e_offset <= 2.0 + 1e-6
        back = clip_sec - 2.0 - 1e-6 <= e_offset <= clip_sec - 0.5 + 1e-6
        if not (front or back):
            errors.append(f"group {group_index}: edge anchor offset {e_offset:.2f} not in [0.5,2] or [8,9.5]")
        if abs(edge.base_clip_start - central.base_clip_start) < shift_min - 1e-6:
            errors.append(f"group {group_index}: window shift {abs(edge.base_clip_start - central.base_clip_start):.2f} < {shift_min}")
        # central/edge positive labels include the pair class
        for row, name in ((central, "central"), (edge, "edge")):
            pos = [i for i, v in enumerate(row.labels) if float(v) > 0]
            if not pos:
                errors.append(f"group {group_index}: {name} has no positive labels")
            for i, v in enumerate(row.online_pair_class_mask):
                if float(v) > 0 and float(row.labels[i]) <= 0:
                    errors.append(f"group {group_index}: {name} pair class {base.LABELS[i]} not positive")
        # edge weights
        target_classes = [i for i, v in enumerate(central.online_pair_class_mask) if float(v) > 0]
        for i in target_classes:
            if abs(float(central.online_clip_loss_weights[i]) - 1.0) > 1e-6:
                errors.append(f"group {group_index}: central clip weight {central.online_clip_loss_weights[i]}")
            if abs(float(edge.online_clip_loss_weights[i]) - edge_clip_w) > 1e-6:
                errors.append(f"group {group_index}: edge clip weight {edge.online_clip_loss_weights[i]}")
            if abs(float(edge.online_frame_loss_weights[i]) - edge_frame_w) > 1e-6:
                errors.append(f"group {group_index}: edge frame weight {edge.online_frame_loss_weights[i]}")
        # near contract
        key = (near.source, near.video_id)
        spans = support_spans(all_events.get(key, ()), rejected_margin, span_min_dur)
        if near.online_negative_kind == "near_event_context":
            anchor_gap = abs(central.anchor_time - ((near.base_clip_start + near.base_clip_end) / 2))
            if not (gap_min - 2.0 <= anchor_gap <= gap_max + clip_sec / 2 + 1e-6):
                errors.append(f"group {group_index}: near gap {anchor_gap:.2f} out of [{gap_min},{gap_max}+window]")
            if near.video_id != central.video_id:
                errors.append(f"group {group_index}: near from different video")
            for s, t in spans:
                if near.base_clip_end > s and near.base_clip_start < t:
                    errors.append(f"group {group_index}: near window overlaps support [{s:.1f},{t:.1f}]")
                    break
            for w in near.online_clip_loss_weights:
                if abs(float(w) - near_w) > 1e-6:
                    errors.append(f"group {group_index}: near weight {w}")
                    break
        elif near.online_negative_kind == "clean_background_fallback":
            if near.video_id == central.video_id:
                errors.append(f"group {group_index}: fallback clean same video")
        else:
            errors.append(f"group {group_index}: unexpected near kind {near.online_negative_kind}")
        # clean contract
        if clean.video_id == central.video_id:
            errors.append(f"group {group_index}: clean same video as pair")
        ckey = (clean.source, clean.video_id)
        cspans = support_spans(all_events.get(ckey, ()), rejected_margin, span_min_dur)
        ok = True
        for s, t in cspans:
            if clean.base_clip_end > s - ignore_sec and clean.base_clip_start < t + ignore_sec:
                errors.append(f"group {group_index}: clean window near support [{s:.1f},{t:.1f}]")
                ok = False
                break
        for w in clean.online_clip_loss_weights:
            if abs(float(w) - 1.0) > 1e-6:
                errors.append(f"group {group_index}: clean weight {w}")
                break

    if len(pair_ids) != pairs_target:
        errors.append(f"unique pair ids {len(pair_ids)} != {pairs_target}")
    print(f"unique pair_ids={len(pair_ids)} near_context={sum(1 for g in groups if g[2].online_negative_kind == 'near_event_context')} "
          f"fallback_clean={sum(1 for g in groups if g[2].online_negative_kind == 'clean_background_fallback')}")

    # distributed sampler: disjoint complete groups per rank, all 4 rows paired
    world = 4
    per_rank = []

    class FakeDataset:
        def __init__(self, recs):
            self.records = recs

    for rank in range(world):
        sampler = sim.OnlineChunkDistributedSampler(
            FakeDataset(records), replicas=world, rank=rank, seed=42, drop_last=True, batch_size=4
        )
        indices = list(sampler)
        per_rank.append(indices)
        assert len(indices) == expected // world, f"rank {rank} got {len(indices)} != {expected // world}"
        for start in range(0, len(indices), 4):
            group = indices[start : start + 4]
            roles = [records[i].online_pair_role for i in group]
            if "central" not in roles or "edge" not in roles:
                errors.append(f"sampler rank {rank}: group without pair at {start}")
    all_assigned = sorted(sum(per_rank, []))
    if all_assigned != list(range(expected)):
        errors.append("sampler indices not a partition of the manifest")

    if errors:
        print(f"FAILED with {len(errors)} errors; first 20:")
        for line in errors[:20]:
            print("  ", line)
        return 1
    print("SAMPLER_GATE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
