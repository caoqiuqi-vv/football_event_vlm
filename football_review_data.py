"""Provenance-aware review cleaning and conservative window supervision.

No model prediction is promoted to a label without a human decision. Ambiguous
regions block negative supervision; temporal precision is independent of class
confirmation. Legacy training is unchanged unless review_manifest is configured.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

LABELS = ("shot", "save", "set_piece")
FINAL = {"accepted", "modified", "deleted"}
POSITIVE = {"accepted", "modified"}
SET_TYPES = {"corner", "free_kick", "freekick", "penalty", "kickoff"}
RAW_TYPES = {"角球": "corner", "任意球": "free_kick", "点球": "penalty", "中圈开球": "kickoff"}


def semantic(label, secondary=(), raw=""):
    if label in SET_TYPES:
        return "free_kick" if label == "freekick" else label
    if label == "set_piece":
        choices = sorted(set(secondary) & SET_TYPES)
        if len(choices) == 1:
            return semantic(choices[0])
        return RAW_TYPES.get(raw, label)
    return label


def parent(label):
    return "set_piece" if label in SET_TYPES else label


def compatible(left, right):
    return left["label"] == right["label"] and (
        left["semantic_label"] == right["semantic_label"]
        or "set_piece" in (left["semantic_label"], right["semantic_label"])
    )


def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1] + 1e-6:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([float(start), float(end)])
    return result


def overlaps(start, end, interval, margin=0.0):
    return start <= interval[1] + margin and end >= interval[0] - margin


def covers(intervals, start, end):
    return any(a <= start + 1e-3 and b >= end - 1e-3 for a, b in intervals)


def subtract_intervals(intervals, blockers):
    """Remove unknown/event regions before selecting trusted background clips."""
    remaining = merge_intervals(intervals)
    for left, right in merge_intervals(blockers):
        next_intervals = []
        for start, end in remaining:
            if right <= start or left >= end:
                next_intervals.append([start, end])
                continue
            if start < left:
                next_intervals.append([start, left])
            if right < end:
                next_intervals.append([right, end])
        remaining = next_intervals
    return remaining


def clean_video(rows, gt_rows, duration, merge_tolerance=3.0):
    """Merge duplicate evidence, never distinct GT instances or shot/save.

    A confirmed AI record may associate with one uniquely compatible confirmed
    GT within tolerance and overlapping support. An ambiguous association is
    quarantined, rather than resolved by temporal NMS or score ranking.
    """
    gt = []
    for index, item in enumerate(gt_rows):
        lab = parent(item["label"])
        gt.append({"id": str(item.get("event_id", f"gt:{index}")), "label": lab,
                   "semantic_label": semantic(lab, raw=item.get("raw_label", "")),
                   "time_sec": float(item["time_sec"])})
    events, unknown, negatives, audit = [], [], defaultdict(list), []
    for row in rows:
        p = row["payload"]
        lab = parent(row.get("corrected_label") or row["source_label"])
        if lab not in LABELS:
            continue
        t = float(row["corrected_time_sec"] if row.get("corrected_time_sec") is not None else row["source_time_sec"])
        start = max(0.0, float(p.get("support_start_sec", p.get("start_sec", t - 5))))
        end = min(duration, float(p.get("support_end_sec", p.get("end_sec", t + 5))))
        if not all(math.isfinite(v) for v in (t, start, end)) or not 0 <= t <= duration or end <= start:
            raise ValueError(f"Invalid event geometry: {row['id']}")
        state = row["status"]
        own_gt_anchors = [float(a["time_sec"]) for a in p.get("evidence_anchors", [])
                          if a.get("source") == "gt" and parent(a.get("label", "")) == lab]
        # Human-added labels often inherit a different class's GT playback.
        # Only an anchor of this exact parent class establishes GT lineage.
        if not p.get("human_added") and parent(row["source_label"]) == lab:
            own_gt_anchors.extend(float(x) for x in p.get("matching_gt_times", []))
        lineage = sorted({g["id"] for g in gt if g["label"] == lab
                          and any(abs(g["time_sec"] - a) <= 0.02 for a in own_gt_anchors)})
        is_gt = bool(lineage) and not p.get("human_added")
        if state == "deleted":
            negatives[lab].append([start, end])
            audit.append({"action": "confirmed_rejection", "source_ids": [row["id"]], "label": lab})
            continue
        if state == "needs_confirmation":
            unknown.append({"label": lab, "interval": [start, end], "kind": "human_ambiguous", "source_ids": [row["id"]]})
            continue
        if state == "unreviewed" and not is_gt:
            unknown.append({"label": lab, "interval": [start, end], "kind": "unchecked_candidate", "source_ids": [row["id"]]})
            continue
        confirmed = state in POSITIVE
        # A changed candidate timestamp is evidence of manual relocation, not
        # proof of an exact action frame. Only existing same-class GT anchors
        # are initially eligible for strong temporal supervision.
        precise = confirmed and is_gt
        support = [t, t] if precise else [min(start, t), max(end, t)]
        events.append({"id": row["id"], "label": lab,
                       "semantic_label": semantic(lab, row.get("secondary_labels", [])),
                       "time_sec": t, "support": support, "playback_support": [start, end],
                       "confirmed": confirmed, "time_precise": precise,
                       "origin": "gt" if is_gt else "human_added" if p.get("human_added") else "candidate",
                       "manual_time_changed": abs(t - float(row["source_time_sec"])) > 0.01,
                       "lineage_gt_ids": lineage, "source_ids": [row["id"]],
                       "sources": [{"id": row["id"], "status": state, "origin": p.get("task_source", "unknown"),
                                    "source_time_sec": float(row["source_time_sec"]), "review_time_sec": t,
                                    "secondary_labels": row.get("secondary_labels", []),
                                    "human_added": bool(p.get("human_added"))}],
                       "window_indices": list(p.get("window_indices", []))})
    events.sort(key=lambda e: (e["time_sec"], e["label"], e["id"]))
    # Group repeated references to ONE original GT, with compatible subtypes.
    canonical = []
    for event in events:
        matches = [c for c in canonical if compatible(c, event)
                   and c["lineage_gt_ids"] and c["lineage_gt_ids"] == event["lineage_gt_ids"]
                   and abs(c["time_sec"] - event["time_sec"]) <= merge_tolerance]
        if len(matches) == 1 and len(event["lineage_gt_ids"]) == 1:
            _merge(matches[0], event, audit, "same_gt_lineage")
        else:
            canonical.append(event)
    gt_confirmed = [c for c in canonical if c["confirmed"] and c["lineage_gt_ids"]]
    removed = set()
    for event in canonical:
        if event["lineage_gt_ids"] or not event["confirmed"]:
            continue
        options = [g for g in gt_confirmed if compatible(g, event)
                   and abs(g["time_sec"] - event["time_sec"]) <= merge_tolerance
                   and overlaps(*event["playback_support"], g["playback_support"])]
        # Even an unreviewed nearby GT makes a nearest-time association unsafe.
        nearby_gt_ids = {g["id"] for g in gt if compatible(g, event)
                         and abs(g["time_sec"] - event["time_sec"]) <= merge_tolerance}
        if len(options) == 1 and nearby_gt_ids == set(options[0]["lineage_gt_ids"]):
            _merge(options[0], event, audit, "unique_confirmed_gt_ai_association")
            removed.add(event["id"])
    canonical = [c for c in canonical if c["id"] not in removed]
    # Exact duplicate AI evidence can repeat due to UI insertion. Require
    # shared model windows; equality in time alone does not establish identity.
    removed = set()
    for i, left in enumerate(canonical):
        if left["id"] in removed or left["lineage_gt_ids"]:
            continue
        for right in canonical[i + 1:]:
            if right["time_sec"] - left["time_sec"] > 0.001:
                break
            if right["id"] in removed or right["lineage_gt_ids"]:
                continue
            if compatible(left, right) and left["confirmed"] and right["confirmed"] and set(left["window_indices"]) & set(right["window_indices"]):
                _merge(left, right, audit, "identical_dense_evidence")
                removed.add(right["id"])
    canonical = [c for c in canonical if c["id"] not in removed]
    conflicts = []
    for i, left in enumerate(canonical):
        for right in canonical[i + 1:]:
            if right["time_sec"] - left["time_sec"] > merge_tolerance:
                break
            if left["label"] != right["label"]:
                continue
            lg, rg = set(left["lineage_gt_ids"]), set(right["lineage_gt_ids"])
            if lg and rg and lg.isdisjoint(rg):
                audit.append({"action": "preserve_distinct_gt", "source_ids": left["source_ids"] + right["source_ids"], "label": left["label"]})
                continue
            interval = [min(left["support"][0], right["support"][0]), max(left["support"][1], right["support"][1])]
            conflicts.append({"label": left["label"], "interval": interval,
                              "kind": "unresolved_same_class_neighbors", "source_ids": left["source_ids"] + right["source_ids"]})
    unknown.extend(conflicts)
    for event in canonical:
        if len(event["lineage_gt_ids"]) > 1:
            unknown.append({"label": event["label"], "interval": event["playback_support"],
                            "kind": "multiple_gt_lineages", "source_ids": event["source_ids"]})
    return {"events": canonical, "unknown": unknown,
            "trusted_negative_intervals": {lab: merge_intervals(negatives[lab]) for lab in LABELS},
            "conflicts": conflicts, "audit": audit}


def _merge(target, source, audit, reason):
    if source["time_precise"] and (not target["time_precise"] or
            (source.get("manual_time_changed") and not target.get("manual_time_changed"))):
        for key in ("time_sec", "support", "time_precise", "confirmed", "manual_time_changed"):
            target[key] = source[key]
    target["source_ids"].extend(source["source_ids"])
    target["sources"].extend(source["sources"])
    target["window_indices"] = sorted(set(target["window_indices"]) | set(source["window_indices"]))
    if target["semantic_label"] == "set_piece":
        target["semantic_label"] = source["semantic_label"]
    audit.append({"action": "merge", "reason": reason, "canonical_id": target["id"],
                  "source_ids": source["source_ids"], "label": target["label"],
                  "source_time_sec": source["time_sec"], "canonical_time_sec": target["time_sec"]})


def window_supervision(policy, start, end, *, temporal_evaluation=False):
    """Return independent clip and frame-class masks for the ACTUAL window."""
    targets, masks, frame_masks = [0.0] * 3, [0.0] * 3, [0.0] * 3
    margin = float(policy.get("negative_margin_sec", 2.0))
    for i, lab in enumerate(LABELS):
        blockers = [u["interval"] for u in policy["unknown"] if u["label"] == lab]
        unknown_overlap = any(overlaps(start, end, interval, margin) for interval in blockers)
        conflict_overlap = any(u["label"] == lab and u["kind"] in
                               {"unresolved_same_class_neighbors", "multiple_gt_lineages"}
                               and overlaps(start, end, u["interval"], margin) for u in policy["unknown"])
        if conflict_overlap or (temporal_evaluation and unknown_overlap):
            continue
        related = [e for e in policy["events"] if e["label"] == lab]
        strong = [e for e in related if e["time_precise"] and start <= e["time_sec"] <= end]
        weak_overlap = [e for e in related if not e["time_precise"] and overlaps(start, end, e["support"], margin)]
        weak_contained = [e for e in weak_overlap if start <= e["support"][0] + 1e-3
                          and end >= e["support"][1] - 1e-3]
        if temporal_evaluation and weak_overlap:
            continue
        if strong or weak_contained:
            targets[i], masks[i] = 1.0, 1.0
            frame_masks[i] = float(bool(strong) and not weak_overlap and not unknown_overlap)
        elif weak_overlap or unknown_overlap:
            continue
        elif any(overlaps(start, end, e["support"], margin) for e in related):
            continue
        elif covers(policy["trusted_negative_intervals"].get(lab, []), start, end):
            masks[i] = frame_masks[i] = 1.0
    return targets, masks, frame_masks


@lru_cache(maxsize=4)
def read_manifest(path):
    return json.loads(Path(path).read_text())


def load_records(cfg, split, *, training=None):
    if training is None:
        import train_football_events as training
    if tuple(training.LABELS) != LABELS:
        raise ValueError("Review supervision requires task.label_schema=set_piece")
    path = str(Path(cfg.data.long_video.review_manifest).resolve())
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    expected = cfg.data.long_video.get("review_manifest_sha256")
    if expected and digest != expected:
        raise ValueError("Frozen review manifest hash changed")
    manifest = read_manifest(path)
    records, events_by_video = [], {}
    video_root_override = cfg.data.long_video.get("review_video_root")
    clip = float(cfg.video.clip_duration)
    ratio = float(cfg.data.long_video.get("negative_ratio_by_split", {}).get(split, 1.0))
    for video in manifest["videos"]:
        if video["split"] != split:
            continue
        vid, source, duration = video["video_id"], "xbotgo_0608", float(video["duration_sec"])
        video_path = (
            str(Path(video_root_override).expanduser().resolve() / (vid + ".mp4"))
            if video_root_override else video["video_path"]
        )
        policy = video["policy"]
        events = []
        for event in policy["events"]:
            values = tuple(float(lab == event["label"]) for lab in LABELS)
            events.append(training.FootballEvent(
                source=source, video_id=vid, event_id=event["id"], event_type="",
                raw_label=event["semantic_label"], start_time=event["time_sec"], end_time=event["time_sec"],
                anchor_time=event["time_sec"], labels=values,
                time_supervision=event["time_precise"],
            ))
        if split == "val":
            eligible_ids = set()
            for event in policy["events"]:
                if not event["time_precise"]:
                    continue
                start = max(0.0, min(event["time_sec"] - clip / 2, duration - clip))
                target, mask, _ = window_supervision(policy, start, min(start + clip, duration), temporal_evaluation=True)
                index = LABELS.index(event["label"])
                if target[index] > 0 and mask[index] > 0:
                    eligible_ids.add(event["id"])
            events = [e for e in events if e.event_id in eligible_ids]
        events_by_video[(source, vid)] = events
        windows = {}
        for event in policy["events"]:
            if split == "val" and not event["time_precise"]:
                continue
            support = event["support"]
            width = max(clip, support[1] - support[0])
            start = max(0.0, min(0.5 * sum(support) - width / 2, duration - width))
            end = min(duration, start + width)
            windows[(round(start, 4), round(end, 4))] = ("pos", event["id"])
        # Include explicitly ambiguous playback windows, without manufacturing
        # a 0.5 probability or treating them as background. Other known classes
        # in the same window can still supply supervised gradients.
        if split == "train":
            for index, region in enumerate(policy["unknown"]):
                if region["kind"] != "human_ambiguous":
                    continue
                start, end = region["interval"]
                windows[(round(start, 4), round(end, 4))] = ("ambiguous", str(index))
        blockers_by_label = {}
        for lab in LABELS:
            blockers = [e["support"] for e in policy["events"] if e["label"] == lab]
            blockers += [u["interval"] for u in policy["unknown"] if u["label"] == lab]
            margin = float(policy.get("negative_margin_sec", 2.0))
            blockers_by_label[lab] = [[a - margin, b + margin] for a, b in blockers]
        candidates = set()
        for lab in LABELS:
            safe = subtract_intervals(policy["trusted_negative_intervals"].get(lab, []), blockers_by_label[lab])
            for start, end in safe:
                if end - start < clip - 1e-5:
                    continue
                for index in range(int((end - start - clip) // clip) + 1):
                    candidates.add(round(start + index * clip, 4))
        import random
        ordered = sorted(candidates)
        random.Random(int(cfg.seed) + training.stable_int(vid + split)).shuffle(ordered)
        limit = max(1, int(round(sum(kind == "pos" for kind, _ in windows.values()) * ratio)))
        for index, start in enumerate(ordered[:limit]):
            windows.setdefault((start, min(start + clip, duration)), ("neg", str(index)))
        for (start, end), (kind, identity) in sorted(windows.items()):
            targets, masks, _ = window_supervision(policy, start, end, temporal_evaluation=(split == "val"))
            # Keep deliberate human ambiguity rows for provenance/coverage,
            # including rows with no hard class supervision.
            if not any(masks) and kind != "ambiguous":
                continue
            records.append(training.LongVideoRecord(
                source=source, split=split, video_id=vid, sample_id=f"review_{kind}_{identity}",
                video_path=video_path, annotation_path=path, anchor_time=0.5 * (start + end),
                base_clip_start=start, base_clip_end=end, video_duration=duration,
                is_negative=not any(targets), labels=tuple(targets), label_mask=tuple(masks),
                focus_labels=tuple(targets), review_manifest_path=path, review_fixed_window=True,
            ))
        # Dense validation needs a template even for a zero-positive video.
        if split == "val" and not any(r.video_id == vid for r in records):
            records.append(training.LongVideoRecord(
                source=source, split=split, video_id=vid, sample_id="review_val_template",
                video_path=video_path, annotation_path=path, anchor_time=clip / 2,
                base_clip_start=0.0, base_clip_end=min(clip, duration), video_duration=duration,
                is_negative=True, labels=(0.0,) * 3, label_mask=(0.0,) * 3,
                review_manifest_path=path, review_fixed_window=True,
            ))
    if not records:
        raise RuntimeError(f"No reviewed records for {split}")
    print(f"review_records split={split} videos={len(events_by_video)} rows={len(records)} "
          f"positive={sum(any(r.labels) for r in records)} manifest_sha256={digest}", flush=True)
    return records, events_by_video


def policy_for_record(record):
    return manifest_index(record.review_manifest_path)[record.video_id]["policy"]


@lru_cache(maxsize=4)
def manifest_index(path):
    return {v["video_id"]: v for v in read_manifest(path)["videos"]}


def clean_final_events(events):
    """Resolve exact-time generic/specific duplicates within ONE final QC case.

    The final file can contain both its generic AI event and the specific final
    manual event. Shared case provenance and compatible class are required;
    distinct GT lineages and different semantic subtypes remain independent.
    """
    from copy import deepcopy
    canonical, decisions = [], []
    for raw in sorted(events, key=lambda e: (e['time_sec'], e['semantic_label'], e.get('source_id', ''))):
        event = deepcopy(raw)
        event['merged_source_ids'] = [str(raw.get('id') or raw.get('source_id') or '')]
        event['merged_sources'] = [deepcopy(raw)]
        options = []
        for index, other in enumerate(canonical):
            if not event.get('case_id') or other.get('case_id') != event['case_id']:
                continue
            left = {'label': parent(event['semantic_label']), 'semantic_label': event['semantic_label']}
            right = {'label': parent(other['semantic_label']), 'semantic_label': other['semantic_label']}
            if not compatible(left, right) or abs(event['time_sec']-other['time_sec']) > 0.001:
                continue
            lg, rg = set(event.get('lineage_gt_ids', [])), set(other.get('lineage_gt_ids', []))
            if lg and rg and lg.isdisjoint(rg):
                continue
            options.append(index)
        if len(options) != 1:
            canonical.append(event)
            continue
        index = options[0]
        other = canonical[index]
        # Prefer the explicit subtype supplied by final manual adjudication.
        chosen = event if other['semantic_label'] == 'set_piece' and event['semantic_label'] != 'set_piece' else other
        chosen['merged_source_ids'] = other['merged_source_ids'] + event['merged_source_ids']
        chosen['merged_sources'] = other['merged_sources'] + event['merged_sources']
        decisions.append({'action': 'merge', 'reason': 'same_final_case_exact_time_compatible_semantics',
                          'case_id': event['case_id'], 'source_ids': chosen['merged_source_ids'],
                          'semantic_label': chosen['semantic_label'], 'time_sec': chosen['time_sec']})
        canonical[index] = chosen
    return canonical, decisions
