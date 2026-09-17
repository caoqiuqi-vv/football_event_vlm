#!/usr/bin/env python3
"""Freeze a read-only review transaction and clean labels for E1/E2."""
from __future__ import annotations
import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_review_data import clean_video, LABELS


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-db', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--gt-run-dir', type=Path, required=True)
    p.add_argument('--video-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--active-only', action='store_true')
    p.add_argument('--merge-tolerance-sec', type=float, default=3.0)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f'Refusing to overwrite frozen data: {a.output}')
    if a.merge_tolerance_sec <= 0:
        raise ValueError('Merge tolerance must be positive')
    manifest_bytes = a.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    c = sqlite3.connect(a.review_db.resolve().as_uri() + '?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA query_only=ON')
    c.execute('BEGIN')
    # Export annotations only, excluding users, passwords, and sessions.
    sql = '''SELECT e.id,e.video_id,e.source_label,e.source_time_sec,e.source_score,e.payload_json,
        r.status,r.corrected_label,r.corrected_time_sec,r.note,r.reviewer,r.secondary_labels_json,
        r.revision,r.updated_at FROM events e JOIN reviews r ON e.id=r.event_id
        ORDER BY e.video_id,e.source_time_sec,e.id'''
    raw_rows = [dict(r) for r in c.execute(sql)]
    captured_at = datetime.now(timezone.utc).isoformat()
    c.rollback()
    c.close()
    by_video = defaultdict(list)
    for item in raw_rows:
        row = dict(item)
        row['payload'] = json.loads(row.pop('payload_json'))
        row['secondary_labels'] = json.loads(row.pop('secondary_labels_json') or '[]')
        by_video[row['video_id']].append(row)
    videos, excluded, audits, conflicts = [], [], [], []
    gt_hashes, stats = {}, Counter()
    states_by_split = defaultdict(Counter)
    for v in manifest['videos']:
        vid = v['video_id']
        rows = by_video[vid]
        states = Counter(r['status'] for r in rows)
        complete = not states['unreviewed'] and not states['needs_confirmation']
        active = any(r['status'] != 'unreviewed' for r in rows)
        split = 'train' if v['split'] == 'train' else 'val'
        if split == 'val' and not complete:
            excluded.append({'video_id': vid, 'reason': 'incomplete_validation', 'status_counts': dict(states)})
            continue
        if split == 'train' and a.active_only and not active:
            excluded.append({'video_id': vid, 'reason': 'training_not_started'})
            continue
        video_path = a.video_root / (vid + '.mp4')
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        gt_path = a.gt_run_dir / vid / 'gt_events.json'
        gt_bytes = gt_path.read_bytes()
        gt_hashes[vid] = hashlib.sha256(gt_bytes).hexdigest()
        policy = clean_video(rows, json.loads(gt_bytes), float(v['duration_sec']), a.merge_tolerance_sec)
        policy['negative_margin_sec'] = 2.0
        for event in policy['events']:
            stats[f"{split}_events_{event['label']}"] += 1
            stats[f'{split}_time_precise' if event['time_precise'] else f'{split}_time_weak'] += 1
            stats[f'{split}_confirmed' if event['confirmed'] else f'{split}_inherited_unreviewed_gt'] += 1
        stats.update(f"unknown_{u['kind']}" for u in policy['unknown'])
        for action in policy.pop('audit'):
            audits.append({'video_id': vid, **action})
            stats['cleaning_' + action['action']] += 1
        conflicts.extend({'video_id': vid, **x} for x in policy.pop('conflicts'))
        states_by_split[split].update(states)
        stat = video_path.stat()
        videos.append({'video_id': vid, 'split': split, 'video_path': str(video_path.absolute()),
                       'resolved_video_path': str(video_path.resolve()), 'media_size': stat.st_size,
                       'media_mtime_ns': stat.st_mtime_ns, 'duration_sec': float(v['duration_sec']),
                       'first_pass_complete': complete, 'status_counts': dict(states), 'policy': policy})
    snapshot = {'schema_version': 'football_review_training_v1', 'captured_at': captured_at,
                'labels': list(LABELS), 'source': {'review_db': str(a.review_db.resolve()),
                'review_manifest': str(a.manifest.resolve()),
                'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                'gt_run_dir': str(a.gt_run_dir.resolve()), 'gt_hashes': gt_hashes,
                'review_max_updated_at': max(r['updated_at'] or '' for r in raw_rows)},
                'policy': {'merge_tolerance_sec': a.merge_tolerance_sec, 'active_only': a.active_only,
                'negatives': 'human_rejected_class_intervals_only_minus_positive_and_unknown_support',
                'ambiguity': 'human_ambiguous_separate_from_unchecked_candidates; no invented_soft_labels',
                'temporal_precision': 'confirmed_same_class_gt_only; relocated_candidates_remain_weak',
                'evaluation_scope': 'completed_validation_videos; trusted_temporal_regions_only',
                'dedup': 'same_gt_lineage_or_unique_confirmed_gt_ai_or_identical_dense_evidence; preserve_distinct_gt'},
                'videos': videos}
    a.output.mkdir(parents=True)
    raw_text = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in raw_rows)
    (a.output / 'annotation_rows_snapshot.jsonl').write_text(raw_text)
    snapshot['source']['annotation_rows_sha256'] = hashlib.sha256(raw_text.encode()).hexdigest()
    dump(a.output / 'training_manifest.json', snapshot)
    digest = hashlib.sha256((a.output / 'training_manifest.json').read_bytes()).hexdigest()
    summary = {'captured_at': captured_at, 'manifest_sha256': digest,
               'video_counts': dict(Counter(v['split'] for v in videos)),
               'status_counts_by_split': {k: dict(v) for k, v in states_by_split.items()},
               'statistics': dict(stats), 'conflict_pairs': len(conflicts), 'excluded': excluded,
               'evaluation_scope': snapshot['policy']['evaluation_scope']}
    dump(a.output / 'cleaning_report.json', summary)
    dump(a.output / 'cleaning_decisions.json', audits)
    dump(a.output / 'cleaning_conflicts.json', conflicts)
    for split in ('train', 'val'):
        (a.output / (split + '_video_ids.txt')).write_text(''.join(v['video_id'] + '\n' for v in videos if v['split'] == split))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
