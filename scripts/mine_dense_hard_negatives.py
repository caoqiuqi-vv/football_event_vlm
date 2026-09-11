#!/usr/bin/env python
"""Mine dense hard negatives from the train split for online-simulation training.

Stage ``predict`` runs the exact fixed-stride online scan (10 s / stride 5 s)
over every train video with a given checkpoint and caches per-window scores
(the same NPZ format the training-time online validation writes).

Stage ``select`` loads that cache plus the train annotations and keeps only
windows that are

* label-complete for the mined class (``masks`` from the cache),
* far from every accepted *and* rejected event support span
  (``--accepted-gap-sec`` / ``--rejected-gap-sec``),
* high scoring under the fused online score (clip prob x frame peak),

then writes a JSON manifest consumed by the E1.7 training sampler via
``data.long_video.online_simulation.hard_negative_manifest``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS = ["shot", "save", "set_piece"]


def run_predict(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    import train_football_events as football
    import train_football_events_online_simulation  # noqa: F401  installs hooks

    cfg = football.load_config(args.config, [])
    cfg["model"]["init_checkpoint"] = str(Path(args.checkpoint).resolve())
    cfg["model"]["init_checkpoint_strict"] = True
    cfg["output_dir"] = str(args.output_dir)
    # Route the online dense scan at the *train* split.  Disable the online
    # simulation sampler branch so "train" is not expanded into training
    # records by the hook.
    cfg["data"]["long_video"]["online_simulation"]["enabled"] = False
    audit = cfg["eval"]["external_audit"]
    audit["enabled"] = True
    audit["split"] = "train"
    online = audit["online_mode"]
    online["enabled"] = True
    online["window_stride_sec"] = 5.0
    online["nms_radius_sec"] = 5.0
    online["tolerance_sec"] = 5.0
    online["capped_clip_sec"] = 10.0
    online["score_fusion"] = "clip_x_frame_peak"
    if args.split_file:
        cfg["data"]["long_video"]["split_files"]["train"] = [str(args.split_file)]
    if args.eval_batch_size:
        cfg["eval"]["per_gpu_batch_size"] = int(args.eval_batch_size)
    if args.eval_frame_chunk:
        cfg["eval"]["backbone_frame_chunk_size"] = int(args.eval_frame_chunk)

    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        cfg["device"] = f"cuda:{local_rank}"
        cfg["gpu_ids"] = list(range(world_size))
    rank = dist.get_rank() if football.distributed_training_active() else 0
    device = torch.device(cfg.device)
    football.configure_runtime_threads(cfg)
    football.configure_label_schema(cfg)
    football.seed_everything(int(cfg.get("seed", 42)) + rank, True)
    topology = football.resolve_runtime_topology(cfg, device)
    if football.distributed_training_active():
        cfg["data"]["num_workers"] = int(topology["num_workers_per_gpu"])

    started = time.time()
    dataset, records = football.prepare_external_audit_dataset(cfg)
    if dataset is None:
        raise RuntimeError("train-split online dataset was not built")
    loader = football.make_loader(
        dataset,
        cfg,
        is_train=False,
        batch_size=int(topology["eval_per_gpu_batch_size"]),
        distributed=football.distributed_training_active(),
    )
    model = football.make_model(cfg, use_cached_features=False, device=device)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"mine_predict checkpoint={cfg.model.init_checkpoint} "
            f"world_size={world_size} records={len(records)}",
            flush=True,
        )
    cache = args.output_dir / "train_online_mining_epoch_000.npz"
    football.evaluate(model, loader, cfg, device, online_cache_path=cache)
    if rank == 0:
        summary = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "records": len(records),
            "runtime_sec": time.time() - started,
            "cache": str(cache),
        }
        (args.output_dir / "train_online_mining_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"cache={cache} runtime_sec={summary['runtime_sec']:.1f}", flush=True)
    if football.distributed_training_active():
        dist.barrier()
        dist.destroy_process_group()


def run_select(args: argparse.Namespace) -> None:
    import train_football_events as base
    import train_football_events_online_simulation_e13 as e13

    cfg = base.load_config(args.config, [])
    base.configure_label_schema(cfg)
    payload = np.load(args.cache, allow_pickle=False)
    labels = [str(v) for v in payload["labels"].tolist()]
    if labels != LABELS:
        raise ValueError(f"unexpected labels {labels}")
    scores = payload["online_probs"].astype(np.float64)
    masks = payload["masks"].astype(np.float64)
    times = payload["candidate_times"].astype(np.float64)
    starts = payload["clip_starts"].astype(np.float64)
    ends = payload["clip_ends"].astype(np.float64)
    meta_path = args.cache.with_suffix(".meta.json")
    metas = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(metas) != len(starts):
        raise ValueError(f"meta rows {len(metas)} != cache rows {len(starts)}")
    video_ids = np.asarray([str(meta.get("video_id", "")) for meta in metas])
    sources = np.asarray([str(meta.get("source", "")) for meta in metas])

    records, events_by_video = base.load_long_video_records(cfg, "train")
    del records
    min_span = float(
        cfg.get("raw_set_piece_supervision", base.ConfigDict()).get(
            "context_span_min_duration_sec", 0.5
        )
    )

    def window_safe(key: tuple[str, str], start: float, end: float) -> bool:
        for event in events_by_video.get(key, ()):
            lo, hi = e13._support_interval(event, min_span)
            margin = (
                float(args.rejected_gap_sec) if event.is_ignored
                else float(args.accepted_gap_sec)
            )
            if e13._interval_gap(start, end, lo - margin, hi + margin) <= 0.0:
                return False
        return True

    # per-class eligible candidates
    per_class_selected: dict[int, dict[tuple[str, str], list[tuple[float, float, int]]]] = {
        c: {} for c in range(len(LABELS))
    }
    eligible_counts = [0] * len(LABELS)
    for i in range(len(video_ids)):
        key = (sources[i], video_ids[i])
        for c in range(len(LABELS)):
            if masks[i, c] <= 0.5:
                continue
            eligible_counts[c] += 1
            if not window_safe(key, float(starts[i]), float(ends[i])):
                continue
            per_class_selected[c].setdefault(key, []).append(
                (float(scores[i, c]), float(times[i, c]), i)
            )

    chosen: dict[int, set[int]] = {}  # window index -> class set
    stats = {}
    for c, label in enumerate(LABELS):
        per_video = per_class_selected[c]
        flat = [item for values in per_video.values() for item in values]
        flat.sort(key=lambda item: item[0], reverse=True)
        stats[label] = {
            "eligible_windows": eligible_counts[c],
            "safe_windows": len(flat),
            "safe_score_p50": float(np.percentile([v[0] for v in flat], 50)) if flat else None,
            "safe_score_p99": float(np.percentile([v[0] for v in flat], 99)) if flat else None,
        }
        taken = 0
        per_video_counts: dict[tuple[str, str], int] = {}
        kept_times: dict[tuple[str, str], list[float]] = {}
        for score, t, i in flat:
            if taken >= int(args.max_per_class):
                break
            key = (sources[i], video_ids[i])
            if per_video_counts.get(key, 0) >= int(args.max_per_video_per_class):
                continue
            if any(abs(t - other) <= args.nms_radius_sec for other in kept_times.get(key, [])):
                continue
            kept_times.setdefault(key, []).append(t)
            per_video_counts[key] = per_video_counts.get(key, 0) + 1
            chosen.setdefault(i, set()).add(c)
            taken += 1
        stats[label]["mined"] = taken

    entries = []
    for i, classes in sorted(chosen.items()):
        entries.append(
            {
                "source": sources[i],
                "video_id": video_ids[i],
                "start": float(starts[i]),
                "end": float(ends[i]),
                "candidate_time": float(times[i, max(classes)]),
                "scores": {LABELS[c]: float(scores[i, c]) for c in range(len(LABELS))},
                "mined_for": sorted(LABELS[c] for c in classes),
            }
        )
    manifest = {
        "cache": str(args.cache),
        "config": str(args.config),
        "accepted_gap_sec": float(args.accepted_gap_sec),
        "rejected_gap_sec": float(args.rejected_gap_sec),
        "nms_radius_sec": float(args.nms_radius_sec),
        "max_per_class": int(args.max_per_class),
        "max_per_video_per_class": int(args.max_per_video_per_class),
        "stats": stats,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    print(f"total_mined_windows={len(entries)}")
    print(f"manifest={args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--split-file", type=Path, default=None,
                   help="optional override for the train split video list (smoke tests)")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-frame-chunk", type=int, default=12)

    s = sub.add_parser("select")
    s.add_argument("--config", required=True)
    s.add_argument("--cache", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--accepted-gap-sec", type=float, default=12.0)
    s.add_argument("--rejected-gap-sec", type=float, default=17.0)
    s.add_argument("--nms-radius-sec", type=float, default=10.0)
    s.add_argument("--max-per-class", type=int, default=8000)
    s.add_argument("--max-per-video-per-class", type=int, default=30)

    args = parser.parse_args()
    if args.mode == "predict":
        run_predict(args)
    else:
        run_select(args)


if __name__ == "__main__":
    main()
