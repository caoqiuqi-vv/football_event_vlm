"""Paired source/index sampling diagnostic; not a human-trajectory O2 result."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def nearest_indices(times, queries):
    if len(times) == 0:
        return np.zeros(len(queries), dtype=int), np.full(len(queries), np.inf)
    right = np.searchsorted(times, queries).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    selected = np.where(abs(times[left] - queries) <= abs(times[right] - queries), left, right)
    return selected, abs(times[selected] - queries)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--video-limit", type=int, default=0)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    split_path = root / "configs/football/splits/thirdparty18_test_long15_val_no_pn_train/internal_val_video_ids.txt"
    videos = sorted(split_path.read_text().split())
    if args.video_limit:
        videos = videos[:args.video_limit]
    coverage_path = root / "outputs/football_o2_pilot_20260907/event_auto_coverage.csv"
    events = defaultdict(list)
    with coverage_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val" and row["video_id"] in videos:
                events[row["video_id"]].append(row)
    source_root = Path("/mnt/data_7t/qiuqi/football_ball_pseudolabels/yolo_fulltrack_v2_test18heldout")
    index_root = root / "outputs/football_external_trajectory_evidence_v2_20260907"
    detail, video_detail, errors = [], [], []
    for video in videos:
        meta_path = source_root / video / "metadata.json"
        raw_path = source_root / video / "ball_pseudolabels.jsonl"
        index_path = index_root / f"{video}.npz"
        if not all(p.exists() for p in [meta_path, raw_path, index_path]):
            errors.append({"video": video, "error": "missing source/index"})
            continue
        meta = json.loads(meta_path.read_text())
        fps, stride = float(meta["fps"]), int(meta["sample_stride"])
        tolerance = stride / fps / 2 + 1e-6
        chosen = {}
        with raw_path.open() as f:
            for line in f:
                row = json.loads(line)
                fid = int(row["source_frame_id"])
                priority = (row.get("source") == "track_observed", float(row.get("quality_weight", 0)), float(row.get("confidence", 0)))
                if fid not in chosen or priority > chosen[fid][0]:
                    chosen[fid] = (priority, row)
        source_frames = np.asarray(sorted(chosen), dtype=np.int64)
        source_times = source_frames / fps
        source_observed = np.asarray([chosen[int(fid)][1].get("source") == "track_observed" for fid in source_frames])
        source_motion = np.asarray([bool(chosen[int(fid)][1].get("usable_for_motion", False)) for fid in source_frames])
        with np.load(index_path) as data:
            names = {str(n): i for i, n in enumerate(data["feature_names"])}
            index_times = data["frame_ids"].astype(float) / float(data["fps"])
            index_feats = data["feats"]
            all_stats = []
            for event in events[video]:
                anchor = float(event["anchor_sec"])
                # Fixed 10 s/24-frame diagnostic queries, rounded to source frames.
                query = np.round(np.linspace(anchor - 5, anchor + 5, 24) * fps) / fps
                old_i, old_gap = nearest_indices(index_times, query)
                source_i, source_gap = nearest_indices(source_times, query)
                old_ok = (old_gap <= .25) & (index_feats[old_i, names["ball_tracked"]] > 0)
                direct_ok = source_gap <= tolerance
                direct_observed = direct_ok & source_observed[source_i] if len(source_frames) else direct_ok.copy()
                direct_motion = direct_ok & source_motion[source_i] if len(source_frames) else direct_ok.copy()
                recovered = ~old_ok & direct_ok
                lost = old_ok & ~direct_ok
                result = {"video_id": video, "event_id": event["event_id"], "label": event["focus_label"],
                          "anchor_sec": anchor, "queries": len(query),
                          "old_index_tracked": int(old_ok.sum()), "direct_source_available": int(direct_ok.sum()),
                          "direct_source_observed": int(direct_observed.sum()),
                          "direct_source_motion_usable": int(direct_motion.sum()),
                          "old_missing_source_available": int(recovered.sum()),
                          "old_available_source_missing": int(lost.sum()),
                          "source_tolerance_sec": tolerance}
                detail.append(result)
                all_stats.append(result)
            counts = {key: sum(r[key] for r in all_stats) for key in
                      ["queries", "old_index_tracked", "direct_source_available", "direct_source_observed",
                       "direct_source_motion_usable", "old_missing_source_available", "old_available_source_missing"]}
            record = {"video_id": video, "events": len(all_stats), "source_fps": fps,
                      "source_stride": stride, "source_sample_hz": fps / stride,
                      "index_sample_hz_median": float(1 / np.median(np.diff(index_times))),
                      "source_rows": len(source_frames), "index_rows": len(index_times), **counts}
            video_detail.append(record)
        print(json.dumps(record), flush=True)

    for name, rows in [("per_event.csv", detail), ("per_video.csv", video_detail)]:
        if rows:
            with (out / name).open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    totals = []
    rng = np.random.default_rng(42)
    for label in ["shot", "save", "set_piece", "all"]:
        rows = [r for r in detail if label == "all" or r["label"] == label]
        n = sum(r["queries"] for r in rows)
        by_video = defaultdict(lambda: np.zeros(3))
        for r in rows:
            by_video[r["video_id"]] += [r["old_index_tracked"], r["direct_source_available"], r["queries"]]
        values = np.asarray(list(by_video.values()))
        boot = []
        if len(values):
            for _ in range(2000):
                sums = values[rng.integers(0, len(values), len(values))].sum(axis=0)
                boot.append((sums[1] - sums[0]) / sums[2])
        totals.append({"label": label, "events": len(rows), "query_slots": n,
                       **{k + "_fraction": sum(r[k] for r in rows) / max(n, 1) for k in
                          ["old_index_tracked", "direct_source_available", "direct_source_observed",
                           "direct_source_motion_usable", "old_missing_source_available", "old_available_source_missing"]},
                       "source_minus_index_coverage_video_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist() if boot else None})
    summary = {"experiment": "O2 prerequisite: paired automatic source/index temporal-alignment diagnostic",
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "human_verified": False, "event_recognition_metrics": None,
               "videos": len(video_detail), "errors": errors, "results": totals,
               "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
               "query_manifest_sha256": hashlib.sha256(coverage_path.read_bytes()).hexdigest(),
               "limitations": ["Direct source values remain detector/track pseudo labels, not corrected GT.",
                               "Coverage gain does not establish localization correctness or event-recognition gain.",
                               "Queries are diagnostic 24-frame uniform anchor-centered windows, not the full deployed grid.",
                               "Overlapping shot/save events contribute separately; bootstrap unit is video.",
                               "Neither the running experiment nor its input index was modified."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = ["自动轨迹时间对齐对照结果（O2 前置诊断）", "",
             "本实验不含人工校正轨迹，不是严格 O2，也没有测量事件识别增益。",
             f"验证视频：{len(video_detail)}；相同事件、相同查询时刻，对比现有索引与原生伪标签时间轴。", "",
             "|类别|事件|现有索引覆盖|原生源覆盖|源 observed 覆盖|", "|---|---:|---:|---:|---:|"]
    for r in totals:
        lines.append(f"|{r['label']}|{r['events']}|{r['old_index_tracked_fraction']:.2%}|{r['direct_source_available_fraction']:.2%}|{r['direct_source_observed_fraction']:.2%}|")
    lines += ["", "查询方式：10 秒 24 帧，按源视频 fps 取整；原生源容差为半个采样周期。",
              "现有索引查询遵循 EvidenceProvider 最近行及 0.25 秒容差。",
              "原生源仍含误检/插值，覆盖增加不能解释为检测召回或事件精度增加。",
              "有效结论应限定为时间轴/查找方式是否影响证据可用性。严格 O2 仍需要同片段校正轨迹和受控事件评测。",
              "所有产物写入独立目录，未修改运行中的 O1 或其索引。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
