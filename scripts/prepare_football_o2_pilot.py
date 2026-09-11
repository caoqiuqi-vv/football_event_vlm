"""Prepare O2 review inputs; never promotes automatic coordinates to human GT."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LABELS = {"射门": "shot", "扑救": "save", "角球": "set_piece",
          "任意球": "set_piece", "点球": "set_piece", "中圈开球": "set_piece"}


def stamp(value):
    parts = str(value).split(":")
    return sum(float(x) * 60 ** i for i, x in enumerate(reversed(parts)))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    split_dir = root / "configs/football/splits/thirdparty18_test_long15_val_no_pn_train"
    files = {"train": "train_video_ids.txt", "val": "internal_val_video_ids.txt",
             "test": "thirdparty18_test_video_ids.txt"}
    splits = {s: set((split_dir / f).read_text().split()) for s, f in files.items()}
    assert not (splits["train"] & splits["val"] or splits["train"] & splits["test"]
                or splits["val"] & splits["test"]), "video split overlap"
    annotation_root = Path("/mnt/data_16t/football/football_events_human_repair")
    index_root = root / "outputs/football_external_trajectory_evidence_v2_20260907"
    dataset_root = Path("/mnt/data/Datasets/Datasets/Football/soccer-ball-detection-datasets/"
                        "spatial_perception/Xbotgo/soccer_ball_merged_v1/datasets")
    source_audit = []
    import re
    for kind in ["static", "spatiotemporal"]:
        path = dataset_root / kind / "metadata/source_manifest.jsonl"
        record = {"path": str(path), "rows": 0, "video_id_matching_rows": 0}
        try:
            with path.open() as f:
                for line in f:
                    record["rows"] += 1
                    if set(re.findall(r"\d{19}", line)) & set.union(*splits.values()):
                        record["video_id_matching_rows"] += 1
            record["status"] = "readable; ID matching alone does not verify human provenance or time alignment"
        except OSError as e:
            record["status"] = f"unavailable: {type(e).__name__}: {e}"
        source_audit.append(record)

    neg_path = root / "outputs/football_hard_negatives/reviewed_mid_score_v2_shot_save_setpiece.json"
    negatives = defaultdict(list)
    for item in json.loads(neg_path.read_text())["hard_negatives"]:
        if item["video_id"] in splits["train"]:
            negatives[item["video_id"]].append(item)
    coverage, candidates, errors, annotation_hashes = [], [], [], {}
    for split in ["train", "val"]:
        for video in sorted(splits[split]):
            ann = annotation_root / f"{video}.json"
            path = index_root / f"{video}.npz"
            media = Path("/mnt/data_16t/football/raw_video_720P") / f"{video}.mp4"
            if not ann.exists() or not path.exists() or not media.exists():
                errors.append({"video_id": video, "reason": "missing annotation/index/media"})
                continue
            raw = ann.read_bytes()
            annotation_hashes[video] = hashlib.sha256(raw).hexdigest()
            events = [(stamp(e["timestamp"]), LABELS[e["label"]], e.get("id", ""))
                      for e in json.loads(raw) if e.get("label") in LABELS
                      and e.get("label_correct") is True]
            with np.load(path) as data:
                feats = data["feats"]
                times = data["frame_ids"].astype(float) / float(data["fps"])
                idx = {str(n): i for i, n in enumerate(data["feature_names"])}
                if not np.isfinite(feats).all() or len(times) == 0:
                    errors.append({"video_id": video, "reason": "invalid index"})
                    continue
                duration = float(times[-1])

                def make(t, label, cohort, provenance, event_id=""):
                    lo, hi = np.searchsorted(times, [max(0, t - 2), t + 2])
                    x = feats[lo:hi]
                    tracked = float((x[:, idx["ball_tracked"]] > 0).mean()) if len(x) else None
                    observed = float((x[:, idx["ball_observed"]] > 0).mean()) if len(x) else None
                    quality = "missing" if tracked is None else ("low" if tracked < .25 else "mid" if tracked < .6 else "high")
                    return {"split": split, "video_id": video, "focus_label": label,
                            "cohort": cohort, "anchor_sec": round(t, 6),
                            "start_sec": round(max(0, t - 5), 6),
                            "end_sec": round(min(duration, t + 5), 6),
                            "event_id": event_id, "media_path": str(media),
                            "provenance": provenance, "event_review_status": "pending_recheck",
                            "spatial_review_status": "pending", "auto_coverage_bin": quality,
                            "auto_ball_tracked_fraction_pm2s": tracked,
                            "auto_ball_observed_fraction_pm2s": observed,
                            "index_rows_pm2s": len(x)}

                for t, label, event_id in events:
                    if 5 <= t <= duration - 5:
                        row = make(t, label, "positive_candidate", str(ann), event_id)
                        coverage.append(row.copy())
                        candidates.append(row)
                if split == "train":
                    for item in negatives[video]:
                        t = float(item["center_sec"])
                        if t < 5 or t > duration - 5:
                            continue
                        if any(abs(t - a) < 15 for a, _, _ in events):
                            continue
                        for label in item["labels"]:
                            candidates.append(make(t, label, "negative_candidate", str(neg_path)))
                else:
                    # No-GT is not a negative label: these are explicitly pending review.
                    for t in np.linspace(10, max(10, duration - 10), 45):
                        if any(abs(t - a) < 20 for a, _, _ in events):
                            continue
                        for label in ["shot", "save", "set_piece"]:
                            candidates.append(make(float(t), label, "negative_candidate",
                                                   "uniform no-GT candidate; NOT a confirmed negative"))

    selected, occupied = [], defaultdict(list)
    for split, quota in [("train", 20), ("val", 10)]:
        for cohort in ["positive_candidate", "negative_candidate"]:
            for label in ["shot", "save", "set_piece"]:
                pool = [r for r in candidates if (r["split"], r["cohort"], r["focus_label"]) == (split, cohort, label)]
                pool.sort(key=lambda r: hashlib.sha256(f"42:{r['video_id']}:{r['anchor_sec']}:{label}".encode()).hexdigest())
                count, per_video = 0, Counter()
                for bucket in ["low", "mid", "high", "missing", "all"]:
                    for r in pool:
                        if count >= quota:
                            break
                        if bucket != "all" and r["auto_coverage_bin"] != bucket:
                            continue
                        if bucket != "all" and sum(s["split"] == split and s["cohort"] == cohort
                               and s["focus_label"] == label and s["auto_coverage_bin"] == bucket
                               for s in selected) >= max(1, quota // 3):
                            continue
                        v, t = r["video_id"], r["anchor_sec"]
                        if per_video[v] >= 3 or any(abs(t - old) < 10 for old in occupied[v]):
                            continue
                        r = dict(r, case_id=f"o2_{len(selected):04d}")
                        selected.append(r)
                        occupied[v].append(t)
                        per_video[v] += 1
                        count += 1

    templates = []
    for r in selected:
        # Full ten-second context plus denser four-second diagnostic interval.
        ts = np.unique(np.round(np.r_[np.arange(r["start_sec"], r["end_sec"] + .001, .5),
                                     np.arange(r["anchor_sec"] - 2, r["anchor_sec"] + 2.001, .125)], 6))
        for t in ts:
            templates.append({"case_id": r["case_id"], "split": r["split"], "video_id": r["video_id"],
                              "timestamp_sec": float(t), "visibility": "unknown",
                              "ball_x_norm": "", "ball_y_norm": "", "ball_track_id": "",
                              "left_goal_x_norm": "", "left_goal_y_norm": "",
                              "right_goal_x_norm": "", "right_goal_y_norm": "",
                              "review_status": "pending", "reviewer": "", "notes": ""})
    assert all(r["video_id"] not in splits["test"] for r in selected)
    assert all(r["ball_x_norm"] == "" for r in templates)
    write_csv(out / "event_auto_coverage.csv", coverage)
    write_csv(out / "review_queue.csv", selected)
    write_csv(out / "human_trajectory_template.csv", templates)
    (out / "review_queue.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "awaiting_verified_human_trajectories", "o2_event_metrics": None,
               "human_corrected_frames_verified": 0,
               "split_counts": {s: len(v) for s, v in splits.items()},
               "selected_cases": len(selected), "annotation_frame_rows": len(templates),
               "selected_counts": dict(Counter(f"{r['split']}/{r['focus_label']}/{r['cohort']}" for r in selected)),
               "accepted_event_coverage_rows": len(coverage), "source_audit": source_audit,
               "annotation_sha256": annotation_hashes,
               "split_sha256": {s: hashlib.sha256((split_dir / f).read_bytes()).hexdigest() for s, f in files.items()},
               "errors": errors,
               "limitations": ["Coverage is automatic-index row coverage, not detection recall.",
                               "Event labels are the current O1 annotation root; candidates require recheck.",
                               "Validation negative candidates are NOT verified negatives.",
                               "Spatial-temporal dataset access may be blocked; no claim of global absence.",
                               "Pilot queue is diagnostic, not a representative full-video evaluation set.",
                               "No model training/evaluation was performed; no O2 metric is available."]}
    (out / "readiness.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    counts = "\n".join(f"- {k}: {v}" for k, v in summary["selected_counts"].items())
    report = f"""O2 准备结果（非 O2 模型效果）

状态：尚未获得可验证且与事件时间轴对齐的人工球轨迹；O2 识别指标为空，没有运行训练。

已输出 {len(selected)} 个待复核片段，{len(templates)} 行人工轨迹填写模板；
统计 {len(coverage)} 个 accepted 事件附近 ±2 秒的自动轨迹覆盖率。

{counts}

review_queue.csv / review_queue.json：原视频绝对路径与片段时间。
human_trajectory_template.csv：坐标为原视频归一化坐标；空值和 unknown 不能当作无球。
event_auto_coverage.csv：自动索引覆盖统计，不是人工检测召回。
readiness.json：数据来源检查、划分/标签哈希、错误和限制。

所有空间坐标留空，未把自动结果冒充人工 GT。负例均需复核；val 无旧 GT 的片段不自动标负。
片段按视频隔离，test18 不进入此准备队列。同片段可包含多个事件，focus_label 不是互斥分类标签。

人工校正应至少确认球是否可见、中心及关联；球门参照需要单独确认。
在身份切换、遮挡和不确定处保留 unknown。4 秒密集区之外也需检查完整 10 秒上下文。
已有静态框清单的直接视频 ID 匹配结果及无法访问的时序目录，见 readiness.json；
零 ID 匹配不排除需要其他映射文件才能找到同源视频。

O2 与 O1 必须在相同样本、事件标签和读出结构上比较；人工轨迹到齐前不输出识别收益。
本次没有停止其他训练，也没有占用 GPU。
"""
    (out / "REPORT.md").write_text(report)
    print(json.dumps({k: summary[k] for k in ["status", "selected_cases", "annotation_frame_rows",
                                              "accepted_event_coverage_rows", "selected_counts", "errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
