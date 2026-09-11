"""Group-held-out exploratory reranker for original versus aligned auto evidence.

Uses internal-val15 only, with disjoint fit/calibration/evaluation videos in
each fold. Does not modify O1 and is not a human-trajectory oracle experiment.
"""
from __future__ import annotations

import os
for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_name] = "2"

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_o2_evidence_alignment import nearest_indices
from scripts.build_external_trajectory_evidence import FEATURE_NAMES, IDX, fill_ball, yolo_rows
from football_online_evaluation import _nms, _metrics, tune_online_event_thresholds


def moments(x):
    return np.concatenate([x.mean(1), x.std(1), x.max(1), x[:, -1] - x[:, 0]], axis=1)


def aligned_native(old_times, old, metadata, rows):
    """Source-rate ball stream, no interpolation across missing native samples."""
    fps, stride = float(metadata["fps"]), int(metadata["sample_stride"])
    ids = np.arange(0, int(metadata["source_frames"]), stride, dtype=np.int64)
    times = ids / fps
    ni, gap = nearest_indices(old_times, times)
    x = np.zeros((len(ids), len(FEATURE_NAMES)), np.float32)
    x[:, IDX["evidence_valid"]] = 1
    # Keep camera estimates unchanged in the alignment-only arm.
    x[:, IDX["camera_vx"]:IDX["camera_speed"] + 1] = old[ni, IDX["camera_vx"]:IDX["camera_speed"] + 1]
    x[gap > .25, IDX["camera_vx"]:IDX["camera_speed"] + 1] = 0
    tracks = np.full(len(ids), -1, np.int64)
    usable = np.zeros(len(ids), bool)
    for fid, row in rows.items():
        if fid % stride or fid < 0 or fid >= metadata["source_frames"]:
            continue
        i = fid // stride
        fill_ball(x, i, row, width=metadata["canonical_width"], height=metadata["canonical_height"], source="yolo")
        tracks[i] = int(row.get("track_id", -1))
        usable[i] = bool(row.get("usable_for_motion", False))
    tracked = x[:, IDX["ball_tracked"]] > 0
    # Reconstruct absolute goal centers only from original rows with a ball.
    for side in ["left", "right"]:
        good = (old[:, IDX[side + "_goal_visible"]] > 0) & (old[:, IDX["ball_tracked"]] > 0)
        if not good.any():
            continue
        goal = old[good][:, [IDX["ball_x"], IDX["ball_y"]]] + old[good][:, [IDX[side + "_goal_dx"], IDX[side + "_goal_dy"]]]
        gi, gg = nearest_indices(old_times[good], times)
        available = (gg <= .25) & (old[ni, IDX[side + "_goal_visible"]] > 0)
        x[:, IDX[side + "_goal_visible"]] = available
        both = available & tracked
        delta = goal[gi] - x[:, [IDX["ball_x"], IDX["ball_y"]]]
        x[both, IDX[side + "_goal_dx"]] = delta[both, 0]
        x[both, IDX[side + "_goal_dy"]] = delta[both, 1]
        x[both, IDX[side + "_goal_dist"]] = np.linalg.norm(delta[both], axis=1)
    left, right = x[:, IDX["left_goal_visible"]] > 0, x[:, IDX["right_goal_visible"]] > 0
    x[:, IDX["both_goals_visible"]] = left & right
    x[:, IDX["any_goal_visible"]] = left | right
    dist = np.stack([np.where(left, x[:, IDX["left_goal_dist"]], np.inf),
                     np.where(right, x[:, IDX["right_goal_dist"]], np.inf)], axis=1).min(1)
    x[:, IDX["nearest_goal_dist"]] = np.where(np.isfinite(dist) & tracked, dist, 0)
    dt = stride / fps
    # Matched identities and adjacent native samples only; missing means unknown velocity.
    valid = tracked[1:] & tracked[:-1] & usable[1:] & usable[:-1] & (tracks[1:] == tracks[:-1]) & (tracks[1:] >= 0)
    velocity = np.zeros((len(x), 2), np.float32)
    delta = np.diff(x[:, [IDX["ball_x"], IDX["ball_y"]]], axis=0) / dt
    velocity[1:][valid] = delta[valid]
    x[:, IDX["ball_raw_vx"]:IDX["ball_raw_vy"] + 1] = velocity.clip(-2, 2)
    comp = velocity - x[:, IDX["camera_vx"]:IDX["camera_vy"] + 1]
    velocity_valid = np.r_[False, valid]
    comp[~velocity_valid] = 0
    x[:, IDX["ball_comp_vx"]:IDX["ball_comp_vy"] + 1] = comp.clip(-2, 2)
    x[:, IDX["ball_comp_speed"]] = np.linalg.norm(comp, axis=1).clip(0, 2)
    acceleration = np.zeros(len(x), np.float32)
    av = velocity_valid[1:] & velocity_valid[:-1]
    acceleration[1:][av] = (np.linalg.norm(np.diff(comp, axis=0), axis=1) / dt)[av]
    x[:, IDX["ball_comp_accel"]] = acceleration.clip(0, 4)
    return ids, times, x


def build_features(data, output):
    old_features = np.zeros((len(data["video_ids"]), 136), np.float32)
    enhanced = np.zeros_like(old_features)
    sampled_old = np.zeros((len(old_features), 24, 34), np.float32)
    sampled_new = np.zeros_like(sampled_old)
    diagnostics = []
    source_root = Path("/mnt/data_7t/qiuqi/football_ball_pseudolabels/yolo_fulltrack_v2_test18heldout")
    for video in sorted(set(data["video_ids"])):
        positions = np.flatnonzero(data["video_ids"] == video)
        metadata = json.loads((source_root / video / "metadata.json").read_text())
        fps = float(metadata["fps"])
        with np.load(ROOT / "outputs/football_external_trajectory_evidence_v2_20260907" / f"{video}.npz") as d:
            assert list(d["feature_names"]) == list(FEATURE_NAMES)
            ot = d["frame_ids"].astype(float) / float(d["fps"])
            old = d["feats"]
        rows = yolo_rows(source_root / video / "ball_pseudolabels.jsonl")
        ids, nt, new = aligned_native(ot, old, metadata, rows)
        start = np.round(data["clip_starts"][positions] * fps).clip(0)
        end = (np.round(data["clip_ends"][positions] * fps) - 1).clip(0, metadata["source_frames"] - 1)
        queries = np.round(start[:, None] + (end - start)[:, None] * np.linspace(0, 1, 24)[None]) / fps
        oi, og = nearest_indices(ot, queries.ravel())
        ni, ng = nearest_indices(nt, queries.ravel())
        a = old[oi].copy(); a[og > .25] = 0
        b = new[ni].copy(); b[ng > metadata["sample_stride"] / fps / 2 + 1e-6] = 0
        a, b = a.reshape(-1, 24, 34), b.reshape(-1, 24, 34)
        sampled_old[positions], sampled_new[positions] = a, b
        old_features[positions], enhanced[positions] = moments(a), moments(b)
        d = {"video": str(video), "windows": len(positions), "old_ball_fraction": float(a[..., 1].mean()),
             "new_ball_fraction": float(b[..., 1].mean())}
        diagnostics.append(d); print("FEATURES", json.dumps(d), flush=True)
        # Small self-contained per-window evidence cache. No original index overwrite.
    np.savez_compressed(output / "window_evidence.npz", old=sampled_old, aligned=sampled_new,
                        video_ids=data["video_ids"], sample_ids=data["sample_ids"])
    (output / "feature_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return old_features, enhanced


def event_metrics(scores, times, metas, thresholds, labels, masks):
    result = {}
    for c, label in enumerate(labels):
        candidates, gt = defaultdict(list), defaultdict(set)
        incomplete = {str(meta["video_id"]) for i, meta in enumerate(metas) if masks[i, c] <= .5}
        for i, meta in enumerate(metas):
            if str(meta["video_id"]) in incomplete:
                continue
            key = (str(meta.get("source", "")), str(meta["video_id"]))
            candidates[key].append((float(scores[i, c]), float(times[i, c])))
            gt[key].update(round(float(t), 4) for t in meta["online_gt_anchors"][c])
        peaks = {key: _nms(items, 5.0) for key, items in candidates.items()}
        result[label] = _metrics(peaks, gt, float(thresholds[c]), 5.0)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "outputs/football_events/vitl16_online_grid_dual_e16_512_from_last6r8/val15_online_ema_epoch_001.npz"
    with np.load(cache) as d:
        data = {k: d[k] for k in d.files}
    metas = json.loads(cache.with_suffix(".meta.json").read_text())
    labels = list(data["labels"])
    assert len(metas) == len(data["video_ids"])
    assert all(m["sample_id"] == s for m, s in zip(metas, data["sample_ids"]))
    assert len(set(data["sample_ids"])) == len(metas), "duplicate scan windows"
    assert set(np.unique(data["masks"])).issubset({0, 1})
    videos = np.asarray(sorted(set(data["video_ids"])))
    expected = set((ROOT / "configs/football/splits/thirdparty18_test_long15_val_no_pn_train/internal_val_video_ids.txt").read_text().split())
    assert set(videos) == expected and len(videos) == 15
    np.random.default_rng(42).shuffle(videos)
    folds = np.array_split(videos, 5)
    protocol = {"status": "preregistered before feature fitting", "fit_videos_per_fold": 9,
                "calibration_videos_per_fold": 3, "heldout_videos_per_fold": 3,
                "folds": [f.tolist() for f in folds], "C": .1, "max_iter": 500,
                "model": "standardized L2 logistic regression; no hyperparameter search",
                "primary": "heldout per-video window AP; strict event P/R secondary",
                "calibration_event_recall_floors": [.85, .8, .8],
                "nms_radius_sec": 5, "one_to_one_tolerance_sec": 5,
                "candidate_times": "unchanged E16 cache frame peaks",
                "scope": "internal-val15 exploratory group cross-validation, not independent final test",
                "unknown_label_policy": "fit/AP only valid windows; strict events only complete video-class pairs",
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "source": str(cache), "time_utc": datetime.now(timezone.utc).isoformat()}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    if (output / "window_evidence.npz").exists():
        with np.load(output / "window_evidence.npz") as d:
            assert np.array_equal(d["sample_ids"], data["sample_ids"])
            old, enhanced = moments(d["old"]), moments(d["aligned"])
    else:
        old, enhanced = build_features(data, output)
    # Quality-only control detects gains attributable to availability/confidence.
    quality_idx = [IDX[k] for k in ["evidence_valid", "ball_tracked", "ball_observed", "ball_confidence",
                                  "ball_quality", "ball_uncertainty", "seconds_since_observed_norm",
                                  "camera_reliability", "left_goal_visible", "right_goal_visible"]]
    qidx = np.asarray([offset + i for offset in [0, 34, 68, 102] for i in quality_idx])
    arms = {"E16_fixed": None, "O0_score_readout": np.zeros((len(old), 0)),
            "O1_old_auto": old, "A2_aligned_auto": enhanced, "A3_quality_only": enhanced[:, qidx]}
    # Same-video circular shift by half a video: preserve marginal scene statistics,
    # remove local alignment. This is a reliance diagnostic, not causal proof.
    shuffled = enhanced.copy()
    for video in videos:
        ii = np.flatnonzero(data["video_ids"] == video)
        ii = ii[np.argsort(data["clip_starts"][ii])]
        shuffled[ii] = np.roll(enhanced[ii], len(ii) // 2, axis=0)
    arms["A4_time_shifted_auto"] = shuffled
    predictions = {arm: np.full_like(data["online_probs"], np.nan) for arm in arms}
    per_video, fold_reports = [], []
    for fold, heldout in enumerate(folds):
        calibration = folds[(fold + 1) % 5]
        train = np.asarray([v for v in videos if v not in heldout and v not in calibration])
        assert not set(train) & set(heldout) and not set(train) & set(calibration) and not set(calibration) & set(heldout)
        fit = np.flatnonzero(np.isin(data["video_ids"], train))
        cal = np.flatnonzero(np.isin(data["video_ids"], calibration))
        test = np.flatnonzero(np.isin(data["video_ids"], heldout))
        for arm, extras in arms.items():
            p = np.zeros_like(data["online_probs"])
            converged = True
            if extras is None:
                p[:] = data["online_probs"]
            else:
                for c in range(3):
                    peak = data["frame_peak_probs"][:, c].clip(1e-6, 1 - 1e-6)
                    fused = data["online_probs"][:, c].clip(1e-6, 1 - 1e-6)
                    scores = np.column_stack([data["logits"][:, c], np.log(peak / (1 - peak)), np.log(fused / (1 - fused))])
                    x = np.concatenate([scores, extras], axis=1)
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500, solver="lbfgs", random_state=42))
                    class_fit = fit[data["masks"][fit, c] > .5]
                    assert len(np.unique(data["targets"][class_fit, c])) == 2
                    model.fit(x[class_fit], data["targets"][class_fit, c])
                    p[:, c] = model.predict_proba(x)[:, 1]
                    converged &= int(model[-1].n_iter_[0]) < 500
            thresholds, calibration_result = tune_online_event_thresholds(
                p[cal], data["candidate_times"][cal], [metas[i] for i in cal], labels,
                ["precision"] * 3, [.85, .8, .8], masks=data["masks"][cal],
                nms_radius_sec=5, tolerance_sec=5, max_candidates=151)
            predictions[arm][test] = p[test]
            for video in heldout:
                indices = np.flatnonzero(data["video_ids"] == video)
                ev = event_metrics(p[indices], data["candidate_times"][indices], [metas[i] for i in indices], thresholds, labels, data["masks"][indices])
                duration_sec = float(data["clip_ends"][indices].max())
                for c, label in enumerate(labels):
                    valid = indices[data["masks"][indices, c] > .5]
                    y = data["targets"][valid, c]
                    complete = bool(np.all(data["masks"][indices, c] > .5))
                    metric = dict(ev[label])
                    per_video.append({"arm": arm, "fold": fold, "video_id": str(video), "label": label,
                                      "window_ap": float(average_precision_score(y, p[valid, c])) if y.sum() else None,
                                      "window_positive_count": int(y.sum()), "duration_sec": duration_sec,
                                      "event_evaluation_included": complete,
                                      "threshold": float(thresholds[c]), **metric})
            held = event_metrics(p[test], data["candidate_times"][test], [metas[i] for i in test], thresholds, labels, data["masks"][test])
            fr = {"fold": fold, "arm": arm, "converged": converged, "fit": train.tolist(),
                  "calibration": calibration.tolist(), "heldout": heldout.tolist(), "thresholds": thresholds.tolist(),
                  "calibration_metrics": calibration_result, "heldout_metrics": held}
            fold_reports.append(fr)
            print("FOLD", json.dumps({"fold": fold, "arm": arm, "converged": converged, "heldout": held}), flush=True)
    assert all(np.isfinite(x).all() for x in predictions.values())
    np.savez_compressed(output / "heldout_predictions.npz", **predictions, video_ids=data["video_ids"],
                        sample_ids=data["sample_ids"], targets=data["targets"], candidate_times=data["candidate_times"])
    (output / "fold_results.json").write_text(json.dumps(fold_reports, indent=2))
    with (output / "per_video_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(per_video[0])); w.writeheader(); w.writerows(per_video)
    summaries = []
    for arm in arms:
        for label in labels:
            rs = [r for r in per_video if r["arm"] == arm and r["label"] == label]
            tp, fp, fn = [sum(r[k] for r in rs) for k in ["tp", "fp", "fn"]]
            aps = [r["window_ap"] for r in rs if r["window_ap"] is not None]
            summaries.append({"arm": arm, "label": str(label), "mean_video_window_ap": float(np.mean(aps)),
                              "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1),
                              "recall": tp / max(tp + fn, 1), "event_evaluation_videos": sum(r["event_evaluation_included"] for r in rs),
                              "fp_per_90min": fp * 5400 / sum(r["duration_sec"] for r in rs if r["event_evaluation_included"])})
    comparisons = []
    for arm, baseline in [("A2_aligned_auto", "O1_old_auto"), ("A2_aligned_auto", "O0_score_readout"),
                          ("A2_aligned_auto", "A3_quality_only"), ("A2_aligned_auto", "A4_time_shifted_auto")]:
        for label in labels:
            aa = {r["video_id"]: r["window_ap"] for r in per_video if r["arm"] == arm and r["label"] == label}
            bb = {r["video_id"]: r["window_ap"] for r in per_video if r["arm"] == baseline and r["label"] == label}
            delta = np.asarray([aa[v] - bb[v] for v in aa if aa[v] is not None and bb[v] is not None])
            rng = np.random.default_rng(42)
            boot = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
            comparisons.append({"arm": arm, "baseline": baseline, "label": str(label), "videos": len(delta),
                                "mean_video_window_ap_delta": float(delta.mean()),
                                "conditional_video_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist()})
    report = {"status": "completed exploratory automatic-evidence experiment", "human_oracle": False,
              "protocol": protocol, "results": summaries, "paired_comparisons": comparisons,
              "all_fits_converged": all(r["converged"] for r in fold_reports),
              "limitations": ["Internal-val15 was historically used for E16 model selection; this is not an independent final benchmark.",
                              "Within each fold, fit/calibration/heldout videos are disjoint; preprocessing fits on fit videos only.",
                              "Labels and original scores come from the frozen E16 cache, not new manual trajectory GT.",
                              "Only score ranking is changed; event candidate times remain frozen.",
                              "Evidence sampling is reconstructed from source fps, not saved decoder timestamps.",
                              "Bootstrap intervals condition on fitted overlapping CV models; no population-level guarantee.",
                              "Aligned arm jointly changes lookup and identity-aware kinematics; gains cannot be attributed to one alone.",
                              "No direct deployment or strict-O2 claim; no O1 job/index changed."]}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    lines = ["自动证据增强对照：内部验证集分组探索", "",
             "不是人工准确轨迹 O2；不是 train132→val15 的正式泛化结果。",
             "使用 E16 的 11,537 个缓存窗口；5 折，每折 9 视频训练、3 视频定阈值、3 视频评测。",
             "L2 logistic C=0.1，无超参数搜索；只改变分数，不改变候选时间。", "",
             "|组|类别|视频平均窗口 AP|事件 P|事件 R|FP/90min|", "|---|---|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"|{r['arm']}|{r['label']}|{r['mean_video_window_ap']:.2%}|{r['precision']:.2%}|{r['recall']:.2%}|{r['fp_per_90min']:.1f}|")
    lines += ["", "所有阈值仅在每折校准视频选择；评测召回不足时不得当作同召回 precision 提升。",
              "组含义：E16_fixed 原分数；O0_score_readout 仅分数学习；O1_old_auto 加旧证据；",
              "A2_aligned_auto 加时间对齐和身份一致运动特征；A3_quality_only 仅质量/可用性特征；",
              "A4_time_shifted_auto 同视频时间移位证据，对照场景统计捷径。",
              "配对差异和条件视频 bootstrap 区间见 summary.json。有效结论仅限该轻量读出、当前标签与内部视频。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("COMPLETE", str(output), flush=True)


if __name__ == "__main__":
    main()
