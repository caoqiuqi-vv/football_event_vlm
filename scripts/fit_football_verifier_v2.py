#!/usr/bin/env python
"""Fit Verifier V2: per-class candidate rerankers on the full166 dense cache.

Protocol (see docs/football_tail_first_fullchain_plan_20260910.md §3.5):
  - train split  : GroupKFold(5, by video) OOF, StandardScaler + LogisticRegression
                   (class_weight=balanced, C=0.1), per label; ignored candidates
                   (peak 3-8 s from GT) are excluded from training only.
  - feature ablation (OOF, event-level P at recall floor):
                   score / score+shape / score+shape+context / full(+audio)
  - thresholds   : chosen on val15 (max precision s.t. recall >= floor), frozen
  - transfer     : test18 scored by the final all-train model, evaluated at the
                   frozen thresholds against the repaired final_labels.json GT.

Evaluation protocol: window_overlap, identical to the 0909 reference run —
a candidate window [start-3s, end+3s] is TP if it covers >=1 GT (else FP);
recall counts unique GT covered; many predictions may cover the same GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

LABELS = ("shot", "save", "set_piece")
SET_PIECE_SOURCES = {"corner", "free_kick", "kickoff", "penalty", "set_piece"}
GT_EXCLUDE = {"back_pass", "throw_in"}
MATCH_TOLERANCE_SEC = 3.0
RECALL_FLOORS = {"shot": 0.85, "save": 0.80, "set_piece": 0.80}

SCORE_FEATURES = [
    "peak_prob", "peak_clip_prob", "peak_response_prob", "peak_global_prob",
    "logit_peak_prob", "logit_clip_prob", "logit_response_prob", "logit_global_prob",
    "other_max_prob", "other_margin", "video_median_logit",
]
SHAPE_FEATURES = [
    "prominence", "half_width_sec", "rise_slope_per_s", "fall_slope_per_s",
    "top2_mean", "plateau_sec",
]
CONTEXT_FEATURES = ["n_same_cand_pm15s", "n_cover_windows_prob03"]
SAVE_CONTEXT_FEATURES = ["shot_ctx_max_6s", "shot_ctx_max_12s", "time_since_last_shot_cand"]
AUDIO_FEATURES = [
    "audio_available", "audio_whistle_peakiness_max", "audio_whistle_peakiness_mean",
    "audio_whistle_activity_flag", "audio_log_energy_mean",
]
ABLATIONS = ["score", "score+shape", "score+shape+context", "full"]

REFERENCE_OPERATING_POINT = {
    "shot": {"precision": 0.412, "recall": 0.802},
    "save": {"precision": 0.285, "recall": 0.751},
    "set_piece": {"precision": 0.252, "recall": 0.865},
}


@dataclass(frozen=True)
class Candidate:
    video_id: str
    split: str
    label: str
    window_index: int
    peak_time_sec: float
    peak_score: float
    is_ignored: bool
    target: int | None
    features: dict[str, float]

    @property
    def key(self) -> tuple[str, str, int]:
        return self.video_id, self.label, self.window_index


def parse_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_candidates(path: Path) -> list[Candidate]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    meta_cols = {
        "video_id", "split", "label", "window_index", "peak_time_sec", "peak_score",
        "nearest_gt_distance_sec", "is_ignored", "target",
    }
    rows: list[Candidate] = []
    for raw in raw_rows:
        target_raw = raw.get("target", "")
        target = int(target_raw) if target_raw not in ("", None) else None
        rows.append(
            Candidate(
                video_id=raw["video_id"],
                split=raw["split"],
                label=raw["label"],
                window_index=int(float(raw["window_index"])),
                peak_time_sec=float(raw["peak_time_sec"]),
                peak_score=float(raw["peak_score"]),
                is_ignored=raw.get("is_ignored", "0") == "1",
                target=target,
                features={k: parse_float(v) for k, v in raw.items() if k not in meta_cols},
            )
        )
    return rows


def project_gt_label(raw: str) -> str | None:
    label = raw.strip()
    if label in ("shot", "save"):
        return label
    if label in SET_PIECE_SOURCES:
        return "set_piece"
    return None


def load_run_gt(run_dir: Path, video_ids: Sequence[str]) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for video_id in video_ids:
        gt: dict[str, list[float]] = {label: [] for label in LABELS}
        path = run_dir / video_id / "gt_events.csv"
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    label = project_gt_label(row.get("label", ""))
                    if label is not None:
                        gt[label].append(float(row["time_sec"]))
        result[video_id] = gt
    return result


def load_test_gt(final_labels_json: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(final_labels_json.read_text(encoding="utf-8"))
    result: dict[str, dict[str, list[float]]] = {}
    for video_id, record in payload["videos"].items():
        gt: dict[str, list[float]] = {label: [] for label in LABELS}
        for event in record.get("events", []):
            raw = event.get("semantic_label") or event.get("label") or ""
            if raw in GT_EXCLUDE:
                continue
            label = project_gt_label(raw)
            if label is not None:
                gt[label].append(float(event["time_sec"]))
        result[video_id] = gt
    return result


def load_durations(run_dir: Path, video_ids: Sequence[str]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for video_id in video_ids:
        path = run_dir / video_id / "summary.json"
        if path.is_file():
            durations[video_id] = float(json.loads(path.read_text()).get("duration_sec", 0.0))
    return durations


def features_for(label: str, ablation: str, audio_ok: bool) -> list[str]:
    names = list(SCORE_FEATURES)
    if ablation in ("score+shape", "score+shape+context", "full"):
        names += SHAPE_FEATURES
    if ablation in ("score+shape+context", "full"):
        names += CONTEXT_FEATURES
        if label == "save":
            names += SAVE_CONTEXT_FEATURES
    if ablation == "full" and audio_ok:
        names += AUDIO_FEATURES
    return names


def build_matrix(
    rows: Sequence[Candidate], feature_names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [[row.features.get(name, float("nan")) for name in feature_names] for row in rows],
        dtype=np.float64,
    )


def impute_fit(x: np.ndarray, feature_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    medians = np.zeros(x.shape[1], dtype=np.float64)
    keep: list[int] = []
    for j in range(x.shape[1]):
        col = x[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        medians[j] = float(np.median(col[finite]))
        keep.append(j)
    return np.asarray(keep, dtype=np.int64), medians, [feature_names[j] for j in keep]


def impute_apply(x: np.ndarray, keep: np.ndarray, medians: np.ndarray) -> np.ndarray:
    x = x[:, keep]
    return np.where(np.isfinite(x), x, medians[keep][None, :])


def fit_oof(
    rows: Sequence[Candidate],
    feature_names: Sequence[str],
    *,
    folds: int,
    regularization_c: float,
) -> dict[tuple[str, str, int], float]:
    scores: dict[tuple[str, str, int], float] = {}
    groups = np.asarray([row.video_id for row in rows])
    split_count = min(max(2, folds), len(np.unique(groups)))
    x = build_matrix(rows, feature_names)
    y = np.asarray([-1 if row.target is None else row.target for row in rows], dtype=np.int64)
    for train_idx, val_idx in GroupKFold(n_splits=split_count).split(x, y, groups):
        fit_idx = train_idx[y[train_idx] >= 0]
        if len(np.unique(y[fit_idx])) < 2:
            raise ValueError("fold lacks both classes")
        keep, medians, _ = impute_fit(x[fit_idx], feature_names)
        scaler = StandardScaler().fit(impute_apply(x[fit_idx], keep, medians))
        model = LogisticRegression(
            class_weight="balanced", C=regularization_c, max_iter=2000, random_state=42
        ).fit(scaler.transform(impute_apply(x[fit_idx], keep, medians)), y[fit_idx])
        pred = model.predict_proba(scaler.transform(impute_apply(x[val_idx], keep, medians)))[:, 1]
        for index, value in zip(val_idx, pred):
            scores[rows[int(index)].key] = float(value)
    return scores


def fit_final(
    rows: Sequence[Candidate], feature_names: Sequence[str], regularization_c: float
) -> dict[str, Any]:
    fit_rows = [row for row in rows if row.target is not None]
    x = build_matrix(fit_rows, feature_names)
    y = np.asarray([int(row.target) for row in fit_rows], dtype=np.int64)
    keep, medians, kept_names = impute_fit(x, feature_names)
    scaler = StandardScaler().fit(impute_apply(x, keep, medians))
    model = LogisticRegression(
        class_weight="balanced", C=regularization_c, max_iter=2000, random_state=42
    ).fit(scaler.transform(impute_apply(x, keep, medians)), y)
    return {
        "feature_names": list(kept_names),
        "keep_indices": keep.tolist(),
        "impute_medians": medians.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "num_samples": len(fit_rows),
        "num_positive": int(y.sum()),
        "num_negative": int((1 - y).sum()),
    }


def apply_model(
    rows: Sequence[Candidate], feature_names: Sequence[str], model: dict[str, Any]
) -> dict[tuple[str, str, int], float]:
    x = build_matrix(rows, feature_names)
    keep = np.asarray(model["keep_indices"], dtype=np.int64)
    medians = np.asarray(model["impute_medians"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    z = (impute_apply(x, keep, medians) - mean[None, :]) / np.where(scale == 0.0, 1.0, scale)[None, :]
    logits = np.clip(z @ coef + model["intercept"], -30.0, 30.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return {row.key: float(prob) for row, prob in zip(rows, probs)}


# ---------------------------------------------------------------------------
# evaluation protocol: window_overlap (same as the 0909 reference run).
# A candidate window [start-tol, end+tol] is TP if it covers >=1 GT, else FP;
# recall counts unique GT covered by >=1 selected window; many predictions may
# cover the same GT.  tol=3 s on 10 s windows -> match iff |center-gt| <= 8 s.
# ---------------------------------------------------------------------------

WINDOW_HALF_SEC = 5.0  # full166 windows are 10 s; candidate time = window center


def exact_curve(
    rows: Sequence[Candidate],
    scores: dict[tuple[str, str, int], float],
    gt_by_video: dict[str, dict[str, list[float]]],
    label: str,
    video_ids: Sequence[str],
    tol: float = MATCH_TOLERANCE_SEC,
) -> list[dict[str, float]]:
    entries: list[tuple[float, bool, tuple[tuple[str, int], ...]]] = []
    total_gt = sum(len(gt_by_video.get(vid, {}).get(label, [])) for vid in video_ids)
    for row in rows:
        if row.label != label:
            continue
        gts = gt_by_video.get(row.video_id, {}).get(label, [])
        lo = row.peak_time_sec - WINDOW_HALF_SEC - tol
        hi = row.peak_time_sec + WINDOW_HALF_SEC + tol
        matched = tuple(
            (row.video_id, gi) for gi, g in enumerate(gts) if lo <= g <= hi
        )
        entries.append((scores[row.key], bool(matched), matched))
    entries.sort(key=lambda item: -item[0])
    curve: list[dict[str, float]] = []
    tp = fp = 0
    matched_gt: set[tuple[str, int]] = set()
    i = 0
    while i < len(entries):
        score = entries[i][0]
        j = i
        while j < len(entries) and entries[j][0] == score:
            _, hit, matched = entries[j]
            if hit:
                tp += 1
                matched_gt.update(matched)
            else:
                fp += 1
            j += 1
        recall = len(matched_gt) / total_gt if total_gt else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        curve.append(
            {
                "threshold": float(score),
                "tp": tp,
                "fp": fp,
                "fn": total_gt - len(matched_gt),
                "precision": precision,
                "recall": recall,
            }
        )
        i = j
    return curve


def eval_frozen(
    rows: Sequence[Candidate],
    scores: dict[tuple[str, str, int], float],
    gt_by_video: dict[str, dict[str, list[float]]],
    label: str,
    video_ids: Sequence[str],
    threshold: float,
) -> dict[str, float]:
    curve = exact_curve(rows, scores, gt_by_video, label, video_ids)
    eligible = [p for p in curve if p["threshold"] >= threshold]
    if eligible:
        return dict(eligible[-1])
    return {
        "threshold": float(threshold), "tp": 0, "fp": 0,
        "fn": sum(len(gt_by_video.get(v, {}).get(label, [])) for v in video_ids),
        "precision": 0.0, "recall": 0.0,
    }


def select_threshold(
    rows: Sequence[Candidate],
    scores: dict[tuple[str, str, int], float],
    gt_by_video: dict[str, dict[str, list[float]]],
    label: str,
    video_ids: Sequence[str],
    recall_floor: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    curve = exact_curve(rows, scores, gt_by_video, label, video_ids)
    feasible = [p for p in curve if p["recall"] + 1e-12 >= recall_floor]
    if feasible:
        best = max(feasible, key=lambda p: (p["precision"], p["recall"], p["threshold"]))
        best = {**best, "recall_shortfall": False}
    else:
        best = max(curve, key=lambda p: (p["recall"], p["precision"], p["threshold"]))
        best = {**best, "recall_shortfall": True}
    return best, curve


ADAPTIVE_ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def prob_logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def adaptive_scores(
    rows: Sequence[Candidate],
    scores: dict[tuple[str, str, int], float],
    alpha: float,
) -> dict[tuple[str, str, int], float]:
    """Per-video median-logit shift (unlabeled adaptation, 0909 protocol style)."""
    logits_by_video: dict[str, list[float]] = {}
    for row in rows:
        logits_by_video.setdefault(row.video_id, []).append(prob_logit(scores[row.key]))
    medians = {
        vid: float(np.median(values)) for vid, values in logits_by_video.items()
    }
    return {
        row.key: prob_logit(scores[row.key]) - alpha * medians[row.video_id]
        for row in rows
    }


def precision_at_recall(
    curve: list[dict[str, float]], recall: float
) -> dict[str, float]:
    feasible = [p for p in curve if p["recall"] + 1e-12 >= recall]
    if not feasible:
        return dict(curve[-1]) if curve else {}
    return dict(max(feasible, key=lambda p: (p["precision"], p["threshold"])))


def metric_short(point: dict[str, float]) -> dict[str, float]:
    p = point["precision"]
    r = point["recall"]
    return {
        "threshold": point["threshold"],
        "tp": point["tp"],
        "fp": point["fp"],
        "fn": point["fn"],
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
    }


def fp_per_90min(fp: float, video_ids: Sequence[str], durations: dict[str, float]) -> float:
    total = sum(durations.get(vid, 0.0) for vid in video_ids)
    return fp / (total / 5400.0) if total else 0.0


def fmt_pr(point: dict[str, float]) -> str:
    return f"P {point['precision']:.3f} / R {point['recall']:.3f} / F1 {point['f1']:.3f}"


def write_report_md(path: Path, report: dict[str, Any], audio_ok: bool) -> None:
    lines: list[str] = []
    lines.append("# Verifier V2 — full166 候选级重排序器报告")
    lines.append("")
    lines.append("日期: 2026-09-10")
    lines.append("")
    lines.append("## 方法")
    lines.append("")
    lines.append("- 候选: full166 稠密缓存 `prob_<label>` 曲线上 prob≥0.05 的窗口"
                 "(类内 NMS 半径 shot/save 3s、set_piece 5s 在 5s stride 网格上不构成抑制,"
                 "即所有过阈窗口均为候选)。")
    lines.append("- 评估协议: window_overlap(与 0909 参考运行同口径)——候选窗 [start-3s,end+3s]"
                 " 覆盖任一同类 GT 记 TP,否则 FP;recall 统计被 ≥1 个选中窗覆盖的 unique GT;"
                 "同一 GT 允许多个预测命中。")
    lines.append("- 特征: 分数系(prob/clip/response/global + logit + 跨类 max/margin + 视频级 median logit)、"
                 "峰形系(prominence/半峰宽/升降斜率/top2 均值/平台时长)、"
                 "时序上下文系(±15s 同类候选数、prob>0.3 覆盖窗数;save 类另有过去 6s/12s 最大 shot prob、"
                 "距上个 shot>0.3 候选时间)、音频系(哨声 peakiness [-5s,+10s] max/mean、activity 标志、log_energy 均值)。")
    lines.append("- 训练: train129 GroupKFold(5, by video) OOF + StandardScaler + "
                 "LogisticRegression(class_weight=balanced, C=0.1),逐类独立;"
                 "is_ignored(峰距 GT 3~8s)候选不参与训练,但参与事件级评估。")
    lines.append("- 定阈: val15 上逐类 max precision s.t. recall≥floor(0.85/0.80/0.80),冻结迁移 test18(±3s 匹配)。")
    lines.append("")
    lines.append("## 数据规模")
    lines.append("")
    data = report["data"]
    lines.append(f"- 视频数: train {data['train_videos']} / val15 {data['val_videos']} / test18 {data['test_videos']}")
    lines.append(f"- 候选数: train {data['train_candidates']} / val15 {data['val_candidates']} / test18 {data['test_candidates']}")
    lines.append(f"- GT 事件数(粗类投影后): `{json.dumps(data['gt_counts'], ensure_ascii=False)}`")
    lines.append(f"- 音频组启用: {audio_ok}(train 候选音频覆盖率 {report['audio_status']['train_audio_candidates_frac']:.2%})")
    lines.append("")
    lines.append("## OOF 特征组消融(train129,阈值在 OOF 曲线上按 recall floor 选)")
    lines.append("")
    lines.append("| label | 特征组 | P | R | F1 | thr |")
    lines.append("|---|---|---|---|---|---|")
    for label in LABELS:
        for group, res in report["oof_ablation"][label].items():
            if "skipped" in res:
                lines.append(f"| {label} | {group} | skipped({res['skipped']}) | | | |")
                continue
            mark = " ⚠shortfall" if res.get("recall_shortfall") else ""
            lines.append(
                f"| {label} | {group} | {res['precision']:.4f} | {res['recall']:.4f} "
                f"| {res['f1']:.4f} | {res['threshold']:.4f}{mark} |"
            )
    lines.append("")
    lines.append("## val15(定阈集,数字仅作过程参考)")
    lines.append("")
    lines.append("| label | verifier P/R/F1 | baseline(peak_score) P/R/F1 | thr | FP/90min v/b |")
    lines.append("|---|---|---|---|---|")
    for label in LABELS:
        v = report["val15"][label]
        lines.append(
            f"| {label} | {fmt_pr(v['verifier'])} | {fmt_pr(v['baseline_peak_score'])} | "
            f"{report['thresholds'][label]:.4f} | {v['verifier_fp_per_90min']:.1f}/{v['baseline_fp_per_90min']:.1f} |"
        )
    lines.append("")
    lines.append("## test18(冻结迁移)")
    lines.append("")
    lines.append("| label | verifier P/R/F1 | FP/90min | baseline(peak_score, val15 冻结) P/R/F1 | baseline FP/90min | baseline@test18 同 recall P(诊断) | 0909 运行点 P/R |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in LABELS:
        t = report["test18"][label]
        ref = t["reference_operating_point_20260909"]
        lines.append(
            f"| {label} | {fmt_pr(t['verifier'])} | {t['verifier_fp_per_90min']:.1f} | "
            f"{fmt_pr(t['baseline_peak_score_frozen'])} | {t['baseline_peak_score_frozen_fp_per_90min']:.1f} | "
            f"{t['baseline_peak_score_matched_recall_on_test18']['precision']:.3f}"
            f"@R{t['baseline_peak_score_matched_recall_on_test18']['recall']:.3f} | "
            f"{ref['precision']:.3f}/{ref['recall']:.3f} |"
        )
    lines.append("")
    lines.append("## test18 自适应迁移(per-video median-logit 平移,alpha 在 val15 上选;无标签适配,0909 协议同款)")
    lines.append("")
    lines.append("| label | verifier alpha | verifier P/R/F1 | FP/90min | baseline alpha | baseline P/R/F1 | FP/90min |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in LABELS:
        t = report["test18"][label]
        va, ba = t["verifier_adaptive"], t["baseline_peak_score_adaptive"]
        lines.append(
            f"| {label} | {va['alpha']} | {fmt_pr(va)} | {va['fp_per_90min']:.1f} | "
            f"{ba['alpha']} | {fmt_pr(ba)} | {ba['fp_per_90min']:.1f} |"
        )
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    goal_hits = []
    for label in LABELS:
        t = report["test18"][label]
        v, b = t["verifier"], t["baseline_peak_score_frozen"]
        va, ba = t["verifier_adaptive"], t["baseline_peak_score_adaptive"]
        ref = t["reference_operating_point_20260909"]
        d_frozen = (v["precision"] - b["precision"]) * 100
        d_adapt = (va["precision"] - ba["precision"]) * 100
        d_ref = (va["precision"] - ref["precision"]) * 100
        hit = d_ref >= 5.0 and va["recall"] + 1e-12 >= ref["recall"]
        goal_hits.append(hit)
        lines.append(
            f"- {label}: 冻结全局阈值下 verifier 相对 baseline ΔP={d_frozen:+.1f}pp;"
            f"自适应迁移下 ΔP={d_adapt:+.1f}pp(R {va['recall']:.3f} vs {ba['recall']:.3f});"
            f"相对 0909 运行点 ΔP={d_ref:+.1f}pp / ΔR={(va['recall']-ref['recall'])*100:+.1f}pp;"
            f" +5pp 且不掉 recall: {'达成' if hit else '未达成'}。"
        )
    lines.append("")
    lines.append(
        f"总体判定: {'三类全部达成' if all(goal_hits) else '未达成(见逐类)'}。"
        " 目标定义: test18 上 P 相对 0909 运行点 ≥+5pp 且 R 不低于该运行点(同 window_overlap tol=3s 口径)。"
    )
    lines.append("")
    lines.append("## 卫生说明")
    lines.append("")
    lines.append("- test18 视频绝不出现在训练 fold(脚本内置 split 交集断言);val15 仅用于定阈。")
    lines.append("- **test18 已非干净 holdout**(历史上多轮选模/评估用过),以上数字为同口径对比,不作泛化保证。")
    lines.append("- 数值稳定: logit 裁剪 ±30;NaN 音频特征用训练 fold 中位数填充 + audio_available 指示列。")
    lines.append("- 0909 运行点与本报告同为 window_overlap tol=3s 口径;差别在其阈值为"
                 " per-video adaptive median-logit 且经 LOOV 选 alpha。本报告主表为 val15 冻结全局阈值,"
                 "附表为同风格自适应阈值变体(alpha 网格 {0,0.5,1,1.5,2,3,4} 在 val15 pooled 上选)。"
                 "受控对比看 baseline(peak_score) 列。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--test-gt-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_candidates(args.candidates_csv)
    rows = [row for row in rows if row.split in ("train", "val15", "test18")]
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val15"]
    test_rows = [row for row in rows if row.split == "test18"]
    train_videos = sorted({row.video_id for row in train_rows})
    val_videos = sorted({row.video_id for row in val_rows})
    test_videos = sorted({row.video_id for row in test_rows})
    leakage = set(train_videos) & (set(val_videos) | set(test_videos))
    if leakage:
        raise ValueError(f"split leakage: {sorted(leakage)}")

    run_gt = load_run_gt(args.run_dir, train_videos + val_videos)
    test_gt = load_test_gt(args.test_gt_json)
    gt_all = {**run_gt, **{vid: test_gt.get(vid, {l: [] for l in LABELS}) for vid in test_videos}}
    durations = load_durations(args.run_dir, train_videos + val_videos + test_videos)

    train_audio_available = [
        row.features.get("audio_available", 0.0) for row in train_rows
    ]
    audio_ok = bool(train_audio_available) and sum(train_audio_available) > 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "params": {
            "folds": args.folds,
            "regularization_c": args.regularization_c,
            "recall_floors": RECALL_FLOORS,
            "match_tolerance_sec": MATCH_TOLERANCE_SEC,
            "candidates_csv": str(args.candidates_csv),
        },
        "data": {
            "train_videos": len(train_videos),
            "val_videos": len(val_videos),
            "test_videos": len(test_videos),
            "train_candidates": len(train_rows),
            "val_candidates": len(val_rows),
            "test_candidates": len(test_rows),
            "gt_counts": {
                split: {
                    label: sum(len(gt_all.get(vid, {}).get(label, [])) for vid in vids)
                    for label in LABELS
                }
                for split, vids in (
                    ("train", train_videos), ("val15", val_videos), ("test18", test_videos)
                )
            },
        },
        "audio_status": {
            "train_audio_candidates_frac": (
                sum(train_audio_available) / len(train_audio_available)
                if train_audio_available
                else 0.0
            ),
            "audio_group_enabled": audio_ok,
        },
        "hygiene": {
            "test18_in_train_folds": False,
            "val15_usage": "threshold selection only",
            "test18_note": "test18 is NOT a clean holdout (historical model selection rounds); numbers are same-protocol comparisons",
        },
    }

    # ---------- OOF feature-group ablation on train ----------
    ablation_results: dict[str, dict[str, Any]] = {}
    oof_scores_full: dict[tuple[str, str, int], float] = {}
    final_feature_names: dict[str, list[str]] = {}
    for label in LABELS:
        label_rows = [row for row in train_rows if row.label == label]
        ablation_results[label] = {}
        for ablation in ABLATIONS:
            feature_names = features_for(label, ablation, audio_ok)
            if ablation == "full" and not audio_ok:
                ablation_results[label][ablation] = {"skipped": "audio features unavailable"}
                continue
            scores = fit_oof(
                label_rows, feature_names,
                folds=args.folds, regularization_c=args.regularization_c,
            )
            best, _ = select_threshold(
                label_rows, scores, gt_all, label, train_videos, RECALL_FLOORS[label]
            )
            ablation_results[label][ablation] = {
                **metric_short(best),
                "recall_shortfall": bool(best.get("recall_shortfall", False)),
                "n_features": len(feature_names),
            }
            if ablation == "full" or (ablation == "score+shape+context" and not audio_ok):
                oof_scores_full.update(scores)
                final_feature_names[label] = feature_names
        base_scores = {row.key: row.peak_score for row in label_rows}
        base_best, _ = select_threshold(
            label_rows, base_scores, gt_all, label, train_videos, RECALL_FLOORS[label]
        )
        ablation_results[label]["baseline_peak_score"] = {
            **metric_short(base_best),
            "recall_shortfall": bool(base_best.get("recall_shortfall", False)),
        }
    report["oof_ablation"] = ablation_results

    # ---------- final models on all train rows ----------
    models: dict[str, Any] = {}
    for label in LABELS:
        label_rows = [row for row in train_rows if row.label == label]
        models[label] = fit_final(
            label_rows, final_feature_names[label], args.regularization_c
        )
    report["final_models"] = {
        label: {
            "feature_names": models[label]["feature_names"],
            "num_samples": models[label]["num_samples"],
            "num_positive": models[label]["num_positive"],
            "num_negative": models[label]["num_negative"],
        }
        for label in LABELS
    }

    # ---------- val15 threshold selection + frozen test18 transfer ----------
    val_eval: dict[str, Any] = {}
    test_eval: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    test_scores_by_label: dict[str, dict[tuple[str, str, int], float]] = {}
    for label in LABELS:
        vrows = [row for row in val_rows if row.label == label]
        trows = [row for row in test_rows if row.label == label]
        feature_names = final_feature_names[label]
        vscores = apply_model(vrows, feature_names, models[label])
        tscores = apply_model(trows, feature_names, models[label])
        test_scores_by_label[label] = tscores

        best, _ = select_threshold(
            vrows, vscores, gt_all, label, val_videos, RECALL_FLOORS[label]
        )
        thresholds[label] = float(best["threshold"])
        vb_scores = {row.key: row.peak_score for row in vrows}
        vb_best, _ = select_threshold(
            vrows, vb_scores, gt_all, label, val_videos, RECALL_FLOORS[label]
        )
        val_eval[label] = {
            "verifier": {**metric_short(best), "recall_shortfall": bool(best.get("recall_shortfall", False))},
            "baseline_peak_score": {
                **metric_short(vb_best),
                "recall_shortfall": bool(vb_best.get("recall_shortfall", False)),
            },
            "verifier_fp_per_90min": fp_per_90min(best["fp"], val_videos, durations),
            "baseline_fp_per_90min": fp_per_90min(vb_best["fp"], val_videos, durations),
        }

        tpoint = eval_frozen(trows, tscores, gt_all, label, test_videos, thresholds[label])
        tb_scores = {row.key: row.peak_score for row in trows}
        tb_frozen = eval_frozen(trows, tb_scores, gt_all, label, test_videos, float(vb_best["threshold"]))
        _, tb_curve = select_threshold(
            trows, tb_scores, gt_all, label, test_videos, 0.0
        )
        tb_matched = precision_at_recall(tb_curve, tpoint["recall"])
        test_eval[label] = {
            "verifier": metric_short(tpoint),
            "verifier_fp_per_90min": fp_per_90min(tpoint["fp"], test_videos, durations),
            "baseline_peak_score_frozen": metric_short(tb_frozen),
            "baseline_peak_score_frozen_fp_per_90min": fp_per_90min(tb_frozen["fp"], test_videos, durations),
            "baseline_peak_score_matched_recall_on_test18": {
                **metric_short(tb_matched),
                "note": "oracle-matched on test18, diagnostic only",
            },
            "reference_operating_point_20260909": REFERENCE_OPERATING_POINT[label],
        }

        # adaptive per-video median-logit transfer (unlabeled; alpha picked on val15)
        for name, vsc, tsc in (
            ("verifier_adaptive", vscores, tscores),
            ("baseline_peak_score_adaptive", vb_scores, tb_scores),
        ):
            chosen: tuple[tuple, float, dict[str, float]] | None = None
            for alpha in ADAPTIVE_ALPHAS:
                va = adaptive_scores(vrows, vsc, alpha)
                sel, _ = select_threshold(
                    vrows, va, gt_all, label, val_videos, RECALL_FLOORS[label]
                )
                key = (
                    not sel["recall_shortfall"],
                    sel["precision"],
                    sel["recall"],
                    -alpha,
                )
                if chosen is None or key > chosen[0]:
                    chosen = (key, alpha, sel)
            assert chosen is not None
            _, alpha_sel, val_sel = chosen
            ta = adaptive_scores(trows, tsc, alpha_sel)
            t_adapt = eval_frozen(
                trows, ta, gt_all, label, test_videos, float(val_sel["threshold"])
            )
            val_eval[label][name] = {
                **metric_short(val_sel),
                "recall_shortfall": bool(val_sel.get("recall_shortfall", False)),
                "alpha": alpha_sel,
            }
            test_eval[label][name] = {
                **metric_short(t_adapt),
                "alpha": alpha_sel,
                "fp_per_90min": fp_per_90min(t_adapt["fp"], test_videos, durations),
            }
    report["thresholds"] = thresholds
    report["val15"] = val_eval
    report["test18"] = test_eval

    # ---------- artifacts ----------
    verifier_artifact = {
        "version": 2,
        "labels": list(LABELS),
        "recall_floors": RECALL_FLOORS,
        "match_tolerance_sec": MATCH_TOLERANCE_SEC,
        "thresholds": thresholds,
        "adaptive_thresholds": {
            label: {
                "alpha": test_eval[label]["verifier_adaptive"]["alpha"],
                "threshold": test_eval[label]["verifier_adaptive"]["threshold"],
            }
            for label in LABELS
        },
        "models": models,
        "final_feature_names": final_feature_names,
        "train_video_count": len(train_videos),
        "audio_group_enabled": audio_ok,
    }
    (args.output_dir / "verifier.json").write_text(json.dumps(verifier_artifact, indent=2) + "\n")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report_md(args.output_dir / "REPORT.md", report, audio_ok)

    with (args.output_dir / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["video_id", "split", "label", "window_index", "peak_time_sec",
             "peak_score", "target", "is_ignored", "oof_score"]
        )
        for row in train_rows:
            writer.writerow([
                row.video_id, row.split, row.label, row.window_index, row.peak_time_sec,
                row.peak_score, "" if row.target is None else row.target,
                int(row.is_ignored), oof_scores_full.get(row.key, ""),
            ])
    with (args.output_dir / "test18_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["video_id", "label", "window_index", "peak_time_sec", "peak_score",
             "target", "is_ignored", "verifier_score", "selected"]
        )
        for row in test_rows:
            score = test_scores_by_label[row.label][row.key]
            writer.writerow([
                row.video_id, row.label, row.window_index, row.peak_time_sec,
                row.peak_score, "" if row.target is None else row.target,
                int(row.is_ignored), score, int(score >= thresholds[row.label]),
            ])

    for label in LABELS:
        v = val_eval[label]["verifier"]
        b = val_eval[label]["baseline_peak_score"]
        t = test_eval[label]["verifier"]
        tb = test_eval[label]["baseline_peak_score_frozen"]
        ta = test_eval[label]["verifier_adaptive"]
        tba = test_eval[label]["baseline_peak_score_adaptive"]
        print(
            f"{label}: val15 verifier P/R={v['precision']:.3f}/{v['recall']:.3f} "
            f"(baseline {b['precision']:.3f}/{b['recall']:.3f}) | "
            f"test18 verifier P/R={t['precision']:.3f}/{t['recall']:.3f} "
            f"(baseline frozen {tb['precision']:.3f}/{tb['recall']:.3f}) | "
            f"adaptive verifier P/R={ta['precision']:.3f}/{ta['recall']:.3f} "
            f"(baseline adaptive {tba['precision']:.3f}/{tba['recall']:.3f})"
        )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
