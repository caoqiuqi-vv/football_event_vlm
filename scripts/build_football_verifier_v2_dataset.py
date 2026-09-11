#!/usr/bin/env python
"""Build the Verifier V2 candidate dataset from a full166 dense cache.

For every video and every coarse label, candidate peaks are extracted from the
fused window score curve (``prob_<label>``): every window with prob >= 0.05
enters greedy score-descending NMS at the per-class radius (shot/save 3 s,
set_piece 5 s).  Candidates are featurized purely from the window score time
series plus optional 5 Hz audio side-channel features.  Targets follow the V1
convention: a peak within ``match_tolerance_sec`` of a same-class GT event is
positive, beyond ``ignore_radius_sec`` is negative, in between is dropped
(``is_ignored=1``).

Output: ``candidates.csv`` (one row per candidate peak) + ``summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

LABELS = ("shot", "save", "set_piece")
SET_PIECE_SOURCES = {"corner", "free_kick", "kickoff", "penalty", "set_piece"}
GT_EXCLUDE = {"back_pass", "throw_in"}
PEAK_MIN_SCORE = 0.05
NMS_RADIUS_SEC = {"shot": 3.0, "save": 3.0, "set_piece": 5.0}
MATCH_TOLERANCE_SEC = 3.0
IGNORE_RADIUS_SEC = 8.0
STRIDE_FALLBACK_SEC = 5.0
LOGIT_EPS = 1e-9
LOGIT_CLIP = 30.0

SCORE_FEATURE_NAMES = [
    "peak_prob", "peak_clip_prob", "peak_response_prob", "peak_global_prob",
    "logit_peak_prob", "logit_clip_prob", "logit_response_prob", "logit_global_prob",
    "other_max_prob", "other_margin", "video_median_logit",
]
SHAPE_FEATURE_NAMES = [
    "prominence", "half_width_sec", "rise_slope_per_s", "fall_slope_per_s",
    "top2_mean", "plateau_sec",
]
CONTEXT_FEATURE_NAMES = [
    "n_same_cand_pm15s", "n_cover_windows_prob03",
    "shot_ctx_max_6s", "shot_ctx_max_12s", "time_since_last_shot_cand",
]
AUDIO_FEATURE_NAMES = [
    "audio_available", "audio_whistle_peakiness_max", "audio_whistle_peakiness_mean",
    "audio_whistle_activity_flag", "audio_log_energy_mean",
]
FEATURE_NAMES = (
    SCORE_FEATURE_NAMES + SHAPE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES + AUDIO_FEATURE_NAMES
)


def logit(prob: float) -> float:
    prob = min(max(float(prob), LOGIT_EPS), 1.0 - LOGIT_EPS)
    return max(-LOGIT_CLIP, min(LOGIT_CLIP, math.log(prob / (1.0 - prob))))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def project_gt_label(raw: str) -> str | None:
    label = raw.strip()
    if label in ("shot", "save"):
        return label
    if label in SET_PIECE_SOURCES:
        return "set_piece"
    return None


def load_gt_from_run_dir(video_dir: Path) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {label: [] for label in LABELS}
    for row in read_csv(video_dir / "gt_events.csv"):
        label = project_gt_label(row.get("label", ""))
        if label is not None:
            result[label].append(float(row["time_sec"]))
    return result


def load_test_gt(final_labels_json: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(final_labels_json.read_text(encoding="utf-8"))
    result: dict[str, dict[str, list[float]]] = {}
    for video_id, record in payload["videos"].items():
        gt: dict[str, list[float]] = {label: [] for label in LABELS}
        for event in record.get("events", []):
            raw = event.get("semantic_label") or event.get("label") or ""
            label = project_gt_label(raw)
            if label is None or raw in GT_EXCLUDE:
                continue
            gt[label].append(float(event["time_sec"]))
        result[video_id] = gt
    return result


def nms_peaks(
    peak_indices: list[int], values: np.ndarray, centers: np.ndarray, radius_sec: float
) -> list[int]:
    # Strict-inequality suppression: on the 5 s stride grid no two window
    # centers are ever < 3 s (shot/save) or < 5 s (set_piece) apart, so NMS is
    # a no-op and every window with prob >= PEAK_MIN_SCORE is kept.  A "<="
    # radius-5 s set_piece NMS was measured to destroy candidate coverage
    # (val15 ceiling 0.89 -> 0.54), breaking the 0.80 recall floor.
    kept: list[int] = []
    for index in sorted(peak_indices, key=lambda i: (-values[i], centers[i])):
        if any(abs(float(centers[index] - centers[other])) < radius_sec for other in kept):
            continue
        kept.append(index)
    return kept


def half_peak_width(centers: np.ndarray, values: np.ndarray, i: int) -> float:
    half = 0.5 * float(values[i])
    n = len(values)

    def crossing(j_high: int, j_low: int) -> float:
        v_high = float(values[j_high])
        v_low = float(values[j_low])
        if v_high <= v_low:
            return float(centers[j_low])
        frac = (v_high - half) / (v_high - v_low)
        return float(centers[j_high]) + frac * float(centers[j_low] - centers[j_high])

    left = i
    while left - 1 >= 0 and values[left - 1] >= half:
        left -= 1
    t_left = crossing(left, left - 1) if left > 0 else float(centers[0])
    right = i
    while right + 1 < n and values[right + 1] >= half:
        right += 1
    t_right = crossing(right, right + 1) if right < n - 1 else float(centers[-1])
    return max(t_right - t_left, 0.0)


class AudioIndex:
    def __init__(self, audio_dir: Path | None, whistle_dir: Path | None) -> None:
        self.audio_dir = audio_dir
        self.whistle_dir = whistle_dir
        self._cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        self._whistle_cache: dict[str, list[tuple[float, float]] | None] = {}

    def features(self, video_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        if video_id not in self._cache:
            self._cache[video_id] = None
            if self.audio_dir is not None:
                path = self.audio_dir / f"{video_id}.npz"
                if path.is_file():
                    try:
                        data = np.load(path, allow_pickle=False)
                        times = data["times"].astype(np.float64)
                        feats = data["feats"].astype(np.float64)
                        if len(times) > 0 and feats.shape == (len(times), 5):
                            self._cache[video_id] = (times, feats)
                    except Exception:
                        self._cache[video_id] = None
        return self._cache[video_id]

    def whistle_intervals(self, video_id: str) -> list[tuple[float, float]]:
        if video_id not in self._whistle_cache:
            intervals: list[tuple[float, float]] = []
            if self.whistle_dir is not None:
                path = self.whistle_dir / f"{video_id}_whistles.csv"
                if path.is_file():
                    for row in read_csv(path):
                        try:
                            intervals.append((float(row["start_sec"]), float(row["end_sec"])))
                        except (KeyError, ValueError):
                            continue
            self._whistle_cache[video_id] = intervals
        return self._whistle_cache[video_id]


def audio_features_for_peak(
    audio: AudioIndex, video_id: str, t: float
) -> dict[str, float]:
    loaded = audio.features(video_id)
    row = {
        "audio_available": 0.0,
        "audio_whistle_peakiness_max": float("nan"),
        "audio_whistle_peakiness_mean": float("nan"),
        "audio_whistle_activity_flag": 0.0,
        "audio_log_energy_mean": float("nan"),
    }
    if loaded is not None:
        times, feats = loaded
        row["audio_available"] = 1.0
        mask = (times >= t - 5.0) & (times <= t + 10.0)
        if mask.any():
            row["audio_whistle_peakiness_max"] = float(feats[mask, 0].max())
            row["audio_whistle_peakiness_mean"] = float(feats[mask, 0].mean())
            row["audio_log_energy_mean"] = float(feats[mask, 1].mean())
    for start, end in audio.whistle_intervals(video_id):
        if start - 5.0 <= t <= end + 10.0:
            row["audio_whistle_activity_flag"] = 1.0
            break
    return row


def build_video_rows(
    video_dir: Path,
    split: str,
    gt_by_label: dict[str, list[float]],
    audio: AudioIndex,
) -> list[dict[str, object]]:
    video_id = video_dir.name
    window_path = video_dir / "window_predictions.csv"
    if not window_path.is_file():
        print(f"  skip {video_id}: missing window_predictions.csv")
        return []
    windows = sorted(read_csv(window_path), key=lambda row: int(row["index"]))
    if not windows:
        return []
    starts = np.asarray([float(r["start_sec"]) for r in windows])
    ends = np.asarray([float(r["end_sec"]) for r in windows])
    centers = (starts + ends) * 0.5
    stride = float(np.median(np.diff(centers))) if len(centers) > 1 else STRIDE_FALLBACK_SEC
    if not math.isfinite(stride) or stride <= 0:
        stride = STRIDE_FALLBACK_SEC

    prob = {
        label: np.asarray([float(r[f"prob_{label}"]) for r in windows])
        for label in LABELS
    }
    aux = {
        name: {
            label: np.asarray([float(r[f"{name}_{label}"]) for r in windows])
            for label in LABELS
        }
        for name in ("clip_prob", "response_prob", "global_prob")
    }

    peaks_by_label: dict[str, list[int]] = {}
    for label in LABELS:
        # Candidate = window with prob >= PEAK_MIN_SCORE surviving greedy
        # score-desc NMS (strict <) at the class radius.  On the 5 s stride
        # grid the radii (3 s / 3 s / 5 s) never strictly suppress a neighbour,
        # so in practice every qualifying window becomes a candidate.
        qualifying = [i for i in range(len(centers)) if prob[label][i] >= PEAK_MIN_SCORE]
        peaks_by_label[label] = nms_peaks(
            qualifying, prob[label], centers, NMS_RADIUS_SEC[label]
        )
    shot_prominent_peaks = sorted(
        float(centers[i]) for i in peaks_by_label["shot"] if prob["shot"][i] > 0.3
    )

    rows: list[dict[str, object]] = []
    for label in LABELS:
        p = prob[label]
        video_median_logit = float(np.median([logit(v) for v in p])) if len(p) else 0.0
        other_labels = [other for other in LABELS if other != label]
        peak_times = [float(centers[i]) for i in peaks_by_label[label]]
        for i in peaks_by_label[label]:
            t = float(centers[i])
            peak = float(p[i])
            baseline_mask = (np.abs(centers - t) > 10.0) & (np.abs(centers - t) <= 30.0)
            baseline = (
                float(np.median(p[baseline_mask])) if baseline_mask.any() else float(np.median(p))
            )
            half = 0.5 * peak
            left = i
            while left - 1 >= 0 and p[left - 1] >= half:
                left -= 1
            right = i
            while right + 1 < len(p) and p[right + 1] >= half:
                right += 1
            plateau_sec = float((right - left + 1) * stride)
            neighbors = [
                float(p[j]) for j in (i - 1, i, i + 1) if 0 <= j < len(p)
            ]
            top2_mean = float(np.mean(sorted(neighbors, reverse=True)[:2]))
            rise = float(p[i] - p[i - 1]) / stride if i > 0 else 0.0
            fall = float(p[i] - p[i + 1]) / stride if i < len(p) - 1 else 0.0
            other_max = max(float(prob[o][i]) for o in other_labels)
            n_same = sum(
                1 for other_t in peak_times if other_t != t and abs(other_t - t) <= 15.0
            )
            n_cover = int(
                ((starts <= t) & (ends >= t) & (p > 0.3)).sum()
            )
            shot_p = prob["shot"]
            mask6 = (centers >= t - 6.0) & (centers <= t)
            mask12 = (centers >= t - 12.0) & (centers <= t)
            shot_ctx_max_6s = float(shot_p[mask6].max()) if mask6.any() else 0.0
            shot_ctx_max_12s = float(shot_p[mask12].max()) if mask12.any() else 0.0
            earlier = [st for st in shot_prominent_peaks if st < t]
            time_since_last_shot = float(t - earlier[-1]) if earlier else 999.0

            gt_times = gt_by_label.get(label, [])
            nearest = min((abs(t - g) for g in gt_times), default=None)
            if nearest is not None and nearest <= MATCH_TOLERANCE_SEC:
                target: object = 1
                is_ignored = 0
            elif nearest is not None and nearest <= IGNORE_RADIUS_SEC:
                target = ""
                is_ignored = 1
            else:
                target = 0
                is_ignored = 0

            features: dict[str, object] = {
                "peak_prob": peak,
                "peak_clip_prob": float(aux["clip_prob"][label][i]),
                "peak_response_prob": float(aux["response_prob"][label][i]),
                "peak_global_prob": float(aux["global_prob"][label][i]),
                "logit_peak_prob": logit(peak),
                "logit_clip_prob": logit(float(aux["clip_prob"][label][i])),
                "logit_response_prob": logit(float(aux["response_prob"][label][i])),
                "logit_global_prob": logit(float(aux["global_prob"][label][i])),
                "other_max_prob": other_max,
                "other_margin": peak - other_max,
                "video_median_logit": video_median_logit,
                "prominence": peak - baseline,
                "half_width_sec": half_peak_width(centers, p, i),
                "rise_slope_per_s": rise,
                "fall_slope_per_s": fall,
                "top2_mean": top2_mean,
                "plateau_sec": plateau_sec,
                "n_same_cand_pm15s": float(n_same),
                "n_cover_windows_prob03": float(n_cover),
                "shot_ctx_max_6s": shot_ctx_max_6s,
                "shot_ctx_max_12s": shot_ctx_max_12s,
                "time_since_last_shot_cand": time_since_last_shot,
            }
            features.update(audio_features_for_peak(audio, video_id, t))
            rows.append(
                {
                    "video_id": video_id,
                    "split": split,
                    "label": label,
                    "window_index": int(windows[i]["index"]),
                    "peak_time_sec": t,
                    "peak_score": peak,
                    "nearest_gt_distance_sec": "" if nearest is None else nearest,
                    "is_ignored": is_ignored,
                    "target": target,
                    **features,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_id_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--train-ids", required=True, type=Path)
    parser.add_argument("--val-ids", required=True, type=Path)
    parser.add_argument("--test-ids", required=True, type=Path)
    parser.add_argument("--test-gt-json", required=True, type=Path)
    parser.add_argument("--train-audio-dir", type=Path, default=None)
    parser.add_argument("--val-audio-dir", type=Path, default=None)
    parser.add_argument("--test-audio-dir", type=Path, default=None)
    parser.add_argument("--train-whistle-dir", type=Path, default=None)
    parser.add_argument("--val-whistle-dir", type=Path, default=None)
    parser.add_argument("--test-whistle-dir", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train135 = set(load_id_list(args.train_ids))
    val15 = set(load_id_list(args.val_ids))
    test18 = set(load_id_list(args.test_ids))
    if train135 & test18 or val15 & test18:
        raise ValueError("split leakage between train/val and test id lists")
    video_ids = [
        line
        for line in load_id_list(args.run_dir / "all_video_ids.txt")
        if (args.run_dir / line).is_dir()
    ]
    test_gt = load_test_gt(args.test_gt_json)
    audio_by_split = {
        "train": AudioIndex(args.train_audio_dir, args.train_whistle_dir),
        "val15": AudioIndex(args.val_audio_dir, args.val_whistle_dir),
        "test18": AudioIndex(args.test_audio_dir, args.test_whistle_dir),
        "unused": AudioIndex(None, None),
    }

    all_rows: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    skipped: list[str] = []
    for video_id in video_ids:
        if video_id in test18:
            split = "test18"
        elif video_id in val15:
            split = "val15"
        elif video_id in train135:
            split = "train"
        else:
            split = "unused"
        video_dir = args.run_dir / video_id
        if not (video_dir / "gt_events.csv").is_file() and split != "test18":
            print(f"  skip {video_id}: missing gt_events.csv")
            skipped.append(video_id)
            continue
        gt = (
            test_gt.get(video_id, {label: [] for label in LABELS})
            if split == "test18"
            else load_gt_from_run_dir(video_dir)
        )
        rows = build_video_rows(video_dir, split, gt, audio_by_split[split])
        if not rows:
            skipped.append(video_id)
        all_rows.extend(rows)
        split_counts[split] = split_counts.get(split, 0) + 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidates.csv", all_rows)

    stats: dict[str, object] = {"skipped_videos": skipped, "videos_per_split": split_counts}
    per_split_label: dict[str, dict[str, dict[str, int]]] = {}
    for row in all_rows:
        bucket = per_split_label.setdefault(
            str(row["split"]), {label: {"n": 0, "pos": 0, "neg": 0, "ign": 0} for label in LABELS}
        )[str(row["label"])]
        bucket["n"] += 1
        if row["is_ignored"]:
            bucket["ign"] += 1
        elif row["target"] == 1:
            bucket["pos"] += 1
        else:
            bucket["neg"] += 1
    stats["candidates"] = per_split_label
    stats["feature_names"] = FEATURE_NAMES
    stats["params"] = {
        "peak_min_score": PEAK_MIN_SCORE,
        "nms_radius_sec": NMS_RADIUS_SEC,
        "match_tolerance_sec": MATCH_TOLERANCE_SEC,
        "ignore_radius_sec": IGNORE_RADIUS_SEC,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(per_split_label, indent=2))
    print(f"wrote {args.output_dir / 'candidates.csv'} rows={len(all_rows)}")


if __name__ == "__main__":
    main()
