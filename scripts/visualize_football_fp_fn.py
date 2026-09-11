#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


DEFAULT_LABELS = ("shot", "save", "set_piece")


@dataclass
class ErrorItem:
    error_type: str
    video_id: str
    label: str
    window_index: int
    start_sec: float
    end_sec: float
    clip_prob: float
    threshold: float
    frame_peak_prob: float
    frame_peak_time: float
    reason: str
    gt_time_sec: float | None = None
    gt_event_id: str = ""
    top_candidate_scores: str = ""
    image_path: str = ""
    decode_failures: int = 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def parse_exclusions(values: Sequence[str]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for value in values:
        video_id, label = value.rsplit(":", 1)
        result.add((video_id.strip(), label.strip()))
    return result


def interval_distance(time_sec: float, start_sec: float, end_sec: float) -> float:
    if start_sec <= time_sec <= end_sec:
        return 0.0
    return min(abs(time_sec - start_sec), abs(time_sec - end_sec))


def find_video(video_root: Path, video_id: str) -> Path:
    for suffix in (".mp4", ".mov", ".mkv", ".avi"):
        path = video_root / f"{video_id}{suffix}"
        if path.exists():
            return path
    candidates = sorted(video_root.glob(f"{video_id}.*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Missing video {video_id} under {video_root}")


def load_thresholds(run_dir: Path, video_id: str, labels: Sequence[str]) -> dict[str, float]:
    summary = json.loads((run_dir / video_id / "summary.json").read_text())
    return {label: float(summary["thresholds"][label]) for label in labels}


def frame_rows_by_window(video_dir: Path) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(video_dir / "frame_event_logits.csv"):
        grouped[int(row["window_index"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["frame_index"]))
    return grouped


def frame_peak(rows: Sequence[dict[str, str]], label: str) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    best = max(rows, key=lambda row: float(row[f"frame_prob_{label}"]))
    return float(best[f"frame_prob_{label}"]), float(best["frame_time_sec"])


def select_display_rows(
    rows: Sequence[dict[str, str]], label: str, count: int
) -> list[dict[str, str]]:
    if len(rows) <= count:
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, count).round().astype(int).tolist()
    peak_index = max(
        range(len(rows)), key=lambda index: float(rows[index][f"frame_prob_{label}"])
    )
    if peak_index not in indices:
        replace = min(
            range(len(indices)), key=lambda index: abs(indices[index] - peak_index)
        )
        indices[replace] = peak_index
    return [rows[index] for index in sorted(set(indices))]


def read_video_frame(
    capture: cv2.VideoCapture,
    time_sec: float,
    size: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(time_sec, 0.0) * 1000.0)
    ok, frame = capture.read()
    width, height = size
    if not ok or frame is None:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "DECODE FAILED",
            (20, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame, False
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), True


def read_video_frames(
    capture: cv2.VideoCapture,
    times_sec: Sequence[float],
    size: tuple[int, int],
) -> list[tuple[np.ndarray, bool]]:
    if not times_sec:
        return []
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        return [read_video_frame(capture, time_sec, size) for time_sec in times_sec]
    capture.set(cv2.CAP_PROP_POS_MSEC, max(times_sec[0], 0.0) * 1000.0)
    results: list[tuple[np.ndarray, bool]] = []
    for time_sec in times_sec:
        target_frame = max(int(round(time_sec * fps)), 0)
        current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        while current_frame < target_frame:
            if not capture.grab():
                break
            current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = capture.retrieve()
        if not ok or frame is None:
            results.append(read_video_frame(capture, time_sec, size))
            continue
        width, height = size
        results.append(
            (
                cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA),
                True,
            )
        )
    return results


def draw_text_lines(
    image: np.ndarray,
    lines: Sequence[str],
    *,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    line_height: int = 22,
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height


def render_contact_sheet(
    item: ErrorItem,
    rows: Sequence[dict[str, str]],
    capture: cv2.VideoCapture,
    output_path: Path,
    *,
    display_frames: int,
    thumb_size: tuple[int, int],
    jpeg_quality: int,
) -> None:
    selected_rows = select_display_rows(rows, item.label, display_frames)
    columns = 4
    rows_count = max(math.ceil(len(selected_rows) / columns), 1)
    thumb_width, thumb_height = thumb_size
    header_height = 132
    sheet = np.zeros(
        (header_height + rows_count * thumb_height, columns * thumb_width, 3),
        dtype=np.uint8,
    )
    header_color = (40, 40, 170) if item.error_type == "FP" else (140, 70, 20)
    sheet[:header_height] = header_color
    title = (
        f"{item.error_type} {item.label}  video={item.video_id}  "
        f"window={item.window_index} [{item.start_sec:.1f},{item.end_sec:.1f}]"
    )
    score_line = (
        f"clip={item.clip_prob:.4f} threshold={item.threshold:.4f}  "
        f"frame_peak={item.frame_peak_prob:.4f} at {item.frame_peak_time:.2f}s"
    )
    extra = f"reason={item.reason}"
    if item.gt_time_sec is not None:
        extra += f"  GT={item.gt_time_sec:.3f}s  candidates={item.top_candidate_scores}"
    draw_text_lines(
        sheet,
        [title, score_line, extra],
        origin=(12, 30),
        scale=0.64,
        thickness=2,
        line_height=36,
    )

    decode_failures = 0
    decoded_frames = read_video_frames(
        capture,
        [float(row["frame_time_sec"]) for row in selected_rows],
        thumb_size,
    )
    for display_index, (row, decoded) in enumerate(
        zip(selected_rows, decoded_frames)
    ):
        frame_time = float(row["frame_time_sec"])
        frame_prob = float(row[f"frame_prob_{item.label}"])
        frame, ok = decoded
        decode_failures += int(not ok)
        is_peak = abs(frame_time - item.frame_peak_time) <= 1e-4
        border_color = (0, 0, 255) if is_peak else (180, 180, 180)
        cv2.rectangle(frame, (1, 1), (thumb_width - 2, thumb_height - 2), border_color, 3)
        gt_text = ""
        if item.gt_time_sec is not None:
            gt_text = f" dGT={frame_time - item.gt_time_sec:+.2f}s"
        text = f"t={frame_time:.2f}s q={frame_prob:.3f}{gt_text}"
        cv2.rectangle(frame, (0, thumb_height - 29), (thumb_width, thumb_height), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (7, thumb_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255) if is_peak else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row_index, column_index = divmod(display_index, columns)
        y = header_height + row_index * thumb_height
        x = column_index * thumb_width
        sheet[y : y + thumb_height, x : x + thumb_width] = frame

    item.decode_failures = decode_failures
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    ):
        raise RuntimeError(f"Failed to write {output_path}")


def build_errors(
    run_dir: Path,
    *,
    labels: Sequence[str],
    tolerance_sec: float,
    exclusions: set[tuple[str, str]],
    requested_video_ids: Sequence[str] | None = None,
) -> tuple[list[ErrorItem], dict[str, Any], list[str]]:
    run_config = json.loads((run_dir / "run_config.json").read_text())
    video_ids = [str(item) for item in run_config["video_ids"]]
    if requested_video_ids is not None:
        requested = {str(item) for item in requested_video_ids}
        video_ids = [video_id for video_id in video_ids if video_id in requested]
    thresholds = load_thresholds(run_dir, video_ids[0], labels)
    errors: list[ErrorItem] = []
    statistics: dict[str, Any] = {
        "thresholds": thresholds,
        "per_class": {},
        "per_video": {},
    }
    class_accumulators: dict[str, dict[str, Any]] = {
        label: {
            "tp_clip": [],
            "tp_frame": [],
            "fp_clip": [],
            "fp_frame": [],
            "fn_clip": [],
            "fn_frame": [],
            "fp_reasons": Counter(),
            "num_gt": 0,
            "matched_gt": 0,
        }
        for label in labels
    }

    for video_id in video_ids:
        video_dir = run_dir / video_id
        windows = read_csv(video_dir / "window_predictions.csv")
        frame_groups = frame_rows_by_window(video_dir)
        gt_rows = read_csv(video_dir / "gt_events.csv")
        gt_by_label = {
            label: [row for row in gt_rows if row["label"] == label]
            for label in labels
        }
        video_counts: Counter[str] = Counter()
        for label in labels:
            if (video_id, label) in exclusions:
                continue
            gt_items = gt_by_label[label]
            gt_times = [float(row["time_sec"]) for row in gt_items]
            class_accumulators[label]["num_gt"] += len(gt_items)
            matched_gt: set[int] = set()
            for window in windows:
                window_index = int(window["index"])
                start_sec = float(window["start_sec"])
                end_sec = float(window["end_sec"])
                clip_prob = float(window[f"prob_{label}"])
                rows = frame_groups.get(window_index, [])
                peak_prob, peak_time = frame_peak(rows, label)
                matches = [
                    index
                    for index, gt_time in enumerate(gt_times)
                    if start_sec - tolerance_sec <= gt_time <= end_sec + tolerance_sec
                ]
                if clip_prob < thresholds[label]:
                    continue
                if matches:
                    matched_gt.update(matches)
                    class_accumulators[label]["tp_clip"].append(clip_prob)
                    class_accumulators[label]["tp_frame"].append(peak_prob)
                    continue

                other_events = [
                    row
                    for row in gt_rows
                    if row["label"] != label
                    and start_sec - tolerance_sec
                    <= float(row["time_sec"])
                    <= end_sec + tolerance_sec
                ]
                nearest_same = min(
                    (
                        interval_distance(gt_time, start_sec, end_sec)
                        for gt_time in gt_times
                    ),
                    default=float("inf"),
                )
                reasons: list[str] = []
                if other_events:
                    reasons.append(
                        "other_event:" + "+".join(sorted({row["label"] for row in other_events}))
                    )
                if tolerance_sec < nearest_same <= tolerance_sec + 15.0:
                    reasons.append("near_same_class")
                if peak_prob >= 0.10:
                    reasons.append("frame_supported")
                if not reasons:
                    reasons.append("background_or_domain_shift")
                reason = ";".join(reasons)
                errors.append(
                    ErrorItem(
                        error_type="FP",
                        video_id=video_id,
                        label=label,
                        window_index=window_index,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        clip_prob=clip_prob,
                        threshold=thresholds[label],
                        frame_peak_prob=peak_prob,
                        frame_peak_time=peak_time,
                        reason=reason,
                    )
                )
                class_accumulators[label]["fp_clip"].append(clip_prob)
                class_accumulators[label]["fp_frame"].append(peak_prob)
                class_accumulators[label]["fp_reasons"].update(reasons)
                video_counts[f"FP_{label}"] += 1

            class_accumulators[label]["matched_gt"] += len(matched_gt)
            for gt_index, gt_item in enumerate(gt_items):
                if gt_index in matched_gt:
                    continue
                gt_time = float(gt_item["time_sec"])
                candidates = [
                    window
                    for window in windows
                    if float(window["start_sec"]) - tolerance_sec
                    <= gt_time
                    <= float(window["end_sec"]) + tolerance_sec
                ]
                if not candidates:
                    candidates = sorted(
                        windows,
                        key=lambda window: interval_distance(
                            gt_time,
                            float(window["start_sec"]),
                            float(window["end_sec"]),
                        ),
                    )[:1]
                candidates.sort(key=lambda window: float(window[f"prob_{label}"]), reverse=True)
                best = candidates[0]
                window_index = int(best["index"])
                rows = frame_groups.get(window_index, [])
                peak_prob, peak_time = frame_peak(rows, label)
                best_prob = float(best[f"prob_{label}"])
                score_text = ",".join(
                    f"{float(window[f'prob_{label}']):.3f}"
                    for window in candidates[:3]
                )
                reasons = []
                if best_prob >= 0.8 * thresholds[label]:
                    reasons.append("clip_near_threshold")
                else:
                    reasons.append("clip_low")
                if peak_prob >= 0.10:
                    reasons.append("frame_detected_but_clip_missed")
                else:
                    reasons.append("frame_also_low")
                errors.append(
                    ErrorItem(
                        error_type="FN",
                        video_id=video_id,
                        label=label,
                        window_index=window_index,
                        start_sec=float(best["start_sec"]),
                        end_sec=float(best["end_sec"]),
                        clip_prob=best_prob,
                        threshold=thresholds[label],
                        frame_peak_prob=peak_prob,
                        frame_peak_time=peak_time,
                        reason=";".join(reasons),
                        gt_time_sec=gt_time,
                        gt_event_id=gt_item.get("event_id", ""),
                        top_candidate_scores=score_text,
                    )
                )
                class_accumulators[label]["fn_clip"].append(best_prob)
                class_accumulators[label]["fn_frame"].append(peak_prob)
                video_counts[f"FN_{label}"] += 1
        statistics["per_video"][video_id] = dict(video_counts)

    def mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    for label, accumulator in class_accumulators.items():
        statistics["per_class"][label] = {
            "num_gt": accumulator["num_gt"],
            "matched_gt": accumulator["matched_gt"],
            "fn": accumulator["num_gt"] - accumulator["matched_gt"],
            "fp": len(accumulator["fp_clip"]),
            "tp_windows": len(accumulator["tp_clip"]),
            "tp_clip_prob_mean": mean(accumulator["tp_clip"]),
            "fp_clip_prob_mean": mean(accumulator["fp_clip"]),
            "tp_frame_prob_mean": mean(accumulator["tp_frame"]),
            "fp_frame_prob_mean": mean(accumulator["fp_frame"]),
            "fn_best_clip_prob_mean": mean(accumulator["fn_clip"]),
            "fn_best_frame_prob_mean": mean(accumulator["fn_frame"]),
            "fp_reason_counts": dict(accumulator["fp_reasons"]),
        }
    return errors, statistics, video_ids


def write_index_csv(path: Path, items: Sequence[ErrorItem]) -> None:
    fields = list(ErrorItem.__dataclass_fields__)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow({field: getattr(item, field) for field in fields})


def write_html(path: Path, items: Sequence[ErrorItem], summary: dict[str, Any]) -> None:
    cards = []
    for item in items:
        gt = "" if item.gt_time_sec is None else f" GT={item.gt_time_sec:.3f}s"
        cards.append(
            "<article class='card' "
            f"data-error='{item.error_type}' data-label='{item.label}' data-video='{item.video_id}'>"
            f"<a href='{html.escape(item.image_path)}'><img loading='lazy' src='{html.escape(item.image_path)}'></a>"
            f"<div><b>{item.error_type} {item.label}</b> {item.video_id} "
            f"[{item.start_sec:.1f},{item.end_sec:.1f}]{gt}<br>"
            f"clip={item.clip_prob:.4f} frame={item.frame_peak_prob:.4f}<br>"
            f"{html.escape(item.reason)}</div></article>"
        )
    summary_text = html.escape(json.dumps(summary["per_class"], ensure_ascii=False, indent=2))
    document = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Football FP/FN audit</title>
<style>
body{{font-family:Arial,sans-serif;background:#111;color:#eee;margin:18px}}
.toolbar{{position:sticky;top:0;background:#222;padding:12px;z-index:2}}
button{{margin:3px;padding:7px 12px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}}
.card{{background:#222;padding:8px;border-radius:7px}} .card img{{width:100%;height:auto}} pre{{white-space:pre-wrap}}
</style></head><body><h1>Football FP/FN audit</h1><pre>{summary_text}</pre>
<div class='toolbar'>
<button onclick="filter('ALL','ALL')">ALL</button>
<button onclick="filter('FP','ALL')">FP</button><button onclick="filter('FN','ALL')">FN</button>
<button onclick="filter('ALL','shot')">shot</button><button onclick="filter('ALL','save')">save</button>
<button onclick="filter('ALL','set_piece')">set_piece</button></div>
<section class='grid'>{''.join(cards)}</section>
<script>function filter(e,l){{document.querySelectorAll('.card').forEach(c=>{{c.style.display=((e==='ALL'||c.dataset.error===e)&&(l==='ALL'||c.dataset.label===l))?'block':'none'}})}};</script>
</body></html>"""
    path.write_text(document)


def parse_size(raw: str) -> tuple[int, int]:
    width, height = raw.lower().replace("x", ",").split(",")
    return int(width), int(height)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render every window-overlap false-positive window and every unmatched GT event."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--display-frames", type=int, default=8)
    parser.add_argument("--thumb-size", default="320x180")
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--max-fp", type=int, default=0)
    parser.add_argument("--max-fn", type=int, default=0)
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--video-id-file", default="")
    parser.add_argument("--sort-errors", default="default", choices=["default", "confidence_desc", "frame_peak_desc"])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    exclusions = parse_exclusions(args.exclude)
    requested_video_ids = None
    if args.video_ids or args.video_id_file:
        requested_video_ids = []
        if args.video_ids:
            requested_video_ids.extend(item.strip() for item in args.video_ids.split(",") if item.strip())
        if args.video_id_file:
            for line in Path(args.video_id_file).read_text().splitlines():
                item = line.strip()
                if item and not item.startswith("#"):
                    requested_video_ids.append(item)
    errors, statistics, video_ids = build_errors(
        run_dir,
        labels=labels,
        tolerance_sec=args.match_tolerance_sec,
        exclusions=exclusions,
        requested_video_ids=requested_video_ids,
    )
    if args.sort_errors == "confidence_desc":
        errors.sort(key=lambda item: (item.error_type, item.label, -item.clip_prob, item.video_id, item.window_index))
    elif args.sort_errors == "frame_peak_desc":
        errors.sort(key=lambda item: (item.error_type, item.label, -item.frame_peak_prob, item.video_id, item.window_index))
    fp_items = [item for item in errors if item.error_type == "FP"]
    fn_items = [item for item in errors if item.error_type == "FN"]
    if args.max_fp > 0:
        fp_items = fp_items[: args.max_fp]
    if args.max_fn > 0:
        fn_items = fn_items[: args.max_fn]
    selected_items = fp_items + fn_items
    output_dir.mkdir(parents=True, exist_ok=True)

    items_by_video: dict[str, list[ErrorItem]] = defaultdict(list)
    for item in selected_items:
        items_by_video[item.video_id].append(item)
    for video_id in video_ids:
        video_items = items_by_video.get(video_id, [])
        if not video_items:
            continue
        capture = cv2.VideoCapture(str(find_video(Path(args.video_root), video_id)))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video {video_id}")
        frame_groups = frame_rows_by_window(run_dir / video_id)
        for item_index, item in enumerate(video_items, start=1):
            safe_gt = "" if item.gt_time_sec is None else f"_gt{item.gt_time_sec:09.3f}"
            filename = (
                f"{item.error_type.lower()}_{item.label}_{video_id}_w{item.window_index:05d}"
                f"_t{item.start_sec:09.3f}{safe_gt}.jpg"
            )
            relative_path = Path(item.error_type.lower()) / item.label / video_id / filename
            render_contact_sheet(
                item,
                frame_groups.get(item.window_index, []),
                capture,
                output_dir / relative_path,
                display_frames=args.display_frames,
                thumb_size=parse_size(args.thumb_size),
                jpeg_quality=args.jpeg_quality,
            )
            item.image_path = relative_path.as_posix()
            if item_index == 1 or item_index % 50 == 0 or item_index == len(video_items):
                print(
                    f"video={video_id} rendered={item_index}/{len(video_items)}",
                    flush=True,
                )
        capture.release()

    statistics["protocol"] = {
        "run_dir": str(run_dir),
        "prediction_postprocess": "window_overlap",
        "match_tolerance_sec": args.match_tolerance_sec,
        "excluded_video_label_pairs": [list(item) for item in sorted(exclusions)],
        "video_ids": video_ids,
    }
    statistics["rendered"] = {
        "fp": len(fp_items),
        "fn": len(fn_items),
        "decode_failures": sum(item.decode_failures for item in selected_items),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2)
    )
    write_index_csv(output_dir / "index.csv", selected_items)
    write_html(output_dir / "index.html", selected_items, statistics)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
