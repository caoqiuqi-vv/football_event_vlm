from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.annotations import LABEL_TO_FAMILY, load_events  # noqa: E402
from football_longform_v2.decoding import decode_family_conditioned_classes  # noqa: E402
from football_longform_v2.canonical import load_canonical_split  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.evaluation import (  # noqa: E402
    average_precision,
    dense_family_targets,
    dense_label_targets,
    events_for_family,
    events_for_label,
    family_proposals,
    infer_continuous_timeline,
    operating_point_at_recall,
    scored_point_matches,
)
from football_longform_v2.feature_store import assert_timeline_provenance, load_aligned_npz  # noqa: E402
from football_longform_v2.models import TemporalLocator  # noqa: E402
from football_longform_v2.models.temporal_locator import MODEL_SCHEMA  # noqa: E402
from football_longform_v2.schema import read_video_ids  # noqa: E402


def resolve(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous calibration-only LF-A0 locator evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--core-seconds", type=float, default=120.0)
    parser.add_argument("--context-seconds", type=float, default=30.0)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    parser.add_argument("--report")
    parser.add_argument("--expected-video-count", type=int, default=18)
    args = parser.parse_args()
    if args.split != "calibration":
        raise ValueError("this experiment evaluator is calibration-only; thirdparty18 is forbidden")

    config = load_config(args.config)
    root = Path(config["_project_root"])
    paths = config["paths"]
    canonical_path = resolve(root, paths["canonical_manifest"])
    canonical = load_canonical_split(canonical_path, "calibration")
    ids = read_video_ids(resolve(root, paths["calibration_ids"]))
    if canonical.media_ids != ids:
        raise RuntimeError("canonical manifest and calibration ID list disagree")
    if len(ids) != args.expected_video_count:
        raise RuntimeError(
            f"calibration video count {len(ids)} != required {args.expected_video_count}"
        )
    feature_root = resolve(root, paths["feature_store"]) / "calibration"
    annotations = resolve(root, paths["annotations"])
    families = tuple(config["task"]["proposal_families"])
    labels = tuple(config["task"]["output_labels"])
    class_nms_radius = {
        label: (3.0 if label in {"shot", "save"} else 6.0) for label in labels
    }
    class_max_per_minute = {
        "shot": 3.0, "save": 2.0, "corner": 1.0, "penalty": 0.5,
        "freekick": 1.0, "kickoff": 0.5,
    }
    class_local_search_radius = {
        label: (2.0 if label == "shot" else 4.0) for label in labels
    }
    config_sha256 = sha256_file(Path(config["_config_path"]))
    context_config = config["features"]["context"]
    weights_sha256 = sha256_file(resolve(root, str(context_config["weights"])))
    preflight_errors: list[str] = []
    for video_id in ids:
        cache_path = feature_root / video_id / "timeline.npz"
        annotation_path = annotations / f"{canonical.annotation_id_by_media_id[video_id]}.json"
        if not annotation_path.is_file():
            preflight_errors.append(f"{video_id}: missing annotation {annotation_path}")
            continue
        try:
            assert_timeline_provenance(
                cache_path,
                expected_backbone_arch=str(context_config["arch"]),
                expected_backbone_id=str(context_config["backbone_id"]),
                expected_source_video=canonical.source_video_by_media_id[video_id],
                expected_config_sha256=config_sha256,
                expected_weights_sha256=weights_sha256,
            )
        except (OSError, KeyError, ValueError) as error:
            preflight_errors.append(f"{video_id}: {error}")
    if preflight_errors:
        raise RuntimeError(
            "strict calibration preflight failed; no partial evaluation is allowed:\n"
            + "\n".join(preflight_errors)
        )
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_canonical = load_canonical_split(canonical_path, "train")
    if payload.get("model_schema") != MODEL_SCHEMA:
        raise RuntimeError("checkpoint model schema is not the conditional class locator contract")
    if tuple(payload.get("output_labels", ())) != tuple(config["task"]["output_labels"]):
        raise RuntimeError("checkpoint output label order disagrees with config")
    if payload.get("train_video_count") != len(train_canonical.media_ids):
        raise RuntimeError("checkpoint was not trained on the complete canonical train split")
    if payload.get("missing_train_video_ids"):
        raise RuntimeError("checkpoint records missing train videos")
    if payload.get("feature_config_sha256") != config_sha256:
        raise RuntimeError("checkpoint feature config hash disagrees with calibration caches")
    if payload.get("feature_weights_sha256") != weights_sha256:
        raise RuntimeError("checkpoint feature weights hash disagrees with calibration caches")
    model = TemporalLocator.from_config(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    timeline_hz = float(config["features"]["timeline_hz"])
    core_steps = int(round(args.core_seconds * timeline_hz))
    context_steps = int(round(args.context_seconds * timeline_hz))

    dense_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    dense_targets: dict[str, list[torch.Tensor]] = defaultdict(list)
    raw_scores: dict[str, list[float]] = defaultdict(list)
    raw_labels: dict[str, list[bool]] = defaultdict(list)
    raw_candidates: dict[str, list[tuple[list, list[float]]]] = defaultdict(list)
    raw_support: dict[str, int] = defaultdict(int)
    budget_tp: dict[str, int] = defaultdict(int)
    budget_fp: dict[str, int] = defaultdict(int)
    budget_support: dict[str, int] = defaultdict(int)
    budget_minutes: dict[str, float] = defaultdict(float)
    class_dense_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    class_dense_targets: dict[str, list[torch.Tensor]] = defaultdict(list)
    class_candidates: dict[str, list[tuple[list, list[float]]]] = defaultdict(list)
    class_raw_scores: dict[str, list[float]] = defaultdict(list)
    class_raw_labels: dict[str, list[bool]] = defaultdict(list)
    class_support: dict[str, int] = defaultdict(int)
    class_budget_tp: dict[str, int] = defaultdict(int)
    class_budget_fp: dict[str, int] = defaultdict(int)
    class_budget_minutes: dict[str, float] = defaultdict(float)
    per_video: list[dict] = []
    missing_caches: list[str] = []
    decode = config["decode"]
    for video_id in ids:
        cache_path = feature_root / video_id / "timeline.npz"
        annotation_path = annotations / f"{canonical.annotation_id_by_media_id[video_id]}.json"
        timeline = load_aligned_npz(cache_path)
        events = load_events(annotation_path)
        output = infer_continuous_timeline(
            model, timeline, device=device, core_steps=core_steps, context_steps=context_steps
        )
        valid = timeline.context_valid & timeline.motion_valid
        targets = dense_family_targets(timeline.timestamps, events, families)
        if output.class_logits is None:
            raise RuntimeError("conditional class logits are missing from continuous inference")
        label_targets = dense_label_targets(timeline.timestamps, events, labels)
        class_unconstrained = decode_family_conditioned_classes(
            output.logits, output.class_logits, timeline.timestamps, families, labels,
            label_to_family=LABEL_TO_FAMILY,
            family_nms_radius_seconds=decode["nms_radius_seconds"],
            class_nms_radius_seconds=class_nms_radius,
            local_search_radius_seconds=class_local_search_radius,
            max_class_per_minute={label: 1000.0 for label in labels},
        )
        class_budgeted = decode_family_conditioned_classes(
            output.logits, output.class_logits, timeline.timestamps, families, labels,
            label_to_family=LABEL_TO_FAMILY,
            family_nms_radius_seconds=decode["nms_radius_seconds"],
            class_nms_radius_seconds=class_nms_radius,
            local_search_radius_seconds=class_local_search_radius,
            max_class_per_minute={label: class_max_per_minute[label] for label in labels},
        )
        unconstrained = family_proposals(
            output.logits, timeline.timestamps, families, threshold=0.0,
            nms_radius_seconds=decode["nms_radius_seconds"],
            max_per_minute={family: 1000.0 for family in families},
        )
        budgeted = family_proposals(
            output.logits, timeline.timestamps, families, threshold=0.0,
            nms_radius_seconds=decode["nms_radius_seconds"], max_per_minute=decode["max_proposals_per_minute"],
        )
        duration_minutes = max(float(timeline.timestamps[-1]) / 60.0, 1.0 / 60.0)
        item: dict = {
            "video_id": video_id,
            "timeline_steps": int(timeline.timestamps.numel()),
            "duration_seconds": float(timeline.timestamps[-1]),
            "valid_steps": int(valid.sum()),
            "continuous_coverage_min": int(output.coverage.min()),
            "continuous_coverage_max": int(output.coverage.max()),
            "families": {},
            "labels": {},
        }
        for index, family in enumerate(families):
            dense_scores[family].append(output.logits[:, index].sigmoid()[valid])
            dense_targets[family].append((targets[:, index] >= 0.5)[valid])
            event_times = events_for_family(events, family)
            raw_scores_one, raw_labels_one, _ = scored_point_matches(
                unconstrained[family], event_times, tolerance_seconds=2.0
            )
            raw_scores[family].extend(raw_scores_one)
            raw_labels[family].extend(raw_labels_one)
            raw_candidates[family].append((unconstrained[family], event_times))
            raw_support[family] += len(event_times)
            _, budget_labels, budget_unmatched = scored_point_matches(
                budgeted[family], event_times, tolerance_seconds=2.0
            )
            budget_tp[family] += sum(budget_labels)
            budget_fp[family] += len(budget_labels) - sum(budget_labels)
            budget_support[family] += len(event_times)
            budget_minutes[family] += duration_minutes
            item["families"][family] = {
                "event_support": len(event_times),
                "candidate_count": len(unconstrained[family]),
                "budget_proposal_count": len(budgeted[family]),
                "budget_tp_at_2s": sum(budget_labels),
                "budget_fp_at_2s": len(budget_labels) - sum(budget_labels),
                "budget_missed_at_2s": budget_unmatched,
            }
        for index, label in enumerate(labels):
            class_dense_scores[label].append(output.class_logits[:, index].sigmoid()[valid])
            class_dense_targets[label].append((label_targets[:, index] >= 0.5)[valid])
            event_times = events_for_label(events, label)
            class_scores_one, class_labels_one, _ = scored_point_matches(
                class_unconstrained[label], event_times, tolerance_seconds=2.0
            )
            class_raw_scores[label].extend(class_scores_one)
            class_raw_labels[label].extend(class_labels_one)
            class_candidates[label].append((class_unconstrained[label], event_times))
            class_support[label] += len(event_times)
            _, budget_labels, budget_unmatched = scored_point_matches(
                class_budgeted[label], event_times, tolerance_seconds=2.0
            )
            class_budget_tp[label] += sum(budget_labels)
            class_budget_fp[label] += len(budget_labels) - sum(budget_labels)
            class_budget_minutes[label] += duration_minutes
            item["labels"][label] = {
                "event_support": len(event_times),
                "candidate_count": len(class_unconstrained[label]),
                "budget_proposal_count": len(class_budgeted[label]),
                "budget_tp_at_2s": sum(budget_labels),
                "budget_fp_at_2s": len(budget_labels) - sum(budget_labels),
                "budget_missed_at_2s": budget_unmatched,
            }
        per_video.append(item)

    summary: dict[str, dict] = {}
    for family in families:
        scores = torch.cat(dense_scores[family]) if dense_scores[family] else torch.empty(0)
        targets = torch.cat(dense_targets[family]) if dense_targets[family] else torch.empty(0, dtype=torch.bool)
        family_target_recall = 0.97 if family in {"shot_chain", "restart"} else 0.90
        family_summary = {
            "dense_family_auprc": average_precision(scores, targets),
            "dense_positive_steps": int(targets.sum()),
            "point_support": raw_support[family],
            "candidate_ceiling_recall_at_2s": (
                sum(raw_labels[family]) / raw_support[family] if raw_support[family] else None
            ),
            "point_ap_at_tolerance": {},
            "budget_recall_at_2s": (
                budget_tp[family] / budget_support[family] if budget_support[family] else None
            ),
            "budget_fp_per_minute_at_2s": (
                budget_fp[family] / budget_minutes[family] if budget_minutes[family] else None
            ),
            "budget_tp_at_2s": budget_tp[family],
            "budget_fp_at_2s": budget_fp[family],
            "operating_point_for_target_recall_at_2s": operating_point_at_recall(
                raw_scores[family], raw_labels[family], positive_count=raw_support[family],
                target_recall=family_target_recall, total_minutes=budget_minutes[family],
            ),
        }
        for tolerance in args.tolerances:
            point_scores: list[float] = []
            point_labels: list[bool] = []
            for proposals, event_times in raw_candidates[family]:
                one_scores, one_labels, _ = scored_point_matches(
                    proposals, event_times, tolerance_seconds=tolerance
                )
                point_scores.extend(one_scores)
                point_labels.extend(one_labels)
            family_summary["point_ap_at_tolerance"][str(tolerance)] = average_precision(
                torch.tensor(point_scores, dtype=torch.float32),
                torch.tensor(point_labels, dtype=torch.bool),
                positive_count=raw_support[family],
            )
        summary[family] = family_summary
    label_summary: dict[str, dict] = {}
    for label in labels:
        label_scores = (
            torch.cat(class_dense_scores[label]) if class_dense_scores[label] else torch.empty(0)
        )
        label_targets_one = (
            torch.cat(class_dense_targets[label])
            if class_dense_targets[label] else torch.empty(0, dtype=torch.bool)
        )
        target_recall = 0.90 if label == "shot" else 0.85
        one_summary = {
            "dense_class_auprc": average_precision(label_scores, label_targets_one),
            "dense_positive_steps": int(label_targets_one.sum()),
            "point_support": class_support[label],
            "candidate_ceiling_recall_at_2s": (
                sum(class_raw_labels[label]) / class_support[label] if class_support[label] else None
            ),
            "point_ap_at_tolerance": {},
            "budget_recall_at_2s": (
                class_budget_tp[label] / class_support[label] if class_support[label] else None
            ),
            "budget_fp_per_minute_at_2s": (
                class_budget_fp[label] / class_budget_minutes[label]
                if class_budget_minutes[label] else None
            ),
            "budget_tp_at_2s": class_budget_tp[label],
            "budget_fp_at_2s": class_budget_fp[label],
            "max_proposals_per_minute": class_max_per_minute[label],
            "operating_point_for_target_recall_at_2s": operating_point_at_recall(
                class_raw_scores[label], class_raw_labels[label],
                positive_count=class_support[label], target_recall=target_recall,
                total_minutes=class_budget_minutes[label],
            ),
        }
        for tolerance in args.tolerances:
            point_scores: list[float] = []
            point_labels: list[bool] = []
            for proposals, event_times in class_candidates[label]:
                one_scores, one_labels, _ = scored_point_matches(
                    proposals, event_times, tolerance_seconds=tolerance
                )
                point_scores.extend(one_scores)
                point_labels.extend(one_labels)
            one_summary["point_ap_at_tolerance"][str(tolerance)] = average_precision(
                torch.tensor(point_scores, dtype=torch.float32),
                torch.tensor(point_labels, dtype=torch.bool),
                positive_count=class_support[label],
            )
        label_summary[label] = one_summary

    report = {
        "experiment_id": config["experiment_id"],
        "evaluation_split": "calibration",
        "strict_complete_split": True,
        "required_video_count": args.expected_video_count,
        "evaluated_video_ids": list(ids),
        "feature_config_sha256": config_sha256,
        "feature_weights_sha256": weights_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "device": str(device),
        "continuous_inference": {
            "core_seconds": args.core_seconds, "context_seconds": args.context_seconds,
            "each_final_timeline_point_emitted_once": all(
                item["continuous_coverage_min"] == 1 and item["continuous_coverage_max"] == 1
                for item in per_video
            ),
        },
        "missing_calibration_caches": missing_caches,
        "class_proposal_route": "family_conditioned_local_search_v1",
        "class_local_search_radius_seconds": class_local_search_radius,
        "evaluated_video_count": len(per_video),
        "families": summary,
        "labels": label_summary,
        "per_video": per_video,
    }
    report_path = (
        Path(args.report).expanduser().resolve() if args.report else
        resolve(root, paths["output_dir"]) / f"calibration_epoch_{int(payload.get('epoch', -1)):03d}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evaluated_video_count": len(per_video), "missing_calibration_caches": missing_caches,
                      "families": summary, "labels": label_summary}, ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
