from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.annotations import LABEL_TO_FAMILY, load_events  # noqa: E402
from football_longform_v2.canonical import load_canonical_split  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.decoding import decode_family_conditioned_classes  # noqa: E402
from football_longform_v2.evaluation import (  # noqa: E402
    average_precision,
    events_for_label,
    infer_continuous_timeline,
    operating_point_at_recall,
    scored_point_matches,
)
from football_longform_v2.feature_store import (  # noqa: E402
    assert_timeline_provenance,
    load_aligned_npz,
)
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


def validate_oof_checkpoint_contract(
    payload: dict, *, canonical_ids: tuple[str, ...], train_ids: tuple[str, ...],
    validation_ids: tuple[str, ...], config_sha256: str, weights_sha256: str,
) -> None:
    canonical = set(canonical_ids)
    train = set(train_ids)
    validation = set(validation_ids)
    if train & validation:
        raise RuntimeError("OOF train and validation IDs overlap")
    if train | validation != canonical:
        raise RuntimeError("OOF train and validation IDs do not partition canonical train")
    if payload.get("model_schema") != MODEL_SCHEMA:
        raise RuntimeError("checkpoint model schema mismatch")
    if tuple(payload.get("training_media_ids", ())) != train_ids:
        raise RuntimeError("checkpoint training IDs do not exactly match OOF fold train IDs")
    if set(payload.get("held_out_train_video_ids", ())) != validation:
        raise RuntimeError("checkpoint held-out IDs do not match OOF validation IDs")
    if int(payload.get("train_video_count", -1)) != len(train_ids):
        raise RuntimeError("checkpoint train video count mismatch")
    if int(payload.get("canonical_train_video_count", -1)) != len(canonical_ids):
        raise RuntimeError("checkpoint canonical train video count mismatch")
    if payload.get("missing_train_video_ids"):
        raise RuntimeError("checkpoint records missing training videos")
    if payload.get("feature_split") != "train":
        raise RuntimeError("OOF checkpoint must use train-namespaced features")
    if payload.get("feature_config_sha256") != config_sha256:
        raise RuntimeError("checkpoint feature config hash mismatch")
    if payload.get("feature_weights_sha256") != weights_sha256:
        raise RuntimeError("checkpoint feature weights hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe held-out-fold evaluator for penalty OOF calibration."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oof-manifest", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--core-seconds", type=float, default=120.0)
    parser.add_argument("--context-seconds", type=float, default=30.0)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["_project_root"])
    paths = config["paths"]
    canonical_path = resolve(root, paths["canonical_manifest"])
    canonical = load_canonical_split(canonical_path, "train")
    oof_path = Path(args.oof_manifest).expanduser().resolve()
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    if oof.get("schema") != "football_longform_v2.penalty_oof_folds.v1":
        raise RuntimeError("unsupported penalty OOF manifest schema")
    if oof.get("canonical_manifest_sha256") != sha256_file(canonical_path):
        raise RuntimeError("OOF manifest canonical hash mismatch")
    fold_records = {int(item["fold"]): item for item in oof.get("folds", [])}
    if args.fold not in fold_records:
        raise RuntimeError(f"OOF fold {args.fold} is absent")
    fold = fold_records[args.fold]
    train_id_file = Path(fold["train_id_file"]).resolve()
    validation_id_file = Path(fold["validation_id_file"]).resolve()
    if sha256_file(train_id_file) != fold["train_id_file_sha256"]:
        raise RuntimeError("OOF train ID file hash mismatch")
    if sha256_file(validation_id_file) != fold["validation_id_file_sha256"]:
        raise RuntimeError("OOF validation ID file hash mismatch")
    train_ids = read_video_ids(train_id_file)
    validation_ids = read_video_ids(validation_id_file)
    if list(train_ids) != fold["train_media_ids"]:
        raise RuntimeError("OOF train ID file and manifest disagree")
    if list(validation_ids) != fold["validation_media_ids"]:
        raise RuntimeError("OOF validation ID file and manifest disagree")
    calibration_ids = set(load_canonical_split(canonical_path, "calibration").media_ids)
    fixed_test_ids = set(read_video_ids(resolve(root, paths["thirdparty_test_ids"])))
    if set(validation_ids) & (calibration_ids | fixed_test_ids):
        raise RuntimeError("OOF validation overlaps calibration or fixed test")

    config_sha256 = sha256_file(Path(config["_config_path"]))
    context_config = config["features"]["context"]
    weights_sha256 = sha256_file(resolve(root, str(context_config["weights"])))
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_oof_checkpoint_contract(
        payload, canonical_ids=canonical.media_ids, train_ids=train_ids,
        validation_ids=validation_ids, config_sha256=config_sha256,
        weights_sha256=weights_sha256,
    )
    if tuple(payload.get("output_labels", ())) != tuple(config["task"]["output_labels"]):
        raise RuntimeError("checkpoint output label order mismatch")

    feature_root = resolve(root, paths["feature_store"]) / "train"
    annotations = resolve(root, paths["annotations"])
    preflight_errors = []
    for video_id in validation_ids:
        cache_path = feature_root / video_id / "timeline.npz"
        annotation_path = annotations / f"{canonical.annotation_id_by_media_id[video_id]}.json"
        if not annotation_path.is_file():
            preflight_errors.append(f"{video_id}: missing annotation")
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
        raise RuntimeError("OOF preflight failed:\n" + "\n".join(preflight_errors))

    model = TemporalLocator.from_config(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    timeline_hz = float(config["features"]["timeline_hz"])
    families = tuple(config["task"]["proposal_families"])
    labels = tuple(config["task"]["output_labels"])
    penalty_index = labels.index("penalty")
    raw_scores: list[float] = []
    raw_labels: list[bool] = []
    raw_candidates_by_video = []
    support = 0
    budget_tp = 0
    budget_fp = 0
    total_minutes = 0.0
    per_video = []
    for video_id in validation_ids:
        timeline = load_aligned_npz(feature_root / video_id / "timeline.npz")
        events = load_events(annotations / f"{canonical.annotation_id_by_media_id[video_id]}.json")
        output = infer_continuous_timeline(
            model, timeline, device=device,
            core_steps=int(round(args.core_seconds * timeline_hz)),
            context_steps=int(round(args.context_seconds * timeline_hz)),
        )
        if output.class_logits is None:
            raise RuntimeError("conditional class logits are missing")
        decode_kwargs = {
            "label_to_family": LABEL_TO_FAMILY,
            "family_nms_radius_seconds": config["decode"]["nms_radius_seconds"],
            "class_nms_radius_seconds": {
                label: (3.0 if label in {"shot", "save"} else 6.0) for label in labels
            },
            "local_search_radius_seconds": {
                label: (2.0 if label == "shot" else 4.0) for label in labels
            },
        }
        unconstrained = decode_family_conditioned_classes(
            output.logits, output.class_logits, timeline.timestamps, families, labels,
            max_class_per_minute={label: 1000.0 for label in labels}, **decode_kwargs,
        )["penalty"]
        budgeted = decode_family_conditioned_classes(
            output.logits, output.class_logits, timeline.timestamps, families, labels,
            max_class_per_minute={
                label: (0.5 if label == "penalty" else 1000.0) for label in labels
            }, **decode_kwargs,
        )["penalty"]
        event_times = events_for_label(events, "penalty")
        scores_one, labels_one, unmatched = scored_point_matches(
            unconstrained, event_times, tolerance_seconds=2.0
        )
        _, budget_labels, budget_unmatched = scored_point_matches(
            budgeted, event_times, tolerance_seconds=2.0
        )
        raw_scores.extend(scores_one)
        raw_labels.extend(labels_one)
        raw_candidates_by_video.append((unconstrained, event_times))
        support += len(event_times)
        budget_tp += sum(budget_labels)
        budget_fp += len(budget_labels) - sum(budget_labels)
        duration_minutes = max(float(timeline.timestamps[-1]) / 60.0, 1.0 / 60.0)
        total_minutes += duration_minutes
        per_video.append({
            "video_id": video_id,
            "penalty_support": len(event_times),
            "candidate_count": len(unconstrained),
            "candidate_tp_at_2s": sum(labels_one),
            "candidate_missed_at_2s": unmatched,
            "budget_proposal_count": len(budgeted),
            "budget_tp_at_2s": sum(budget_labels),
            "budget_missed_at_2s": budget_unmatched,
        })

    point_ap = {}
    for tolerance in (1.0, 2.0, 5.0):
        scores: list[float] = []
        matched: list[bool] = []
        for candidates, event_times in raw_candidates_by_video:
            scores_one, labels_one, _ = scored_point_matches(
                candidates, event_times, tolerance_seconds=tolerance
            )
            scores.extend(scores_one)
            matched.extend(labels_one)
        point_ap[str(tolerance)] = average_precision(
            torch.tensor(scores), torch.tensor(matched, dtype=torch.bool),
            positive_count=support,
        )
    operating = operating_point_at_recall(
        raw_scores, raw_labels, positive_count=support, target_recall=0.85,
        total_minutes=total_minutes,
    )
    report = {
        "schema": "football_longform_v2.penalty_oof_fold_report.v1",
        "evaluation_split": "penalty_oof_held_out_train_fold",
        "development_only": True,
        "fold": args.fold,
        "oof_manifest": str(oof_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "train_video_count": len(train_ids),
        "evaluated_video_count": len(validation_ids),
        "evaluated_video_ids": list(validation_ids),
        "evaluated_duration_minutes": total_minutes,
        "fixed_test_overlap_count": 0,
        "fixed_test_labels_or_predictions_used": False,
        "penalty": {
            "point_support": support,
            "candidate_ceiling_recall_at_2s": sum(raw_labels) / support if support else None,
            "budget_recall_at_2s": budget_tp / support if support else None,
            "budget_fp_per_minute_at_2s": budget_fp / total_minutes if total_minutes else None,
            "point_ap_at_tolerance": point_ap,
            "operating_point_for_target_recall_at_2s": operating,
            "raw_candidate_scores": raw_scores,
            "raw_candidate_match_labels_at_2s": raw_labels,
        },
        "per_video": per_video,
    }
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path), "fold": args.fold,
        "penalty": {key: value for key, value in report["penalty"].items() if not key.startswith("raw_")},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
