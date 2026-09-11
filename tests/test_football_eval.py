from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts import route_football_eval_runs
from scripts.eval_long_video_checkpoint import (
    RobustWindowCropper,
    compute_event_metrics,
    analyze_frame_event_outputs,
    random_topk_expected_min_offset,
    merge_predictions,
    parse_thresholds,
    point_nms_predictions,
    pred_matches_gt,
)
from scripts.evaluate_football_model import thresholds_compatible
from scripts.summarize_window_metrics import evaluate_video_label, parse_sweep
from train_football_events import (
    ConfigDict,
    FootballEvent,
    VideoEventClassifier,
    frame_detection_loss,
    gaussian_frame_targets,
)


class FootballEventEvaluationTest(unittest.TestCase):
    def test_runtime_dynamic_roi_overrides_checkpoint_crop_mode(self) -> None:
        args = SimpleNamespace(
            spatial_crop_mode="robust_detector_aware",
            detector_index_root="/tmp/runtime-roi-index",
            roi_temporal_mode="dynamic",
            roi_dynamic_context_sec=2.5,
            roi_temporal_smoothing_window_sec=1.25,
            roi_temporal_max_hold_sec=0.75,
            roi_temporal_confidence_decay_sec=0.5,
        )
        cfg = ConfigDict(
            {
                "spatial_crop": {
                    "index_root": "/tmp/checkpoint-roi-index",
                    "temporal_mode": "clip",
                    "dynamic_context_sec": 3.0,
                }
            }
        )
        runtime = RobustWindowCropper.from_args(args, (384, 640), cfg)
        self.assertIsNotNone(runtime)
        assert runtime is not None
        self.assertEqual(str(runtime.cropper.index_root), "/tmp/runtime-roi-index")
        self.assertEqual(runtime.cropper.temporal_mode, "dynamic")
        self.assertAlmostEqual(runtime.cropper.dynamic_context_sec, 2.5)
        self.assertAlmostEqual(runtime.cropper.temporal_smoothing_window_sec, 1.25)
        self.assertAlmostEqual(runtime.cropper.temporal_max_hold_sec, 0.75)
        self.assertAlmostEqual(runtime.cropper.temporal_confidence_decay_sec, 0.5)

    def test_point_matching_does_not_use_full_clip_extent(self) -> None:
        pred = {"label": "shot", "time_sec": 5.0, "start_sec": 0.0, "end_sec": 10.0}
        gt = {"label": "shot", "time_sec": 11.9, "start_sec": 11.9, "end_sec": 11.9}
        self.assertFalse(pred_matches_gt(pred, gt, 2.0, matching_mode="point"))
        self.assertTrue(pred_matches_gt(pred, gt, 2.0, matching_mode="interval"))

    def test_point_nms_does_not_chain_merge_an_arbitrary_interval(self) -> None:
        rows = [
            {"index": index, "start_sec": start, "end_sec": start + 10.0, "prob_shot": 0.9}
            for index, start in enumerate([0.0, 5.0, 10.0, 15.0])
        ]
        legacy = merge_predictions(rows, ["shot"], {"shot": 0.5}, merge_gap_sec=2.0)
        points = point_nms_predictions(rows, ["shot"], {"shot": 0.5}, nms_radius_sec=5.0)
        self.assertEqual((legacy[0]["start_sec"], legacy[0]["end_sec"]), (0.0, 25.0))
        self.assertEqual([item["time_sec"] for item in points], [5.0, 15.0])

    def test_point_metrics_are_one_to_one(self) -> None:
        predictions = [
            {"label": "shot", "time_sec": 10.0, "start_sec": 10.0, "end_sec": 10.0, "score": 0.9},
            {"label": "shot", "time_sec": 11.0, "start_sec": 11.0, "end_sec": 11.0, "score": 0.8},
        ]
        gt = [{"label": "shot", "time_sec": 10.5, "start_sec": 10.5, "end_sec": 10.5}]
        metrics = compute_event_metrics(predictions, gt, 2.0, matching_mode="point")
        self.assertEqual(metrics["per_class"]["shot"]["tp"], 1)
        self.assertEqual(metrics["per_class"]["shot"]["fp"], 1)
        self.assertEqual(metrics["per_class"]["shot"]["fn"], 0)

    def test_scalar_threshold_applies_to_every_label(self) -> None:
        thresholds = parse_thresholds("0.4", ["shot", "save"], {"shot": 0.2, "save": 0.3})
        self.assertEqual(thresholds, {"shot": 0.4, "save": 0.4})

    def test_route_eval_uses_per_label_source_probabilities_and_thresholds(self) -> None:
        def write_source(run_dir: Path, *, thresholds: dict[str, float], probs: dict[str, float]) -> None:
            video_dir = run_dir / "v1"
            video_dir.mkdir(parents=True)
            (video_dir / "summary.json").write_text(json.dumps({"thresholds": thresholds}))
            (video_dir / "gt_events.json").write_text(
                json.dumps(
                    [
                        {"label": "shot", "time_sec": 5.0, "start_sec": 5.0, "end_sec": 5.0},
                        {"label": "save", "time_sec": 5.0, "start_sec": 5.0, "end_sec": 5.0},
                        {"label": "set_piece", "time_sec": 5.0, "start_sec": 5.0, "end_sec": 5.0},
                    ]
                )
            )
            with (video_dir / "window_predictions.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["index", "start_sec", "end_sec", "prob_shot", "prob_save", "prob_set_piece"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "index": 0,
                        "start_sec": 0.0,
                        "end_sec": 10.0,
                        "prob_shot": probs["shot"],
                        "prob_save": probs["save"],
                        "prob_set_piece": probs["set_piece"],
                    }
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e1 = root / "e1"
            e2 = root / "e2"
            write_source(
                e1,
                thresholds={"shot": 0.9, "save": 0.9, "set_piece": 0.6},
                probs={"shot": 0.1, "save": 0.2, "set_piece": 0.8},
            )
            write_source(
                e2,
                thresholds={"shot": 0.5, "save": 0.7, "set_piece": 0.95},
                probs={"shot": 0.6, "save": 0.75, "set_piece": 0.1},
            )
            labels = ["shot", "save", "set_piece"]
            label_sources = {"shot": "e2", "save": "e2", "set_piece": "e1"}
            source_runs = {"e1": e1, "e2": e2}
            thresholds = route_football_eval_runs.load_thresholds(
                raw="checkpoint",
                labels=labels,
                label_sources=label_sources,
                source_runs=source_runs,
                video_ids=["v1"],
            )
            self.assertEqual(thresholds, {"shot": 0.5, "save": 0.7, "set_piece": 0.6})
            rows, gt_events = route_football_eval_runs.merge_video_rows(
                video_id="v1",
                labels=labels,
                label_sources=label_sources,
                source_runs=source_runs,
                thresholds=thresholds,
            )
            self.assertEqual(rows[0]["prob_shot"], 0.6)
            self.assertEqual(rows[0]["prob_save"], 0.75)
            self.assertEqual(rows[0]["prob_set_piece"], 0.8)
            self.assertEqual(rows[0]["source_set_piece"], "e1")
            predictions, matching_mode, allow_many = route_football_eval_runs.build_predictions(
                rows,
                labels=labels,
                thresholds=thresholds,
                postprocess="point_nms",
                nms_radius_sec=5.0,
                merge_gap_sec=2.0,
            )
            metrics = compute_event_metrics(
                predictions,
                gt_events,
                tolerance_sec=2.0,
                matching_mode=matching_mode,
                allow_many_predictions_per_gt=allow_many,
            )
            self.assertEqual(metrics["micro"]["tp"], 3)
            self.assertEqual(metrics["micro"]["fp"], 0)
            self.assertEqual(metrics["micro"]["fn"], 0)

    def test_cached_thresholds_must_match_requested_scalar(self) -> None:
        summary = {"thresholds": {"shot": 0.5, "save": 0.5, "set_piece": 0.5}}
        labels = ["shot", "save", "set_piece"]
        self.assertTrue(thresholds_compatible("0.5", summary, labels))
        self.assertFalse(thresholds_compatible("0.4", summary, labels))

    def test_overlapping_windows_are_independent_positive_samples(self) -> None:
        windows = [
            {"index": "0", "start_sec": "0", "end_sec": "10", "prob_shot": "0.9"},
            {"index": "1", "start_sec": "5", "end_sec": "15", "prob_shot": "0.4"},
            {"index": "2", "start_sec": "10", "end_sec": "20", "prob_shot": "0.8"},
        ]
        gt_events = [{"label": "shot", "time_sec": 7.0, "event_id": "shot-1"}]
        metrics, events = evaluate_video_label(windows, gt_events, "shot", 0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
        self.assertEqual(metrics["num_hit_gt_events"], 1)
        self.assertEqual(events[0]["num_covering_windows"], 2)
        self.assertEqual(events[0]["num_predicted_positive_windows"], 1)


    def test_gaussian_frame_targets_ignore_far_positive_frames(self) -> None:
        event = FootballEvent(
            source="test",
            video_id="v1",
            event_id="e1",
            event_type="",
            raw_label="shot",
            start_time=5.0,
            end_time=5.0,
            anchor_time=5.0,
            labels=(1.0, 0.0, 0.0, 0.0, 0.0),
        )
        targets, masks = gaussian_frame_targets(
            [event],
            0.0,
            10.0,
            [1.0, 5.0, 9.0],
            (1.0, 1.0, 1.0, 1.0, 1.0),
            {"shot": 1.0, "save": 1.0, "corner": 1.0, "freekick": 1.0, "penalty": 1.0},
            {"shot": 2.0, "save": 2.0, "corner": 2.0, "freekick": 2.0, "penalty": 2.0},
        )
        self.assertAlmostEqual(float(targets[1, 0]), 1.0, places=6)
        self.assertEqual(masks[:, 0].tolist(), [0.0, 1.0, 0.0])
        self.assertTrue(bool((targets[:, 1:] == 0).all()))
        self.assertTrue(bool((masks[:, 1:] == 1).all()))

    def test_event_topk_keeps_context_and_sorts_indices(self) -> None:
        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=4,
            hidden_dim=8,
            num_labels=2,
            fusion="event_topk_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=8,
            event_topk=2,
            context_frames=2,
        )
        x = torch.zeros(1, 8, 8)
        logits = torch.zeros(1, 8, 2)
        logits[0, 6, 0] = 5.0
        logits[0, 2, 1] = 4.0
        _, indices = model._select_event_topk(x, logits)
        self.assertEqual(indices.shape, (1, 4))
        self.assertEqual(indices[0].tolist(), sorted(indices[0].tolist()))
        self.assertTrue({0, 2, 6, 7}.issubset(set(indices[0].tolist())))

    def test_frame_detection_loss_is_finite_and_reports_components(self) -> None:
        outputs = {
            "frame_event_logits": torch.tensor(
                [
                    [
                        [1.0, -1.0, 0.0, 0.0, 0.0],
                        [3.0, -2.0, 0.0, 0.0, 0.0],
                        [-1.0, 1.5, 0.0, 0.0, 0.0],
                        [-2.0, 2.0, 0.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            )
        }
        batch = {
            "frame_targets": torch.tensor(
                [
                    [
                        [0.3, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.6, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),
            "frame_target_masks": torch.ones(1, 4, 5),
            "targets": torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]),
            "label_masks": torch.ones(1, 5),
        }
        cfg = ConfigDict(
            {
                "train": ConfigDict(
                    {
                        "frame_det_loss_type": "focal_bce",
                        "frame_mil_topk": 2,
                        "frame_heatmap_loss_weight": 0.5,
                        "frame_mil_loss_weight": 0.3,
                        "frame_rank_loss_weight": 0.2,
                    }
                )
            }
        )
        loss, components = frame_detection_loss(outputs, batch, cfg, torch.device("cpu"))
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(loss), 0.0)
        self.assertIn("frame_heatmap_loss", components)
        self.assertIn("frame_mil_loss", components)
        self.assertIn("frame_rank_loss", components)

    def test_temporal_heads_return_auxiliary_frame_outputs(self) -> None:
        for fusion in ("cls_transformer", "attn_pool_transformer", "class_query_transformer", "event_topk_transformer"):
            model = VideoEventClassifier(
                backbone=None,
                frame_feature_dim=4,
                hidden_dim=8,
                num_labels=3,
                fusion=fusion,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
                max_frames=8,
                event_topk=3,
                context_frames=1,
            )
            outputs = model(torch.randn(2, 8, 4), return_aux=True)
            self.assertEqual(tuple(outputs["logits"].shape), (2, 3))
            self.assertEqual(tuple(outputs["frame_event_logits"].shape), (2, 8, 3))
            if fusion == "event_topk_transformer":
                self.assertEqual(tuple(outputs["topk_indices"].shape), (2, 4))
            if fusion in {"attn_pool_transformer", "class_query_transformer"}:
                self.assertIn("frame_attention", outputs)

    def test_frame_event_localization_compares_topk_with_random_baseline(self) -> None:
        frame_rows = []
        scores = [-2.0, -1.0, 4.0, 0.0]
        for index, score in enumerate(scores):
            frame_rows.append(
                {
                    "window_index": 0,
                    "start_sec": 0.0,
                    "end_sec": 4.0,
                    "frame_index": index,
                    "frame_time_sec": float(index),
                    "frame_logit_shot": score,
                    "frame_prob_shot": float(torch.sigmoid(torch.tensor(score))),
                }
            )
        gt = [{"label": "shot", "time_sec": 2.0, "event_id": "shot-1"}]
        matches = [
            {
                "label": "shot",
                "pred_start_sec": 0.0,
                "pred_end_sec": 4.0,
                "gt_time_sec": 2.0,
                "gt_event_id": "shot-1",
            }
        ]
        analysis, samples, window_scores = analyze_frame_event_outputs(
            frame_rows,
            gt,
            matches,
            ["shot"],
            topk=2,
            sigma_by_label={"shot": 0.8},
        )
        matched = analysis["localization"]["matched_windows"]["global"]["shot"]
        self.assertEqual(matched["num_samples"], 1)
        self.assertEqual(matched["top1_abs_offset_sec"], 0.0)
        self.assertEqual(matched["topk_min_abs_offset_sec"], 0.0)
        self.assertGreater(matched["topk_min_gain_vs_random_sec"], 0.0)
        self.assertEqual(len(samples), 2)
        self.assertEqual(len(window_scores), 1)

    def test_random_topk_expected_min_offset(self) -> None:
        self.assertAlmostEqual(random_topk_expected_min_offset([0.0, 2.0], 1), 1.0)
        self.assertAlmostEqual(random_topk_expected_min_offset([0.0, 2.0], 2), 0.0)

    def test_threshold_sweep_is_inclusive(self) -> None:
        self.assertEqual(parse_sweep("0.1:0.3:0.1"), [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
