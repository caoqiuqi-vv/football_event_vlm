from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from aggregate_penalty_oof import aggregate_oof_reports, wilson_interval  # noqa: E402
from build_penalty_oof_folds import build_balanced_folds  # noqa: E402
from evaluate_penalty_oof_fold import validate_oof_checkpoint_contract  # noqa: E402
from run_penalty_oof import build_commands  # noqa: E402
from train_locator import parse_data_parallel_device_ids  # noqa: E402
from football_longform_v2.models.temporal_locator import MODEL_SCHEMA  # noqa: E402
from build_official_feature_store import (  # noqa: E402
    atomic_json,
    cache_claim,
    cache_duration_validation,
    shard_items,
)
from aggregate_official_feature_store import canonical_run_sha256  # noqa: E402
from watch_official_fullscale_gate import (  # noqa: E402
    checkpoint_selection_key,
    diagnose_calibration_report,
    manifest_contract_error,
    run_logged,
)
from coordinate_official_feature_store import expected_worker_paths, worker_terminal_error  # noqa: E402


class ShardedFeatureStoreTests(unittest.TestCase):
    def test_penalty_oof_runner_uses_main_selected_epoch_and_cuda0(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = root / "configs/lf_a0_official_fullscale.yaml"
        oof_path = root / "experiments/lf_a0_official_fullscale/penalty_oof_folds.json"
        oof = json.loads(oof_path.read_text())
        jobs, aggregate_command, aggregate_output = build_commands(
            root=root, config_path=config, oof_path=oof_path,
            selection={"selected": {"epoch": 1}}, oof=oof,
            device="cuda:0", workers=4,
        )
        self.assertEqual(len(jobs), 5)
        self.assertTrue(all(job["fixed_epoch"] == 1 for job in jobs))
        self.assertTrue(all("--epochs" in job["train_command"] for job in jobs))
        self.assertTrue(all("2" in job["train_command"] for job in jobs))
        self.assertTrue(all("cuda:0" in job["train_command"] for job in jobs))
        self.assertIn("aggregate_penalty_oof.py", " ".join(aggregate_command))
        self.assertEqual(aggregate_output.name, "penalty_oof_aggregate.json")

    def test_penalty_oof_aggregate_is_recall_first_and_not_deployment_eligible(self) -> None:
        oof = {
            "schema": "football_longform_v2.penalty_oof_folds.v1",
            "source_train_video_count": 4,
            "source_penalty_event_count": 4,
            "folds": [
                {"fold": 0, "validation_media_ids": ["a", "b"], "validation_penalty_event_count": 2},
                {"fold": 1, "validation_media_ids": ["c", "d"], "validation_penalty_event_count": 2},
            ],
        }
        def report(fold: int, ids: list[str], scores: list[float], labels: list[bool]) -> dict:
            return {
                "schema": "football_longform_v2.penalty_oof_fold_report.v1",
                "development_only": True,
                "fold": fold,
                "evaluated_video_ids": ids,
                "evaluated_duration_minutes": 10.0,
                "fixed_test_overlap_count": 0,
                "fixed_test_labels_or_predictions_used": False,
                "penalty": {
                    "point_support": 2,
                    "raw_candidate_scores": scores,
                    "raw_candidate_match_labels_at_2s": labels,
                },
            }
        reports = [
            report(0, ["a", "b"], [0.9, 0.8, 0.7], [True, True, False]),
            report(1, ["c", "d"], [0.85, 0.75, 0.6], [True, True, False]),
        ]
        result = aggregate_oof_reports(oof, reports)
        self.assertTrue(result["operating_point_for_target_recall_at_2s"]["target_achieved"])
        self.assertEqual(result["operating_point_for_target_recall_at_2s"]["precision"], 1.0)
        self.assertEqual(result["worst_fold_recall_at_global_oof_threshold"], 1.0)
        self.assertFalse(result["deployment_threshold_eligible"])
        self.assertIsNotNone(wilson_interval(4, 4))

    def test_penalty_oof_checkpoint_contract_rejects_fold_leakage(self) -> None:
        payload = {
            "model_schema": MODEL_SCHEMA,
            "training_media_ids": ("a", "b", "c"),
            "held_out_train_video_ids": ("d",),
            "train_video_count": 3,
            "canonical_train_video_count": 4,
            "missing_train_video_ids": (),
            "feature_split": "train",
            "feature_config_sha256": "config",
            "feature_weights_sha256": "weights",
        }
        kwargs = {
            "canonical_ids": ("a", "b", "c", "d"),
            "train_ids": ("a", "b", "c"),
            "validation_ids": ("d",),
            "config_sha256": "config",
            "weights_sha256": "weights",
        }
        validate_oof_checkpoint_contract(payload, **kwargs)
        leaked = {**kwargs, "train_ids": ("a", "b", "c", "d")}
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            validate_oof_checkpoint_contract(payload, **leaked)
        wrong_checkpoint_ids = {**payload, "training_media_ids": ("a", "b")}
        with self.assertRaisesRegex(RuntimeError, "training IDs"):
            validate_oof_checkpoint_contract(wrong_checkpoint_ids, **kwargs)

    def test_penalty_oof_folds_are_deterministic_disjoint_and_event_balanced(self) -> None:
        rows = [(f"video-{index}", 1 if index < 5 else 0) for index in range(15)]
        first = build_balanced_folds(rows, fold_count=5, seed=42)
        second = build_balanced_folds(rows, fold_count=5, seed=42)
        self.assertEqual(first, second)
        flattened = [video_id for fold in first for video_id in fold]
        self.assertEqual(len(flattened), 15)
        self.assertEqual(len(set(flattened)), 15)
        positives = {video_id for video_id, count in rows if count > 0}
        self.assertEqual([len(set(fold) & positives) for fold in first], [1] * 5)

    def test_static_shards_are_disjoint_and_cover_canonical_153(self) -> None:
        items = [("train", str(index)) for index in range(135)] + [
            ("calibration", str(index)) for index in range(18)
        ]
        workers = [shard_items(items, num_shards=2, shard_index=index) for index in range(2)]
        assigned = [global_index for worker in workers for global_index, _, _ in worker]
        self.assertEqual(len(assigned), 153)
        self.assertEqual(set(assigned), set(range(153)))
        self.assertTrue(set(row[0] for row in workers[0]).isdisjoint(row[0] for row in workers[1]))
        self.assertEqual({row[0] % 2 for row in workers[0]}, {0})
        self.assertEqual({row[0] % 2 for row in workers[1]}, {1})

    def test_worker_manifests_are_independent(self) -> None:
        names = {f"manifests/run-a/worker-{index}.json" for index in range(2)}
        self.assertEqual(names, {"manifests/run-a/worker-0.json", "manifests/run-a/worker-1.json"})

    def test_atomic_manifest_and_duration_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            atomic_json(manifest, {"records": ["cache_truth"]})
            self.assertEqual(json.loads(manifest.read_text())["records"], ["cache_truth"])
            self.assertFalse(list(root.glob(".manifest.json.*.tmp")))
            cache = root / "timeline.npz"
            np.savez(cache, timestamps=np.asarray([0.0, 10.0], dtype=np.float32))
            self.assertEqual(cache_duration_validation(cache, source_duration_s=10.1), (True, "valid"))
            valid, reason = cache_duration_validation(cache, source_duration_s=20.0)
            self.assertFalse(valid)
            self.assertIn("truncated cache", reason)


    def test_claim_is_exclusive_and_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with cache_claim(root, "train", "video-a") as (claimed, claim):
                self.assertTrue(claimed)
                probe = subprocess.run(
                    [sys.executable, "-c",
                     "import fcntl,sys; f=open(sys.argv[1],\"a+\"); "
                     "\ntry: fcntl.flock(f.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB) "
                     "\nexcept BlockingIOError: raise SystemExit(0) "
                     "\nraise SystemExit(1)", str(claim)],
                    check=False,
                )
                self.assertEqual(probe.returncode, 0)
            self.assertTrue(claim.is_file())


    def test_data_parallel_device_contract(self) -> None:
        self.assertEqual(parse_data_parallel_device_ids("6,7", "cuda:6"), (6, 7))
        self.assertEqual(parse_data_parallel_device_ids(None, "cpu"), ())
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_data_parallel_device_ids("6,6", "cuda:6")
        with self.assertRaisesRegex(ValueError, "first data-parallel"):
            parse_data_parallel_device_ids("6,7", "cuda:0")

    def test_run_logged_persists_subprocess_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "gate.log"
            run_logged(
                [sys.executable, "-c", "print(12345)"],
                cwd=root, environment={}, log_path=log_path,
            )
            content = log_path.read_text()
            self.assertIn("launch=", content)
            self.assertIn("12345", content)

    def test_gate_manifest_contract_requires_exact_fresh_ready_run(self) -> None:
        expected = [(0, "train", "a"), (1, "calibration", "b")]
        state = {
            "aggregation": "cache_validation_truth", "run_id": "run-a",
            "config_sha256": "config", "weights_sha256": "weights",
            "canonical_run_sha256": canonical_run_sha256([( "train", "a"), ("calibration", "b")]),
            "updated_unix": 100.0, "total": 2, "processed": 2, "completed": 2,
            "skipped_valid": 2, "failed": 0, "missing": 0, "invalid_cache": 0,
            "records": [
                {"global_index": 0, "split": "train", "video_id": "a", "status": "skipped_valid"},
                {"global_index": 1, "split": "calibration", "video_id": "b", "status": "skipped_valid"},
            ],
        }
        kwargs = {"expected_records": expected, "expected_config_sha256": "config",
                  "expected_weights_sha256": "weights", "required_run_id": "run-a",
                  "max_age_seconds": 10.0, "now": 105.0}
        self.assertIsNone(manifest_contract_error(state, **kwargs))
        wrong_ids = {**state, "records": list(reversed(state["records"]))}
        self.assertEqual(manifest_contract_error(wrong_ids, **kwargs), "record IDs/order mismatch")
        stale = {**state, "updated_unix": 1.0}
        self.assertEqual(manifest_contract_error(stale, **kwargs), "manifest is stale or has invalid updated_unix")
        wrong_hash = {**state, "weights_sha256": "other"}
        self.assertEqual(manifest_contract_error(wrong_hash, **kwargs), "weights_sha256 mismatch")


    def test_checkpoint_selection_prioritizes_shot_gate_then_other_targets(self) -> None:
        def report(shot_recall: float, other_recall: float) -> dict:
            def metric(recall: float, target: float) -> dict:
                return {
                    "point_support": 10, "budget_recall_at_2s": recall,
                    "point_ap_at_tolerance": {"2.0": recall},
                    "operating_point_for_target_recall_at_2s": {
                        "target_achieved": recall >= target, "precision": recall,
                    },
                }
            return {"labels": {
                "shot": metric(shot_recall, 0.90), "save": metric(other_recall, 0.85),
                "corner": metric(other_recall, 0.85), "freekick": metric(other_recall, 0.85),
                "penalty": {"point_support": 0},
            }}
        balanced = report(0.91, 0.86)
        shot_below_gate = report(0.89, 0.99)
        shot_only = report(0.96, 0.70)
        self.assertGreater(
            checkpoint_selection_key(balanced), checkpoint_selection_key(shot_below_gate)
        )
        self.assertGreater(
            checkpoint_selection_key(balanced), checkpoint_selection_key(shot_only)
        )

    def test_checkpoint_selection_maximizes_precision_after_recall_targets(self) -> None:
        def metric(*, recall: float, precision: float, achieved: bool = True) -> dict:
            return {
                "point_support": 10,
                "budget_recall_at_2s": recall,
                "point_ap_at_tolerance": {"2.0": 0.5},
                "operating_point_for_target_recall_at_2s": {
                    "target_achieved": achieved,
                    "precision": precision,
                },
            }

        precise = {"labels": {
            "shot": metric(recall=0.90, precision=0.60),
            "save": metric(recall=0.85, precision=0.40),
            "corner": metric(recall=0.85, precision=0.40),
        }}
        excess_recall_low_precision = {"labels": {
            "shot": metric(recall=0.98, precision=0.10),
            "save": metric(recall=0.95, precision=0.10),
            "corner": metric(recall=0.95, precision=0.10),
        }}
        self.assertGreater(
            checkpoint_selection_key(precise),
            checkpoint_selection_key(excess_recall_low_precision),
        )

    def test_gate_diagnostics_separates_candidate_ranking_and_support_failures(self) -> None:
        def metric(candidate: float, *, achieved: bool, recall: float) -> dict:
            return {
                "point_support": 10,
                "candidate_ceiling_recall_at_2s": candidate,
                "budget_recall_at_2s": recall,
                "operating_point_for_target_recall_at_2s": {
                    "target_achieved": achieved,
                    "recall": recall,
                    "precision": 0.4 if achieved else None,
                    "fp_per_minute": 1.0 if achieved else None,
                },
            }

        report = {"evaluation_split": "calibration", "labels": {
            "shot": metric(0.95, achieved=True, recall=0.91),
            "save": metric(0.80, achieved=False, recall=0.70),
            "corner": metric(0.95, achieved=False, recall=0.80),
            "freekick": metric(0.90, achieved=True, recall=0.86),
            "penalty": {"point_support": 0},
        }}
        result = diagnose_calibration_report(report)
        self.assertFalse(result["all_required_recall_targets_verified"])
        self.assertFalse(result["external_test_allowed_for_final_comparison"])
        self.assertEqual(result["labels"]["shot"]["bottleneck"], "target_met_optimize_precision")
        self.assertEqual(result["labels"]["save"]["bottleneck"], "candidate_recall_bottleneck")
        self.assertEqual(result["labels"]["corner"]["bottleneck"], "ranking_or_budget_bottleneck")
        self.assertEqual(
            result["labels"]["penalty"]["bottleneck"],
            "unverified_no_calibration_support",
        )

    def test_coordinator_validates_expected_worker_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = expected_worker_paths(root, "run-a", 2)
            self.assertEqual([path.name for path in paths], ["worker-0.json", "worker-1.json"])
            for index, path in enumerate(paths):
                atomic_json(path, {"run_id": "run-a", "num_shards": 2, "shard_index": index, "failed": 0})
            self.assertIsNone(worker_terminal_error(paths, run_id="run-a", num_shards=2))
            atomic_json(paths[1], {"run_id": "run-a", "num_shards": 2, "shard_index": 1, "failed": 1})
            self.assertEqual(
                worker_terminal_error(paths, run_id="run-a", num_shards=2),
                "worker-1 reports failed=1",
            )


if __name__ == "__main__":
    unittest.main()
