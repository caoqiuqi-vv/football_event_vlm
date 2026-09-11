from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from football_eval_semantics import SCORE_SEMANTICS_VERSION
from scripts.evaluate_football_model import (
    parse_args as parse_outer_eval_args,
    run_eval,
)
from scripts.prepare_full_review_dense_run import compatible_cache


class FootballEvalScoreSemanticsTest(unittest.TestCase):
    def test_old_summary_is_rejected_and_current_summary_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            checkpoint.touch()
            annotation = root / "v1.json"
            annotation.write_text("{}", encoding="utf-8")
            run_dir = root / "run"
            output = run_dir / "v1"
            output.mkdir(parents=True)
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            (output / "window_predictions.csv").write_text(
                "index,prob_shot\n", encoding="utf-8"
            )
            from scripts.evaluate_football_model import file_sha256

            summary = {
                "checkpoint": str(checkpoint.resolve()),
                "annotation_path": str(annotation.resolve()),
                "annotation_sha256": file_sha256(annotation),
                "clip_sec": 10.0,
                "stride_sec": 5.0,
                "image_size": [720, 1280],
                "match_tolerance_sec": 2.0,
                "prediction_postprocess": "window_overlap",
                "thresholds": {"shot": 0.5, "save": 0.5, "set_piece": 0.5},
                "score_source": "clip",
                "fail_on_zero_object_residual": False,
                "spatial_crop": {"mode": "none"},
            }
            (output / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            with patch(
                "sys.argv",
                [
                    "evaluate_football_model.py",
                    "--checkpoint", str(checkpoint.resolve()),
                    "--mode", "dense",
                ],
            ):
                outer_args = parse_outer_eval_args()

            with self.assertRaisesRegex(RuntimeError, "different evaluation protocol"):
                run_eval(
                    "v1",
                    outer_args,
                    ["shot", "save", "set_piece"],
                    run_dir,
                    "test",
                    root / "v1.mp4",
                    annotation,
                )

            prepare_args = SimpleNamespace(
                clip_sec=10.0,
                stride_sec=5.0,
                image_size="720,1280",
                match_tolerance_sec=2.0,
                prediction_postprocess="window_overlap",
                score_source="clip",
            )
            compatible, reason = compatible_cache(
                output,
                checkpoint=checkpoint,
                annotation=annotation,
                annotation_hash=summary["annotation_sha256"],
                args=prepare_args,
            )
            self.assertFalse(compatible)
            self.assertIn("score_semantics_version", reason)

            summary["score_semantics_version"] = SCORE_SEMANTICS_VERSION
            (output / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            with patch(
                "scripts.evaluate_football_model.subprocess.run"
            ) as subprocess_run:
                reused = run_eval(
                    "v1",
                    outer_args,
                    ["shot", "save", "set_piece"],
                    run_dir,
                    "test",
                    root / "v1.mp4",
                    annotation,
                )
            self.assertEqual(reused, output)
            subprocess_run.assert_not_called()

            compatible, reason = compatible_cache(
                output,
                checkpoint=checkpoint,
                annotation=annotation,
                annotation_hash=summary["annotation_sha256"],
                args=prepare_args,
            )
            self.assertTrue(compatible, reason)

            outer_args.fail_on_zero_object_residual = True
            with self.assertRaisesRegex(RuntimeError, "different evaluation protocol"):
                run_eval(
                    "v1",
                    outer_args,
                    ["shot", "save", "set_piece"],
                    run_dir,
                    "test",
                    root / "v1.mp4",
                    annotation,
                )

    def test_outer_evaluator_forwards_fail_on_zero_object_residual(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.evaluate_football_model.subprocess.run"
        ) as subprocess_run:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            checkpoint.touch()
            gt_path = root / "v1.json"
            gt_path.write_text("{}", encoding="utf-8")
            with patch(
                "sys.argv",
                [
                    "evaluate_football_model.py",
                    "--checkpoint", str(checkpoint),
                    "--mode", "dense",
                    "--fail-on-zero-object-residual",
                ],
            ):
                args = parse_outer_eval_args()
            run_eval(
                "v1",
                args,
                ["shot", "save", "set_piece"],
                root,
                "test",
                root / "v1.mp4",
                gt_path,
            )
            command = subprocess_run.call_args.args[0]
            self.assertIn("--fail-on-zero-object-residual", command)


if __name__ == "__main__":
    unittest.main()
