from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PrecisionAutopilotTest(unittest.TestCase):
    def _run_decision(
        self, *, guard_pass: bool, fail_mode: str = "online_ohem"
    ) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_output = root / "current"
            current_output.mkdir()
            run_name = "fixture_event_run"
            (current_output / "hardneg_post_pipeline.log").write_text(
                "selected_event_checkpoint=/tmp/event.pt at=now\n"
                f"pipeline_completed_at=now event_run_name={run_name}\n",
                encoding="utf-8",
            )
            report_dir = root / "eval" / run_name
            report_dir.mkdir(parents=True)
            (report_dir / "vs_e1_recall_guard_1pp.json").write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "protocols": {
                                    "window_overlap": {
                                        "recall_guard_pass": guard_pass,
                                        "precision_improved": guard_pass,
                                    }
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            log = root / "autopilot.log"
            env = {
                **os.environ,
                "CURRENT_OUTPUT": str(current_output),
                "EVAL_ROOT": str(root / "eval"),
                "AUTOPILOT_LOG": str(log),
                "DRY_RUN": "true",
                "POLL_SEC": "1",
                "FAIL_MODE": fail_mode,
            }
            subprocess.run(
                ["bash", "scripts/run_uniform_event_precision_autopilot.sh"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            return log.read_text(encoding="utf-8")

    def test_pass_selects_adaptive_gate(self) -> None:
        log = self._run_decision(guard_pass=True)
        self.assertIn("selected_mode=adaptive_gate", log)
        self.assertIn("selected_init_checkpoint=/tmp/event.pt", log)
        self.assertIn("autopilot_dry_run_completed_at=", log)

    def test_failure_selects_online_ohem(self) -> None:
        log = self._run_decision(guard_pass=False)
        self.assertIn("selected_mode=online_ohem", log)
        self.assertIn("next_epochs=1", log)
        self.assertIn("autopilot_dry_run_completed_at=", log)

    def test_failure_can_select_online_rank(self) -> None:
        log = self._run_decision(
            guard_pass=False, fail_mode="online_rank"
        )
        self.assertIn("selected_mode=online_rank", log)
        self.assertIn(
            "selected_output_dir=outputs/football_events/"
            "vitl16_uniform_event_duration_online_rank_refine",
            log,
        )
        self.assertIn("autopilot_dry_run_completed_at=", log)

    def test_failure_can_select_duration_ohem(self) -> None:
        log = self._run_decision(
            guard_pass=False, fail_mode="duration_ohem"
        )
        self.assertIn("selected_mode=duration_ohem", log)
        self.assertIn(
            "selected_output_dir=outputs/football_events/"
            "vitl16_uniform_event_duration_ohem_refine",
            log,
        )
        self.assertIn("autopilot_dry_run_completed_at=", log)


if __name__ == "__main__":
    unittest.main()
