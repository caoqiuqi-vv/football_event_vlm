from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_football_temporal_branch_metrics import load_branch_metrics


class CompareFootballTemporalBranchMetricsTest(unittest.TestCase):
    def _write(self, root: Path, payload: dict) -> Path:
        path = root / "metrics.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_event_metrics_from_epoch_summary(self) -> None:
        per_class = {
            "shot": {"precision": 0.7, "recall": 0.8}
        }
        branch = {
            "tuned": {"mAP": 0.6, "per_class": per_class},
            "thresholds": {"shot": 0.4},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(
                Path(temp_dir),
                {
                    "epoch": 2,
                    "metrics": {
                        "temporal_branches": {"event": branch}
                    },
                },
            )

            loaded = load_branch_metrics(path, "event", "tuned")

        self.assertEqual(loaded["epoch"], 2)
        self.assertEqual(loaded["metrics"]["per_class"], per_class)
        self.assertEqual(loaded["thresholds"]["shot"], 0.4)

    def test_loads_event_metrics_from_direct_eval(self) -> None:
        branch = {
            "tuned": {
                "mAP": 0.6,
                "per_class": {
                    "shot": {"precision": 0.7, "recall": 0.8}
                },
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(
                Path(temp_dir),
                {"temporal_branches": {"event": branch}},
            )

            loaded = load_branch_metrics(path, "event", "tuned")

        self.assertEqual(loaded["metrics"]["mAP"], 0.6)

    def test_rejects_missing_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(Path(temp_dir), {"metrics": {}})
            with self.assertRaisesRegex(
                ValueError, "temporal_branches.event"
            ):
                load_branch_metrics(path, "event", "tuned")


if __name__ == "__main__":
    unittest.main()
