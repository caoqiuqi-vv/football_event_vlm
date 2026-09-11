from __future__ import annotations

import unittest

from scripts.export_temporal_class_route_checkpoint import branch_thresholds


class TemporalClassRouteExportTest(unittest.TestCase):
    def test_branch_thresholds_supports_direct_metrics(self) -> None:
        payload = {
            "temporal_branches": {
                "event": {
                    "thresholds": {
                        "shot": 0.2,
                        "save": 0.3,
                        "set_piece": 0.4,
                    }
                }
            }
        }
        self.assertEqual(
            branch_thresholds(payload, "event"),
            {"shot": 0.2, "save": 0.3, "set_piece": 0.4},
        )

    def test_branch_thresholds_supports_checkpoint_metrics(self) -> None:
        payload = {
            "metrics": {
                "temporal_branches": {
                    "uniform": {
                        "thresholds": {
                            "shot": 0.1,
                            "save": 0.2,
                            "set_piece": 0.6,
                        }
                    }
                }
            }
        }
        self.assertEqual(branch_thresholds(payload, "uniform")["set_piece"], 0.6)


if __name__ == "__main__":
    unittest.main()
