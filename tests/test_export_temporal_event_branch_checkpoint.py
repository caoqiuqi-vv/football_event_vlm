from __future__ import annotations

import unittest

from scripts.export_temporal_event_branch_checkpoint import resolve_event_metrics


class ExportTemporalEventBranchCheckpointTest(unittest.TestCase):
    def test_resolves_direct_eval_metrics(self) -> None:
        event = {"thresholds": {"shot": 0.4}}
        root = {"temporal_branches": {"event": event}}

        resolved_root, resolved_event = resolve_event_metrics(root)

        self.assertIs(resolved_root, root)
        self.assertIs(resolved_event, event)

    def test_resolves_epoch_summary_metrics(self) -> None:
        event = {"thresholds": {"shot": 0.4}}
        root = {"temporal_branches": {"event": event}}

        resolved_root, resolved_event = resolve_event_metrics(
            {"epoch": 1, "metrics": root}
        )

        self.assertIs(resolved_root, root)
        self.assertIs(resolved_event, event)

    def test_rejects_missing_event_metrics(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "temporal_branches.event"
        ):
            resolve_event_metrics({"metrics": {}})


if __name__ == "__main__":
    unittest.main()
