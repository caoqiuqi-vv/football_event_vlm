from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.check_football_global_resolution import balanced_subset_indices


class FootballResolutionCheckTest(unittest.TestCase):
    def test_balanced_subset_is_deterministic_and_covers_groups(self) -> None:
        records = []
        for video_id in ("a", "b"):
            for is_negative, labels in ((False, (1.0, 0.0, 0.0)), (False, (0.0, 1.0, 0.0)), (True, (0.0, 0.0, 0.0))):
                records.extend(
                    SimpleNamespace(video_id=video_id, is_negative=is_negative, labels=labels)
                    for _ in range(4)
                )

        first = balanced_subset_indices(records, max_samples=12, seed=42)
        second = balanced_subset_indices(records, max_samples=12, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        represented = {
            (records[index].video_id, records[index].is_negative, records[index].labels)
            for index in first
        }
        self.assertEqual(len(represented), 6)


if __name__ == "__main__":
    unittest.main()
