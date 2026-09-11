from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import train_football_events as base
import train_football_events_online_simulation_e13 as e13


def record(video_id: str, sample_id: str = "seed") -> base.LongVideoRecord:
    return base.LongVideoRecord(
        source="test", split="train", video_id=video_id, sample_id=sample_id,
        video_path=f"/{video_id}.mp4", annotation_path=f"/{video_id}.json",
        anchor_time=-1.0, base_clip_start=0.0, base_clip_end=10.0,
        video_duration=120.0, is_negative=True, labels=(0.0, 0.0, 0.0),
        label_mask=(1.0, 1.0, 1.0),
    )


def event(
    video_id: str, event_id: str, anchor: float, labels: tuple[float, ...],
    *, context: tuple[float, float] | None = None, ignored: bool = False,
) -> base.FootballEvent:
    return base.FootballEvent(
        source="test", video_id=video_id, event_id=event_id,
        event_type="", raw_label="", start_time=anchor, end_time=anchor,
        anchor_time=anchor, labels=labels,
        context_start_time=context[0] if context else anchor,
        context_end_time=context[1] if context else anchor,
        is_ignored=ignored,
    )


def config() -> base.ConfigDict:
    return base.to_config({
        "seed": 7,
        "task": {"label_schema": "set_piece"},
        "video": {"clip_duration": 10.0},
        "data": {"long_video": {"online_simulation": {
            "enabled": True,
            "chunk_duration_sec": 40.0,
            "window_stride_sec": 5.0,
            "event_pairs_per_epoch": 2,
            "clean_background_windows_per_epoch": 2,
            "event_chunk_fraction": 0.5,
            "primary_positive_min_sec": 3.0,
            "primary_positive_max_sec": 7.0,
            "edge_position_min_sec": 0.5,
            "edge_position_max_sec": 2.0,
            "pair_min_shift_sec": 2.0,
            "edge_clip_loss_weight": 0.35,
            "edge_frame_loss_weight": 0.5,
            "near_event_ignore_sec": 2.0,
            "near_event_max_sec": 8.0,
            "near_event_negative_weight": 0.5,
            "background_max_attempts": 1000,
        }}},
        "raw_set_piece_supervision": {
            "enabled": True,
            "context_span_min_duration_sec": 0.5,
            "rejected_ignore_enabled": True,
            "rejected_ignore_margin_sec": 5.0,
        },
    })


class OnlineSimulationE13Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base.configure_label_schema(config())

    def test_builder_emits_unique_central_edge_pairs(self) -> None:
        rows = [record("v1"), record("v2")]
        events = {
            ("test", "v1"): [event("v1", "shot1", 50.0, (1.0, 0.0, 0.0))],
            ("test", "v2"): [event("v2", "shot2", 60.0, (1.0, 0.0, 0.0))],
        }
        built = e13.build_online_records_e13(rows, events, config(), epoch=0)
        paired = [row for row in built if row.online_pair_id]
        self.assertEqual(len(paired), 4)
        groups: dict[str, list[base.LongVideoRecord]] = {}
        for row in paired:
            groups.setdefault(row.online_pair_id, []).append(row)
        self.assertEqual(len(groups), 2)
        linked = [row for row in built if row.online_chunk_id in groups and not row.online_pair_id]
        self.assertEqual(len(linked), 2)
        self.assertTrue(all(row.online_negative_kind == "near_event_context" for row in linked))
        for pair_rows in groups.values():
            self.assertEqual({row.online_pair_role for row in pair_rows}, {"central", "edge"})
            self.assertEqual(len({row.video_id for row in pair_rows}), 1)
            central = next(row for row in pair_rows if row.online_pair_role == "central")
            edge = next(row for row in pair_rows if row.online_pair_role == "edge")
            central_relative = central.anchor_time - central.base_clip_start
            edge_relative = edge.anchor_time - edge.base_clip_start
            self.assertGreaterEqual(central_relative, 3.0)
            self.assertLessEqual(central_relative, 7.0)
            self.assertTrue(edge_relative <= 2.0 or edge_relative >= 8.0)
            self.assertGreaterEqual(abs(central.base_clip_start - edge.base_clip_start), 2.0)
            self.assertEqual(central.online_pair_class_mask, (1.0, 0.0, 0.0))
            self.assertEqual(edge.online_clip_loss_weights[0], 0.35)
            near = next(row for row in linked if row.online_chunk_id == central.online_pair_id)
            self.assertEqual(near.online_pair_id, "")
            self.assertTrue(near.is_negative)
            anchor = central.anchor_time
            gap = (anchor - near.base_clip_end if near.base_clip_end < anchor else near.base_clip_start - anchor)
            self.assertGreaterEqual(gap, 5.0)
            self.assertLessEqual(gap, 15.0)
            self.assertFalse(e13._window_conflicts_support(
                events[("test", central.video_id)], near.base_clip_start, near.base_clip_end,
                min_context_duration_sec=0.5, rejected_margin_sec=5.0,
            ))

    def test_dataset_preserves_online_pair_base_window(self) -> None:
        dataset = object.__new__(base.FootballLongVideoDataset)
        dataset.is_train = True
        dataset.clip_duration = 10.0
        central = replace(
            record("v1"), sample_id="pair_c", is_negative=False,
            labels=(1.0, 0.0, 0.0), anchor_time=50.0,
            base_clip_start=46.0, base_clip_end=56.0,
            online_pair_id="pair0", online_pair_role="central",
        )
        edge = replace(
            central, sample_id="pair_e", base_clip_start=49.0,
            base_clip_end=59.0, online_pair_role="edge",
        )
        self.assertEqual(dataset._sample_window(central, []), (46.0, 56.0))
        self.assertEqual(dataset._sample_window(edge, []), (49.0, 59.0))

    def test_raw_context_and_rejected_zone_are_never_negative(self) -> None:
        accepted = event(
            "v1", "sp1", 50.0, (0.0, 0.0, 1.0), context=(40.0, 55.0)
        )
        rejected = event(
            "v1", "sp_bad", 75.0, (0.0, 0.0, 1.0),
            context=(72.0, 78.0), ignored=True,
        )
        # 42--48 overlaps the accepted raw span but not the reviewed anchor.
        weights = e13.e13_window_weights(
            [accepted, rejected], (0.0, 0.0, 0.0), 42.0, 48.0, 2,
            mode="paired_event", role="", min_context_duration_sec=0.5,
            rejected_ignore_margin_sec=5.0, near_event_ignore_sec=2.0,
            near_event_max_sec=8.0, near_event_negative_weight=0.5,
            edge_clip_weight=0.35, edge_frame_weight=0.5,
        )
        self.assertEqual(weights, (0.0, 0.0, "accepted_context_ignore"))
        # 67--70 is inside the rejected span's 5 s safety margin.
        weights = e13.e13_window_weights(
            [accepted, rejected], (0.0, 0.0, 0.0), 67.0, 70.0, 2,
            mode="paired_event", role="", min_context_duration_sec=0.5,
            rejected_ignore_margin_sec=5.0, near_event_ignore_sec=2.0,
            near_event_max_sec=8.0, near_event_negative_weight=0.5,
            edge_clip_weight=0.35, edge_frame_weight=0.5,
        )
        self.assertEqual(weights, (0.0, 0.0, "rejected_context_ignore"))

    def test_sampler_keeps_pairs_and_uses_cross_video_clean(self) -> None:
        v1, v2 = record("v1"), record("v2")
        rows = [
            replace(v1, sample_id="p1c", online_pair_id="p1", online_pair_role="central",
                    online_pair_class_mask=(1.0, 0.0, 0.0), labels=(1.0, 0.0, 0.0), is_negative=False),
            replace(v1, sample_id="p1e", online_pair_id="p1", online_pair_role="edge",
                    online_pair_class_mask=(1.0, 0.0, 0.0), labels=(1.0, 0.0, 0.0), is_negative=False),
            replace(v2, sample_id="p2c", online_pair_id="p2", online_pair_role="central",
                    online_pair_class_mask=(1.0, 0.0, 0.0), labels=(1.0, 0.0, 0.0), is_negative=False),
            replace(v2, sample_id="p2e", online_pair_id="p2", online_pair_role="edge",
                    online_pair_class_mask=(1.0, 0.0, 0.0), labels=(1.0, 0.0, 0.0), is_negative=False),
            replace(v1, sample_id="p1n", online_chunk_id="p1",
                    online_negative_kind="near_event_context"),
            replace(v2, sample_id="p2n", online_chunk_id="p2",
                    online_negative_kind="near_event_context"),
            replace(v2, sample_id="clean_v2", online_negative_kind="clean_background"),
            replace(v1, sample_id="clean_v1", online_negative_kind="clean_background"),
        ]
        dataset = SimpleNamespace(records=rows)
        sampler = e13.OnlinePairDistributedSampler(
            dataset, replicas=1, rank=0, seed=3, drop_last=True, batch_size=4
        )
        ordered = list(iter(sampler))
        for offset in range(0, len(ordered), 4):
            batch_rows = [rows[index] for index in ordered[offset:offset + 4]]
            pair_ids = {row.online_pair_id for row in batch_rows if row.online_pair_id}
            self.assertEqual(len(pair_ids), 1)
            pair_id = next(iter(pair_ids))
            pair_rows = [row for row in batch_rows if row.online_pair_id == pair_id]
            self.assertEqual({row.online_pair_role for row in pair_rows}, {"central", "edge"})
            pair_video = pair_rows[0].video_id
            linked_rows = [row for row in batch_rows if row.online_chunk_id == pair_id and not row.online_pair_id]
            self.assertEqual(len(linked_rows), 1)
            self.assertEqual(linked_rows[0].online_negative_kind, "near_event_context")
            self.assertTrue(any(
                row.online_negative_kind == "clean_background" and row.video_id != pair_video
                for row in batch_rows
            ))

    def test_online_validation_grid_is_exact_and_carries_gt_anchors(self) -> None:
        rows = [record("v1")]
        events = {
            ("test", "v1"): [event("v1", "shot1", 50.0, (1.0, 0.0, 0.0))]
        }
        built = e13.build_online_eval_records(
            rows,
            events,
            config(),
            online_cfg=base.to_config({"window_stride_sec": 5.0}),
            sample_prefix="online_val",
        )
        self.assertEqual(len(built), 23)
        self.assertEqual(built[0].base_clip_start, 0.0)
        self.assertEqual(built[-1].base_clip_start, 110.0)
        self.assertTrue(all(row.online_chunk_mode == "evaluation" for row in built))
        containing = [
            row for row in built if any(row.online_gt_anchors[0])
        ]
        self.assertEqual(len(containing), 2)
        dataset = object.__new__(base.FootballLongVideoDataset)
        dataset.is_train = False
        dataset.clip_duration = 10.0
        self.assertEqual(
            dataset._sample_window(containing[0], events[("test", "v1")]),
            (containing[0].base_clip_start, containing[0].base_clip_end),
        )


if __name__ == "__main__":
    unittest.main()
