from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "football_e2e_spotter" / "src"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from football_e2e_spotter.goal_annotations import load_goal_annotations
from football_e2e_spotter.goal_matching import hungarian_indices
from football_e2e_spotter.goal_retriever import GoalLongContextRetriever
from football_e2e_spotter.goal_retriever_eval import intervals_union_seconds


def test_five_class_aliases_and_penalty_is_shot(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text(json.dumps([
        {"id": "a", "timestamp": 1, "label": "点球", "label_correct": True},
        {"id": "b", "timestamp": 2, "label": "中圈开球", "label_correct": True},
        {"id": "c", "timestamp": 3, "label": "扑救", "label_correct": True},
    ], ensure_ascii=False), encoding="utf-8")
    assert [event.label for event in load_goal_annotations(path)] == ["shot", "kickoff", "save"]


def test_adjacent_same_class_uses_two_slots() -> None:
    logits = torch.full((4, 6), -5.0)
    logits[0, 0] = 5.0; logits[1, 0] = 5.0
    times = torch.tensor([10.0, 10.5, 20.0, 30.0])
    rows, columns = hungarian_indices(
        logits, times, torch.ones(4), torch.tensor([0, 0]), torch.tensor([10.0, 10.5]),
    )
    assert len(set(rows.tolist())) == 2
    assert set(columns.tolist()) == {0, 1}


def test_simultaneous_shot_save_uses_independent_slots() -> None:
    logits = torch.full((3, 6), -5.0)
    logits[0, 0] = 5.0; logits[1, 1] = 5.0
    rows, columns = hungarian_indices(
        logits, torch.tensor([12.0, 12.0, 30.0]), torch.ones(3),
        torch.tensor([0, 1]), torch.tensor([12.0, 12.0]),
    )
    assert len(set(rows.tolist())) == 2
    assert set(columns.tolist()) == {0, 1}


def test_model_geometry_and_output_shapes() -> None:
    model = GoalLongContextRetriever(32, motion_dim=64, audio_dim=64, hidden_dim=64, decoder_layers=1, decoder_heads=4)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 180, 32), torch.randn(2, 180, 64), torch.randn(2, 180, 64))
    assert output["class_logits"].shape == (2, 60, 6)
    assert output["event_time"].shape == (2, 60)
    assert bool(((output["event_time"] >= -1.0) & (output["event_time"] <= 61.0)).all())


def test_review_time_is_union_not_candidate_sum() -> None:
    assert intervals_union_seconds([(0, 10), (5, 12), (20, 25), (20, 25)]) == 17

