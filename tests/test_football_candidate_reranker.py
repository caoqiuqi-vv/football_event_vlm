from __future__ import annotations

import unittest

from scripts.fit_football_candidate_reranker import (
    Candidate,
    FEATURE_NAMES,
    evaluate_predictions,
    fit_oof_models,
    probability_logit,
)


def make_candidate(
    video_id: str,
    window_index: int,
    *,
    target: int | None,
    clip_prob: float,
    peak_time_sec: float,
) -> Candidate:
    feature = probability_logit(clip_prob)
    return Candidate(
        video_id=video_id,
        label="shot",
        window_index=window_index,
        window_start_sec=float(window_index * 5),
        window_end_sec=float(window_index * 5 + 10),
        center_time_sec=float(window_index * 5 + 5),
        peak_time_sec=peak_time_sec,
        clip_prob=clip_prob,
        features=(feature, feature, feature, 0.1, 0.0, feature, 1.0),
        target=target,
        nearest_gt_distance_sec=0.0 if target == 1 else 20.0,
    )


class FootballCandidateRerankerTest(unittest.TestCase):
    def test_feature_schema_is_stable(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 7)
        self.assertAlmostEqual(probability_logit(0.5), 0.0)

    def test_oof_scores_every_candidate_without_group_leakage(self) -> None:
        candidates: list[Candidate] = []
        for video_index in range(4):
            video_id = f"v{video_index}"
            candidates.extend(
                [
                    make_candidate(
                        video_id,
                        0,
                        target=1,
                        clip_prob=0.9,
                        peak_time_sec=5.0,
                    ),
                    make_candidate(
                        video_id,
                        1,
                        target=0,
                        clip_prob=0.1,
                        peak_time_sec=15.0,
                    ),
                    make_candidate(
                        video_id,
                        2,
                        target=None,
                        clip_prob=0.5,
                        peak_time_sec=25.0,
                    ),
                ]
            )
        scores, folds = fit_oof_models(
            candidates,
            ["shot"],
            folds=2,
            regularization_c=0.1,
        )
        self.assertEqual(set(scores), {candidate.key for candidate in candidates})
        for fold in folds["shot"]:
            self.assertFalse(set(fold["train_videos"]) & set(fold["val_videos"]))

    def test_point_nms_metrics_use_one_to_one_matching(self) -> None:
        candidates = [
            make_candidate("v1", 0, target=1, clip_prob=0.9, peak_time_sec=10.0),
            make_candidate("v1", 1, target=1, clip_prob=0.8, peak_time_sec=12.0),
            make_candidate("v1", 2, target=0, clip_prob=0.7, peak_time_sec=30.0),
        ]
        scores = {candidate.key: candidate.clip_prob for candidate in candidates}
        metrics = evaluate_predictions(
            candidates,
            scores,
            {"shot": 0.5},
            {"v1": {"shot": [10.0]}},
            ["shot"],
            nms_radius_sec=5.0,
            match_tolerance_sec=5.0,
            use_peak_time=True,
        )
        self.assertEqual(metrics["per_class"]["shot"]["tp"], 1)
        self.assertEqual(metrics["per_class"]["shot"]["fp"], 1)
        self.assertEqual(metrics["per_class"]["shot"]["fn"], 0)


if __name__ == "__main__":
    unittest.main()
