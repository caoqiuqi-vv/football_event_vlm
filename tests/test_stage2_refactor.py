"""Compatibility checks for the maintained Stage2 package."""
import unittest
import numpy as np


class Stage2CompatibilityTests(unittest.TestCase):
    def test_existing_imports_resolve_to_the_maintained_implementation(self):
        import football_stage2_position_prior as old_model
        from football_events.stage2 import model, sampling, training
        from scripts import run_football_stage2_joint as joint_cli
        from scripts import run_football_stage2_position_prior as current_cli
        self.assertIs(old_model.PositionPriorEventModel, model.PositionPriorEventModel)
        self.assertIs(old_model.select_peaks, sampling.select_peaks)
        self.assertIs(joint_cli.train, training.train)
        self.assertIs(current_cli.train, training.train)

    def test_metrics_match_the_historical_protocol(self):
        from football_events.stage2.metrics import EventCurves, window_ap, paired_video_bootstrap
        from scripts.run_football_stage2_corepatch import window_ap as reference_ap
        from scripts.run_football_stage2_corepatch import paired_video_bootstrap as reference_bootstrap
        labels = ['shot', 'save', 'set_piece']
        rows = [
            {'video_id':video,'start_sec':start,'end_sec':start+10,'label_mask':[1,1,1]}
            for video in ['a','b','c'] for start in [0,5,30]
        ]
        manifest = {
            'videos':{video:{'label_mask':[1,1,1],'duration':3600} for video in ['a','b','c']},
            'gt':{video:{label:[7.] for label in labels} for video in ['a','b','c']},
        }
        curves = EventCurves(rows,manifest)
        left = np.random.default_rng(42).normal(size=(9,3))
        right = np.random.default_rng(43).normal(size=(9,3))
        probability = 1 / (1 + np.exp(-left))
        self.assertEqual(window_ap(curves,probability), reference_ap(curves,probability))
        self.assertEqual(
            paired_video_bootstrap(rows,curves,left,right,repeats=32),
            reference_bootstrap(rows,curves,left,right,repeats=32),
        )


if __name__ == '__main__':
    unittest.main()
