"""Dependency-free launcher wiring tests: no torch import, GPUs or training."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='motion-launcher-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root / 'scripts'
        scripts.mkdir()
        source = Path(__file__).resolve().parents[1] / 'scripts'
        for name in ('run_object_motion_adapter_v3.sh',
                     'run_object_motion_adapter_v7_offline_tracks.sh',
                     'run_object_motion_adapter_v8.sh'):
            shutil.copyfile(source / name, scripts / name)
        (self.root / 'checkpoint.pt').write_text('fixture, never loaded')
        (self.root / 'config.yaml').write_text('{}')
        (self.root / 'manifest.json').write_text('{}')
        capture = self.root / 'torchrun-capture'
        capture.write_text('#!/bin/sh\nprintf "%s\\n" "TRAIN_REACHED" "$PWD" "$@"\n')
        capture.chmod(0o755)
        self.env = dict(
            PATH=os.defpath, PYTHON_BIN='/usr/bin/true',
            TORCHRUN_BIN=str(capture), SOURCE_CHECKPOINT=str(self.root / 'checkpoint.pt'),
            BASE_CONFIG=str(self.root / 'config.yaml'),
            MOTION_OFFLINE_BALL_INDEX_ROOT=str(self.root),
        )

    def launch(self, mode, **overrides):
        return subprocess.run(
            ['bash', str(self.root / 'scripts/run_object_motion_adapter_v8.sh')],
            env={**self.env, 'MOTION_V8_MODE': mode, **overrides},
            capture_output=True, text=True, timeout=10,
        )

    def test_four_modes_reach_same_checkout_with_expected_overrides(self):
        for mode, values in {
            'control': ('false', 'false', 'false', 'false'),
            'grad': ('true', 'false', 'false', 'false'),
            'context': ('true', 'true', 'true', 'false'),
            'uniform': ('true', 'true', 'true', 'true'),
        }.items():
            with self.subTest(mode=mode):
                result = self.launch(mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                lines = result.stdout.splitlines()
                self.assertEqual(lines[:2], ['TRAIN_REACHED', str(self.root)])
                names = ('event_relation_grad_enabled', 'event_context_enabled',
                         'event_frame_fusion_enabled', 'event_context_uniform')
                for name, value in zip(names, values):
                    self.assertIn(f'model.object_motion.{name}={value}', lines)
                self.assertIn('model.object_motion.image_size=[720,1280]', lines)
                self.assertIn('model.object_motion.shared_anchor_ball_lora=true', lines)
                self.assertIn('model.object_motion.event_context_feature_grad=false', lines)
                self.assertIn('model.object_motion.event_ranking_target=residual', lines)
                self.assertIn(f'model.init_checkpoint={self.root}/checkpoint.pt', lines)

    def test_unknown_mode_stops_before_training(self):
        result = self.launch('typo')
        self.assertEqual(result.returncode, 64)
        self.assertNotIn('TRAIN_REACHED', result.stdout)

    def test_failed_tensor_preflight_stops_before_training(self):
        result = self.launch('grad', PYTHON_BIN='/usr/bin/false')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('TRAIN_REACHED', result.stdout)

    def test_followup_options_are_forwarded(self):
        result = self.launch('context', MOTION_EVENT_CONTEXT_FEATURE_GRAD='true',
                             MOTION_EVENT_RANKING_TARGET='final')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('model.object_motion.event_context_feature_grad=true', result.stdout)
        self.assertIn('model.object_motion.event_ranking_target=final', result.stdout)


if __name__ == '__main__':
    unittest.main()
