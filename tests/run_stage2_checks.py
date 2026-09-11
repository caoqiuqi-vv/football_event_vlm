#!/usr/bin/env python
"""Run every check in the small Stage2 suite without requiring pytest or a GPU.

Legacy function-style tests and explicit check() entrypoints are adapted to
unittest. This is a bounded Stage2 runner, not a general pytest replacement.
"""
import inspect
import os
from pathlib import Path
import runpy
import sys
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

def main():
    torch.set_num_threads(1)
    suite = unittest.TestSuite()
    names = ['test_stage2_optional', 'test_stage2_corepatch', 'test_stage2_joint',
             'test_stage2_position_prior', 'test_stage2_refactor']
    for name in names:
        namespace = runpy.run_path(str(ROOT/'tests'/(name+'.py')),run_name=name)
        for key, value in sorted(namespace.items()):
            if inspect.isfunction(value) and (key.startswith('test_') or key == 'check'):
                suite.addTest(unittest.FunctionTestCase(value,description=f'{name}.{key}'))
            elif inspect.isclass(value) and issubclass(value,unittest.TestCase):
                suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(value))
    if suite.countTestCases() == 0:
        raise RuntimeError('No Stage2 checks collected')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    raise SystemExit(main())
