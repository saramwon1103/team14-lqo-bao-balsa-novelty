import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

PATH = Path(__file__).resolve().parents[1] / 'scripts/run_experiment.py'
spec = importlib.util.spec_from_file_location('run_experiment_test_module', str(PATH))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ExperimentOrchestrationTests(unittest.TestCase):
    def test_train_returns_before_fresh_evaluation_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'new-run'
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    train = output / 'training'
                    train.mkdir()
                    (train / 'training_summary.json').write_text(json.dumps(dict(checkpoint='trusted.pt')))
                else:
                    self.assertTrue((output / 'training/training_summary.json').is_file())
                    self.assertIn('evaluate_checkpoint.py', command[1])
                return SimpleNamespace(returncode=0)
            argv = ['run_experiment.py', '--manifest', 'fixture.json', '--output-dir', str(output), '--simulator-smoke']
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run', side_effect=run):
                self.assertEqual(runner.main(), 0)
            self.assertEqual(len(calls), 2)
            self.assertIn('--simulator-smoke', calls[0])
            self.assertEqual(json.loads((output / 'run_summary.json').read_text())['status'], 'COMPLETED')

    def test_train_failure_does_not_launch_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'new-run'
            argv = ['run_experiment.py', '--manifest', 'fixture.json', '--output-dir', str(output), '--allow-full']
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=7)) as process:
                self.assertEqual(runner.main(), 7)
            self.assertEqual(process.call_count, 1)
            self.assertEqual(json.loads((output / 'run_summary.json').read_text())['status'], 'FAILED')
