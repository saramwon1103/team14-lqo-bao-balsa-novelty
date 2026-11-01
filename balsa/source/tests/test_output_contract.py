import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'balsa'))
sys.path.insert(0, str(ROOT / 'scripts'))
from structured_logging import StructuredExperimentLogger
from make_splits import validate


class OutputContractTests(unittest.TestCase):
    def test_complete_timing_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = StructuredExperimentLogger(directory, 'fixture')
            logger.start_iteration(0)
            logger.record_evaluation('2a', 10, 100, 2, 200,
                                     postgres_hinted_planning_ms=3,
                                     result_source='MEASURED')
            row = logger.current['evaluation_rows'][0]
            self.assertEqual(row['end_to_end_ms'], 113)
            self.assertEqual(row['speedup_execution'], 2)
            self.assertAlmostEqual(row['speedup_end_to_end'], 202 / 113)

    def test_timeout_missing_measurements_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = StructuredExperimentLogger(directory, 'fixture')
            logger.start_iteration(0)
            logger.record_evaluation('2a', 10, None, 2, 200, status='TIMEOUT')
            row = logger.current['evaluation_rows'][0]
            self.assertIsNone(row['end_to_end_ms'])
            self.assertIsNone(row['speedup_execution'])
            self.assertIsNone(row['speedup_end_to_end'])

    def test_locked_manifests_and_tamper(self):
        for name in ('basequery1', 'random52', 'smoke'):
            manifest = json.loads((ROOT / 'manifests' / (name + '.json')).read_text())
            validate(manifest)
            manifest['test_queries'].append('16b.sql')
            with self.assertRaises(ValueError):
                validate(manifest)

    def test_no_run_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = StructuredExperimentLogger(directory, 'fixture')
            logger.start_iteration(0)
            logger.finish_iteration()
            with self.assertRaises(FileExistsError):
                StructuredExperimentLogger(directory, 'fixture')
