from __future__ import print_function

import csv
import os
import sys
import tempfile
import unittest

# Import the standalone logger without importing balsa/__init__.py.  The latter
# intentionally loads the complete legacy ML stack, which is unnecessary for
# these standard-library tests.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'balsa'))

from structured_logging import StructuredExperimentLogger  # noqa: E402
from structured_logging import extract_postgres_metrics  # noqa: E402


def fake_result(cost=1842.3, rows=12500, planning=18, execution=870):
    return [[[{
        'Plan': {
            'Total Cost': cost,
            'Plan Rows': rows,
        },
        'Planning Time': planning,
        'Execution Time': execution,
    }]]]


class StructuredLoggingTests(unittest.TestCase):

    def test_extract_postgres_metrics(self):
        metrics = extract_postgres_metrics(fake_result())
        self.assertEqual(metrics['cost'], 1842.3)
        self.assertEqual(metrics['cardinality'], 12500)
        self.assertEqual(metrics['planning_ms'], 18)
        self.assertEqual(metrics['execution_ms'], 870)

    def test_exact_sections_and_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = StructuredExperimentLogger(directory, 'BaseQuery-1')
            logger.start_iteration(0)
            logger.record_training_plan('16b', fake_result(), 840)
            logger.record_encoding('16b', 120)
            logger.record_training(2500, '/tmp/checkpoint.pt')
            logger.record_evaluation('2a', 120, 930, 18, 870)
            source_checkpoint = os.path.join(directory, 'source.pt')
            with open(source_checkpoint, 'wb') as handle:
                handle.write(b'checkpoint')
            checkpoint = logger.mirror_checkpoint(source_checkpoint, 0)
            path = logger.finish_iteration()
            with open(path) as handle:
                text = handle.read()
            self.assertIn('===== EXPERIMENT START =====', text)
            self.assertIn('Split: BaseQuery-1', text)
            self.assertIn('Model: Balsa', text)
            self.assertIn('[1] TRAINING DATA GENERATION', text)
            self.assertIn('Train query: q16b', text)
            self.assertIn('Cardinality: 12500', text)
            self.assertIn('[2] QUERY & PLAN ENCODING', text)
            self.assertIn('[3] LQO TRAINING', text)
            self.assertIn('[4] LQO EVALUATION', text)
            self.assertIn('Test query: q2a', text)
            self.assertIn('[POSTGRESQL]', text)
            self.assertIn('===== EXPERIMENT END =====', text)
            self.assertTrue(os.path.isfile(checkpoint))

            with open(os.path.join(directory, 'balsa_results.csv')) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['model'], 'Balsa')


if __name__ == '__main__':
    unittest.main()
