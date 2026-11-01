"""Offline tests against the pinned, online-verified corpus mapping."""
from collections import Counter
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_splits import validate
from verify_job_templates import structure, signed, sha


class LeaveTemplateOutTests(unittest.TestCase):
    def setUp(self):
        self.mapping = json.loads((ROOT / 'manifests/job_templates.json').read_text())
        self.folds = [json.loads(p.read_text()) for p in sorted((ROOT / 'manifests/loo_template').glob('fold_*.json'))]

    def test_full_33_fold_coverage_and_no_group_leakage(self):
        self.assertEqual(len(self.folds), 33)
        expected = {p.name for p in (ROOT / 'queries/join-order-benchmark').glob('*.sql')}
        tests = Counter()
        for fold in self.folds:
            validate(fold)
            self.assertEqual(set(fold['train_queries'] + fold['test_queries']), expected)
            self.assertEqual(set(fold['test_queries']), set(self.mapping['templates'][str(fold['fold_id'])]))
            train_groups = {fold['query_templates'][q] for q in fold['train_queries']}
            test_groups = {fold['query_templates'][q] for q in fold['test_queries']}
            self.assertFalse(train_groups & test_groups)
            tests.update(fold['test_queries'])
        self.assertEqual(tests, Counter({q: 1 for q in expected}))

    def test_pinned_sql_hashes_and_real_graphs(self):
        self.assertTrue(self.mapping['source']['upstream_bytes_verified'])
        representatives = set()
        for template, names in self.mapping['templates'].items():
            graphs = set()
            for q in names:
                raw = (ROOT / 'queries/join-order-benchmark' / q).read_bytes()
                self.assertEqual(sha(raw), self.mapping['queries'][q]['sql_sha256'])
                graph = structure(raw.decode())
                self.assertEqual(graph, self.mapping['queries'][q]['structure'])
                graphs.add(json.dumps(graph, sort_keys=True))
            self.assertEqual(len(graphs), 1)
            representatives.update(graphs)
        self.assertEqual(len(representatives), 33)

    def test_renamed_aliases_not_false_templates(self):
        for prefix in ('8', '15'):
            paths = list((ROOT / 'queries/join-order-benchmark').glob(prefix + '[a-z].sql'))
            self.assertEqual(len({json.dumps(structure(p.read_text()), sort_keys=True) for p in paths}), 1)

    def test_rehashed_cross_template_swap_rejected(self):
        fold = self.folds[0]
        fold.pop('sha256')
        fold['train_queries'][0], fold['test_queries'][0] = fold['test_queries'][0], fold['train_queries'][0]
        with self.assertRaises(ValueError):
            validate(signed(fold))
