"""Resource-controller fixtures only: no experiment, PostgreSQL or training."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot_safety_fixture', str(ROOT / 'scripts/pilot_supervisor.py'))
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class PilotSafetyTests(unittest.TestCase):
    def test_memory_limit_is_forwarded_to_worker_service(self):
        with patch.object(supervisor, 'command', return_value='fixture') as run:
            supervisor.start_service('fixture.service', ['/bin/true'], {},
                                     'fixture.slice', memory_gib=18)
        args = run.call_args[0][0]
        self.assertIn('--property=MemoryMax=18G', args)
        self.assertIn('--property=KillMode=control-group', args)
        self.assertIn('--property=MemorySwapMax=0', args)

    def test_atomic_summary_preserves_previous_on_rename_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'summary.json'
            supervisor.atomic_json(target, dict(status='OLD_COMPLETE'))
            with patch.object(supervisor.os, 'replace', side_effect=OSError('fixture interrupted')):
                with self.assertRaises(OSError):
                    supervisor.atomic_json(target, dict(status='NEW'))
            self.assertEqual(json.loads(target.read_text())['status'], 'OLD_COMPLETE')

    def test_export_keeps_completed_bundle_not_partial_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'image'
            source.mkdir()
            (source / 'evaluation.pt').write_bytes(b'fixture-complete')
            (source / 'evaluation.pt.tmp').write_bytes(b'fixture-partial')
            output = Path(directory) / 'export'
            result = supervisor.safe_export(source, output)
            self.assertEqual((output / 'evaluation.pt').read_bytes(), b'fixture-complete')
            self.assertFalse((output / 'evaluation.pt.tmp').exists())
            self.assertEqual(result['exported_bytes'], 16)

    def test_chunked_tree_featurization_bit_identical_and_no_node_leak(self):
        import numpy as np
        import torch
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from balsa.models import treeconv
        from balsa.util import plans_lib

        class _DummyWI:
            all_ops = np.asarray(['Join', 'Scan'])
            rel_ids = np.asarray(['a AS a', 'b AS b', 'c AS c', 'd AS d'])
            def HasPhysicalOps(self):
                return False

        featurizer = plans_lib.TreeNodeFeaturizer(_DummyWI())
        leaf_a = plans_lib.Node('Scan', table_name='a').with_alias('a')
        leaf_b = plans_lib.Node('Scan', table_name='b').with_alias('b')
        leaf_c = plans_lib.Node('Scan', table_name='c').with_alias('c')
        leaf_d = plans_lib.Node('Scan', table_name='d').with_alias('d')
        def _join(left, right):
            n = plans_lib.Node('Join')
            n.children = [left, right]
            return n

        join_ab = _join(leaf_a, leaf_b)
        join_abc = _join(join_ab, leaf_c)
        join_abcd = _join(join_abc, leaf_d)
        trees = [join_abcd, join_abc, join_ab]

        # Original reference formula before chunked preallocation.
        ref_indexes = torch.from_numpy(
            treeconv._batch([treeconv._make_indexes(x) for x in trees])).long()
        ref_trees = torch.from_numpy(
            treeconv._batch([treeconv._featurize_tree(x, featurizer) for x in trees])
        ).transpose(1, 2)
        for t in trees:
            treeconv._clear_node_feature_vec(t)

        got_trees, got_indexes = treeconv.make_and_featurize_trees(
            trees, featurizer, consume_trees=False)
        self.assertTrue(torch.equal(ref_trees, got_trees))
        self.assertTrue(torch.equal(ref_indexes, got_indexes))
        leaked = []
        plans_lib.MapNode(join_abcd, lambda n: leaked.append(hasattr(n, '__node_feature_vec')))
        self.assertFalse(any(leaked))

        consumable = list(trees)
        got_trees2, got_indexes2 = treeconv.make_and_featurize_trees(
            consumable, featurizer, consume_trees=True)
        self.assertTrue(torch.equal(ref_trees, got_trees2))
        self.assertTrue(torch.equal(ref_indexes, got_indexes2))
        self.assertEqual(consumable, [None, None, None])

        # A chunk boundary must not change sample order, shared-subtree features or padding.
        many = [join_abcd, join_abc, join_ab] * 1366
        reference_indexes = torch.from_numpy(treeconv._batch([treeconv._make_indexes(t) for t in many])).long()
        reference_trees = torch.from_numpy(treeconv._batch([treeconv._featurize_tree(t, featurizer) for t in many])).transpose(1, 2)
        for t in trees:
            treeconv._clear_node_feature_vec(t)
        actual_trees, actual_indexes = treeconv.make_and_featurize_trees(many, featurizer)
        self.assertTrue(torch.equal(reference_trees, actual_trees))
        self.assertTrue(torch.equal(reference_indexes, actual_indexes))
        self.assertEqual(len(many), 4098)
        self.assertTrue(all(t is not None for t in many))
