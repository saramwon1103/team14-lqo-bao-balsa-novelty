"""Acceptance correctness regressions without executing additional experiments."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'scripts'))


class AcceptanceTests(unittest.TestCase):
    def test_skipped_expert_round_has_no_optimizer_steps(self):
        # Lightning 0.9 leaves optimizers=None until fit; it is not an iterable.
        text = (SOURCE / 'run.py').read_text()
        self.assertIn('for opt in (trainer.optimizers or [])', text)
        optimizers = None
        steps = [float(s['step']) for opt in (optimizers or [])
                 for s in opt.state.values() if 'step' in s]
        self.assertEqual(steps, [])
    def test_random_membership_is_exact_locked_upstream(self):
        from make_splits import validate, constant
        manifest = validate(json.loads((SOURCE / 'manifests/random52.json').read_text()))
        self.assertEqual(manifest['test_queries'], constant((SOURCE / 'experiments.py').read_text(), 'RAND_52_TEST_QUERIES'))
        self.assertEqual((len(manifest['train_queries']), len(manifest['test_queries'])), (94, 19))
        self.assertEqual(manifest['seed'], 52)

    def test_simulator_restore_rejects_cross_split_before_loading(self):
        from balsa.simulator_restore import restore_simulator
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metadata.json'
            path.write_text(json.dumps(dict(split_sha256='wrong')))
            with patch.dict('os.environ', {'BALSA_SPLIT_SHA256': 'expected'}):
                with self.assertRaisesRegex(ValueError, 'split mismatch'):
                    restore_simulator(None, str(path))

    def test_eval_contains_no_training_calls_and_installs_fail_closed_guards(self):
        text = (SOURCE / 'scripts/evaluate_checkpoint.py').read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ('Train', 'fit', 'backward', 'step'))
        self.assertIn('run.BalsaAgent.Train = forbidden_train', text)
        self.assertIn('sim.Sim.Train = forbidden_train', text)

    def test_streaming_hash_matches_bytes(self):
        from balsa.evaluation_artifacts import file_hash
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.bin'
            value = b'x' * (9 * 1024**2)
            path.write_bytes(value)
            self.assertEqual(file_hash(str(path)), hashlib.sha256(value).hexdigest())

    def test_default_featurization_does_not_clear_caller_metadata(self):
        import numpy as np
        from balsa.experience import SimpleReplayBuffer
        from balsa.util import plans_lib, postgres
        a = plans_lib.Node('Scan', table_name='a').with_alias('a')
        b = plans_lib.Node('Scan', table_name='b').with_alias('b')
        root = plans_lib.Node('Join')
        root.children = [a, b]
        root.info = dict(sentinel='must survive', all_filters_est_rows={})
        root.cost = 12
        wi = plans_lib.WorkloadInfo([root])
        buffer = object.__new__(SimpleReplayBuffer)
        buffer.nodes = [root]
        buffer.featurizer = plans_lib.TreeNodeFeaturizer(wi)
        buffer.query_featurizer = lambda node: np.asarray([1, 2], dtype=np.float32)
        buffer.prepare = lambda *args: None
        result = buffer.featurize_with_subplans([root])
        self.assertEqual(root.info['sentinel'], 'must survive')
        self.assertEqual(buffer.nodes, [root])
        self.assertEqual(result[3], [12])
