from __future__ import print_function

import ast
import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_basequery1_split():
    path = os.path.join(REPO_ROOT, 'experiments.py')
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets
                     if isinstance(target, ast.Name)]
            if 'BASE_QUERY_1_TEST_QUERIES' in names:
                return ast.literal_eval(node.value)
    raise AssertionError('Không tìm thấy BASE_QUERY_1_TEST_QUERIES')


class BaseQuery1ConfigTests(unittest.TestCase):

    def test_split_is_valid(self):
        test_queries = read_basequery1_split()
        self.assertEqual(len(test_queries), 19)
        self.assertEqual(len(set(test_queries)), 19)
        self.assertNotIn('16b.sql', test_queries)
        self.assertNotIn('17a.sql', test_queries)

        query_dir = os.path.join(REPO_ROOT, 'queries',
                                 'join-order-benchmark')
        all_queries = set(name for name in os.listdir(query_dir)
                          if name.endswith('.sql'))
        self.assertEqual(len(all_queries), 113)
        self.assertTrue(set(test_queries).issubset(all_queries))
        self.assertEqual(len(all_queries - set(test_queries)), 94)


if __name__ == '__main__':
    unittest.main()
