"""Explicit fixtures, not benchmark measurements."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'balsa'))
from explain_parser import normalize_explain


class ExplainParserTests(unittest.TestCase):
    def test_real_shapes(self):
        document = {'Plan': {'Total Cost': 12, 'Plan Rows': 3},
                    'Planning Time': 0.2, 'Execution Time': 1.3}
        for value in (document, [document], [([document],)],
                      [[[document]]], json.dumps([document])):
            self.assertEqual(normalize_explain(value), document)

    def test_reject_invalid_and_multiple_documents(self):
        for value in (None, [], {}, [{'Plan': {}}, {'Plan': {}}], 'not json'):
            with self.assertRaises(ValueError):
                normalize_explain(value)
