"""Normalize PostgreSQL JSON, JSON text and psycopg2 fetchall wrappers."""
import json


def normalize_explain(value):
    for _ in range(8):
        if isinstance(value, str):
            value = json.loads(value)
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        elif isinstance(value, dict) and isinstance(value.get('Plan'), dict):
            return value
        else:
            raise ValueError('Expected one EXPLAIN JSON document with Plan')
    raise ValueError('Too many EXPLAIN wrappers')
