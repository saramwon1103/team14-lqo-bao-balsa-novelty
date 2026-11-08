#!/usr/bin/env python3
"""Verify the pinned JOB corpus and its relation/column join graphs (not filenames alone)."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request

COMMIT = 'a0b45f2371c8c569bb90ad611f39911e5b850852'
URL = 'https://raw.githubusercontent.com/balsa-project/balsa/' + COMMIT + '/queries/join-order-benchmark/'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def signed(body):
    body = dict(body)
    body['sha256'] = sha(json.dumps(body, sort_keys=True, separators=(',', ':')).encode())
    return body


def structure(sql):
    # Scope: JOB's single SELECT/FROM/WHERE block, comma joins and equijoins.
    # Strip string constants before extracting identifiers; do not mistake
    # strings containing "a.id=b.id" for joins.
    text = re.sub(r"'(?:''|[^'])*'", "''", sql).lower()
    text = re.sub(r'--[^\n]*|/\*.*?\*/', ' ', text, flags=re.S)
    parts = re.split(r'\bfrom\b|\bwhere\b', text)
    if len(parts) != 3:
        raise ValueError('Unsupported SQL shape')
    tables = re.findall(r'\b(\w+)\s+as\s+(\w+)\b', parts[1])
    if len(tables) != len(parts[1].split(',')) or not tables:
        raise ValueError('Unparsed FROM relation')
    by_table = defaultdict(list)
    for table, alias in tables:
        by_table[table].append(alias)
    aliases = {alias: '{}#{}'.format(table, rank)
               for table, values in by_table.items()
               for rank, alias in enumerate(sorted(values))}
    edges = []
    for a, ac, b, bc in re.findall(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', parts[2]):
        edges.append(sorted([aliases[a] + '.' + ac, aliases[b] + '.' + bc]))
    if not edges:
        raise ValueError('No equijoin graph')
    return dict(relations=sorted(aliases.values()), join_edges=sorted(edges))


def verify(query_dir, online=True):
    paths = sorted(query_dir.glob('*.sql'))
    if len(paths) != 113:
        raise ValueError('Expected 113 SQL')
    def inspect(p):
        raw = p.read_bytes()
        if online:
            upstream = urllib.request.urlopen(URL + p.name, timeout=60).read()
            if upstream != raw:
                raise ValueError('SQL differs from pinned upstream: ' + p.name)
        match = re.fullmatch(r'(\d+)[a-z]\.sql', p.name)
        if not match:
            raise ValueError('Unexpected JOB identifier')
        return p.name, dict(template_id=int(match.group(1)), sql_sha256=sha(raw),
                            structure=structure(raw.decode()))
    with ThreadPoolExecutor(max_workers=6) as pool:
        queries = dict(pool.map(inspect, paths))
    groups = defaultdict(list)
    for name, record in queries.items():
        groups[record['template_id']].append(name)
    if set(groups) != set(range(1, 34)):
        raise ValueError('Expected templates 1..33')
    for group, names in groups.items():
        if len({json.dumps(queries[q]['structure'], sort_keys=True) for q in names}) != 1:
            raise ValueError('Join structure differs within template ' + str(group))
    if len({json.dumps(queries[names[0]]['structure'], sort_keys=True)
            for names in groups.values()}) != 33:
        raise ValueError('Expected 33 distinct relation/column join graphs')
    return signed(dict(protocol='LOO-template', status='team extension, not official Balsa split',
        source=dict(sql_url_prefix=URL, upstream_commit=COMMIT,
            upstream_bytes_verified=online,
            paper_url='https://www.vldb.org/pvldb/vol9/p204-leis.pdf',
            paper_section='2.2, printed page 205: 33 query structures, 2-6 variants, 113 queries',
            verification='pinned SQL bytes + FROM relation roles + column equijoin graphs; alias normalization including templates 8 and 15'),
        templates={str(k): sorted(v) for k, v in sorted(groups.items())}, queries=queries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'queries/join-order-benchmark')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.query_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print('Verified 113 pinned SQL / 33 template structures:', result['sha256'])


if __name__ == '__main__':
    main()
