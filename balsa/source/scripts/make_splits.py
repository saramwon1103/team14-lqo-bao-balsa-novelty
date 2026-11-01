#!/usr/bin/env python3
"""Lock verified Random, custom BaseQuery and user-approved leave-template-out manifests."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import urllib.request
from verify_job_templates import signed

UPSTREAM = 'https://raw.githubusercontent.com/balsa-project/balsa/a0b45f2371c8c569bb90ad611f39911e5b850852/experiments.py'


def constant(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing constant ' + name)


def validate(manifest):
    body = dict(manifest)
    expected = body.pop('sha256')
    actual = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    if actual != expected:
        raise ValueError('Manifest hash mismatch')
    train, test = body['train_queries'], body['test_queries']
    if not train or not test or len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError('Empty/duplicate membership')
    if set(train) & set(test):
        raise ValueError('Train/test overlap')
    if body.get('protocol') == 'LOO-template':
        groups = body['query_templates']
        held = body['fold_id']
        if any(groups[q] != held for q in test) or any(groups[q] == held for q in train):
            raise ValueError('Template leakage')
        if set(train + test) != set(groups) or len(groups) != 113:
            raise ValueError('Incomplete fold coverage')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=['BaseQuery-1', 'Random-52', 'Smoke', 'LOO-template'])
    parser.add_argument('--template-map', type=Path)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate', type=Path)
    args = parser.parse_args()
    if args.validate:
        validate(json.loads(args.validate.read_text()))
        print('VALID:', args.validate)
        return
    if not args.protocol or not args.output:
        parser.error('--protocol and --output required when creating a manifest')
    if args.protocol == 'LOO-template':
        if not args.template_map:
            parser.error('--template-map required; no grouping inferred here')
        mapping = json.loads(args.template_map.read_text())
        body = dict(mapping)
        expected = body.pop('sha256')
        if signed(body)['sha256'] != expected or not mapping['source']['upstream_bytes_verified']:
            raise ValueError('Unverified template mapping')
        groups = {q: int(k) for k, names in mapping['templates'].items() for q in names}
        queries = sorted(groups)
        for q in queries:
            raw = (args.repo_root / 'queries/join-order-benchmark' / q).read_bytes()
            if hashlib.sha256(raw).hexdigest() != mapping['queries'][q]['sql_sha256']:
                raise ValueError('SQL changed after verification: ' + q)
        args.output.mkdir(parents=True, exist_ok=False)
        for held in range(1, 34):
            manifest = signed(dict(protocol='LOO-template', split_id='LOO-template',
                fold_id=held, seed=None, query_templates=groups,
                source=dict(status='team extension approved by user; not an official Balsa split',
                    template_map_sha256=expected, **mapping['source']),
                train_queries=[q for q in queries if groups[q] != held],
                test_queries=[q for q in queries if groups[q] == held]))
            validate(manifest)
            (args.output / 'fold_{:02d}.json'.format(held)).write_text(json.dumps(manifest, indent=2))
        print('LOCKED: 33 leave-one-template-out folds in', args.output)
        return
    text = (args.repo_root / 'experiments.py').read_text()
    source = {'path': 'experiments.py', 'sha256': hashlib.sha256(text.encode()).hexdigest(),
              'status': 'custom assignment split, not official Balsa'}
    queries = sorted(p.name for p in (args.repo_root / 'queries/join-order-benchmark').glob('*.sql'))
    if len(queries) != 113:
        raise ValueError('Expected 113 JOB SQL files')
    seed = None
    if args.protocol == 'Random-52':
        upstream = urllib.request.urlopen(UPSTREAM, timeout=30).read()
        test = constant(upstream.decode(), 'RAND_52_TEST_QUERIES')
        if test != constant(text, 'RAND_52_TEST_QUERIES'):
            raise ValueError('Local Random differs from pinned upstream')
        source = dict(url=UPSTREAM, sha256=hashlib.sha256(upstream).hexdigest(),
                      status='verified exact upstream membership')
        seed = 52
    elif args.protocol == 'Smoke':
        queries = ['1a.sql', '1b.sql', '3a.sql', '16b.sql', '2a.sql', '2b.sql']
        test = ['2a.sql', '2b.sql']
    else:
        test = constant(text, 'BASE_QUERY_1_TEST_QUERIES')
    if not set(test) <= set(queries):
        raise ValueError('Test SQL missing')
    manifest = dict(protocol=args.protocol, split_id=args.protocol, fold_id=None,
                    seed=seed, source=source,
                    train_queries=[q for q in queries if q not in test], test_queries=test)
    manifest['sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    validate(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(manifest, f, indent=2)
    print('LOCKED:', args.output, manifest['sha256'])


if __name__ == '__main__':
    main()
