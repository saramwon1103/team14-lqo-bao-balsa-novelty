#!/usr/bin/env python3
"""Strict adapter from the team's split_master.csv to locked Balsa manifests."""
import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path


EXPECTED = {
    'LOO': (80, 33, 'LOO-1'),
    'Random': (91, 22, 'Random-1'),
    'BaseQuery': (94, 19, 'BaseQuery-1'),
}
REQUIRED = ('split_type', 'split_id', 'query_id', 'family', 'role', 'sql_file')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_commit(repo_root, source_path):
    try:
        return subprocess.check_output(
            ['git', '-C', str(repo_root), 'log', '-1', '--format=%H', '--',
             str(source_path)], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_and_validate(source_csv, sql_dir, expected_source_commit=None):
    source_csv = Path(source_csv).resolve()
    sql_dir = Path(sql_dir).resolve()
    raw = source_csv.read_bytes()
    source_hash = sha256(raw)
    with source_csv.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(REQUIRED).issubset(reader.fieldnames):
            raise ValueError('split_master.csv missing required columns: ' + ','.join(REQUIRED))
        rows = list(reader)
    if len(rows) != 339:
        raise ValueError('Expected exactly 339 membership rows for three split-1 protocols; got {}'.format(len(rows)))
    if any(r['split_type'] not in EXPECTED or r['split_id'] != '1' for r in rows):
        raise ValueError('Unexpected split_type or split_id in common source')
    global_keys=[(r['split_type'],r['split_id'],r['query_id'],r['role']) for r in rows]
    if len(global_keys)!=len(set(global_keys)):
        raise ValueError('Duplicate split membership row in common source')
    sql_files = {p.name for p in sql_dir.glob('*.sql') if p.is_file()}
    if len(sql_files) != 113:
        raise ValueError('Canonical JOB SQL directory must contain exactly 113 files')
    result = {}
    for split_type, (expected_train, expected_test, split_name) in EXPECTED.items():
        split_rows = [r for r in rows if r['split_type'] == split_type and r['split_id'] == '1']
        if len(split_rows) != 113:
            raise ValueError('{} expected 113 source rows; got {}'.format(split_name, len(split_rows)))
        membership_keys = [(r['query_id'], r['role']) for r in split_rows]
        if len(set(membership_keys)) != 113:
            raise ValueError('{} has duplicate query/role rows'.format(split_name))
        if len({r['query_id'] for r in split_rows}) != 113 or len({r['sql_file'] for r in split_rows}) != 113:
            raise ValueError('{} has duplicate query or SQL file'.format(split_name))
        train = [r for r in split_rows if r['role'] == 'train']
        test = [r for r in split_rows if r['role'] == 'test']
        if len(train) != expected_train or len(test) != expected_test:
            raise ValueError('{} count mismatch: train={} test={}'.format(split_name, len(train), len(test)))
        if {r['role'] for r in split_rows} != {'train', 'test'}:
            raise ValueError('{} has unsupported role'.format(split_name))
        if {r['sql_file'] for r in split_rows} != sql_files:
            raise ValueError('{} SQL membership differs from canonical 113 JOB files'.format(split_name))
        seen = set()
        for row in split_rows:
            if row['split_id'] != '1':
                raise ValueError('{} has unexpected split_id'.format(split_name))
            sql_file = row['sql_file']
            if Path(sql_file).name != sql_file or not re.fullmatch(r'\d+[a-z]+\.sql', sql_file):
                raise ValueError('Invalid JOB sql_file: ' + sql_file)
            query_id = 'q' + Path(sql_file).stem
            match = re.fullmatch(r'q(\d+)[a-z]+', query_id)
            if row['query_id'] != query_id or row['family'] != 'q' + match.group(1):
                raise ValueError('Inconsistent query_id/family/sql_file: ' + repr(row))
            if not (sql_dir / sql_file).is_file():
                raise ValueError('Missing canonical SQL: ' + sql_file)
            seen.add(row['query_id'])
        if {r['query_id'] for r in train} & {r['query_id'] for r in test}:
            raise ValueError('{} train/test overlap'.format(split_name))
        if {r['query_id'] for r in train + test} != {'q' + Path(name).stem for name in sql_files}:
            raise ValueError('{} membership union is not canonical JOB'.format(split_name))
        ordered = sorted(split_rows, key=lambda r: (r['query_id']))
        membership_bytes = '\n'.join('{},{},{},{},{},{}'.format(
            r['split_type'], r['split_id'], r['query_id'], r['family'], r['role'], r['sql_file'])
            for r in ordered).encode()
        manifest = dict(protocol=split_name, split_id='1', fold_id=None, seed=None,
            source=dict(path='analysis/data/split_master.csv', sha256=source_hash,
                source_commit=expected_source_commit, membership_sha256=sha256(membership_bytes),
                status='team common split source; authoritative'),
            train_queries=[r['sql_file'] for r in sorted(train, key=lambda r:r['query_id'])],
            test_queries=[r['sql_file'] for r in sorted(test, key=lambda r:r['query_id'])])
        manifest['sha256'] = sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode())
        result[split_name] = dict(manifest=manifest, snapshot=ordered)
    return dict(source_csv=source_csv, source_sha256=source_hash,
                source_commit=expected_source_commit, splits=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-csv', type=Path, required=True)
    parser.add_argument('--sql-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    commit = args.source_commit or source_commit(args.source_csv.resolve().parents[2], args.source_csv)
    if not commit:
        parser.error('Cannot determine split_master source commit; pass --source-commit explicitly')
    locked = load_and_validate(args.source_csv, args.sql_dir, commit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, values in locked['splits'].items():
        (args.output_dir / (name.lower().replace('-', '') + '.json')).write_text(
            json.dumps(values['manifest'], indent=2) + '\n')
    print(json.dumps(dict(source_sha256=locked['source_sha256'], source_commit=commit,
        counts={name: dict(train=len(v['manifest']['train_queries']), test=len(v['manifest']['test_queries']))
                for name,v in locked['splits'].items()}), indent=2))


if __name__ == '__main__':
    main()
