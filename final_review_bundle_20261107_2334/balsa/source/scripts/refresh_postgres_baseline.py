#!/usr/bin/env python3
"""Refresh native PostgreSQL baselines for an existing exported Balsa run.

This script does not train or evaluate Balsa.  It re-executes only the raw SQL
queries on fresh PostgreSQL connections so Balsa-only session overrides cannot
leak into the native PostgreSQL baseline.  Existing Balsa timings/plans are
preserved.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _status(exc):
    return 'TIMEOUT' if getattr(exc, 'pgcode', None) == '57014' else 'ERROR'


def _write_csv(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _qid(value):
    value = Path(value).stem
    return value[1:] if value.startswith('q') else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True,
                        help='Exported Balsa run, e.g. balsa/runs/LOO-1/balsa-v2-run-51121')
    parser.add_argument('--query-dir', default=str(ROOT / 'queries' / 'join-order-benchmark'))
    parser.add_argument('--timeout-ms', type=int, default=None,
                        help='Defaults to timeout_ms recorded in balsa_results.csv')
    args = parser.parse_args()

    if args.timeout_ms is not None and args.timeout_ms < 1:
        parser.error('--timeout-ms must be positive')

    import pg_executor
    from balsa.explain_parser import normalize_explain

    run_dir = Path(args.run_dir).resolve()
    query_dir = Path(args.query_dir).resolve()
    evaluation_dir = run_dir / 'evaluation'
    raw_csv = evaluation_dir / 'balsa_results.csv'
    common_pg_csv = run_dir / 'postgres_results.csv'
    manifest_path = evaluation_dir / 'split.json'
    run_manifest_path = run_dir / 'run_manifest.json'
    plans_dir = evaluation_dir / 'plans'

    for required in (raw_csv, common_pg_csv, manifest_path, run_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not query_dir.is_dir():
        raise FileNotFoundError(query_dir)

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    test_queries = {_qid(name): name for name in manifest['test_queries']}

    with raw_csv.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        raw_fields = reader.fieldnames
        raw_rows = list(reader)
    if len(raw_rows) != len(test_queries):
        raise ValueError('Expected one Balsa result row per test query')
    if {_qid(row['query_id']) for row in raw_rows} != set(test_queries):
        raise ValueError('Balsa rows do not match split test membership')

    with common_pg_csv.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        pg_fields = reader.fieldnames
        pg_rows = list(reader)
    pg_by_qid = {_qid(row['query_id']): row for row in pg_rows}
    if set(pg_by_qid) != set(test_queries):
        raise ValueError('postgres_results.csv does not match split test membership')

    # Capture native session defaults before any Balsa-specific SET commands.
    with pg_executor.Cursor() as cur:
        cur.execute('select current_database(), current_user, version(), inet_server_addr(), inet_server_port()')
        connection = list(cur.fetchone())
        cur.execute('show data_directory')
        connection.append(cur.fetchone()[0])
        cur.execute('show geqo')
        native_geqo = cur.fetchone()[0]
        cur.execute('show max_parallel_workers_per_gather')
        native_parallel = cur.fetchone()[0]

    counts = {}
    for raw in raw_rows:
        qid = _qid(raw['query_id'])
        repeat = raw.get('repeat_id') or '0'
        query_file = query_dir / test_queries[qid]
        if not query_file.is_file():
            raise FileNotFoundError(query_file)
        sql = query_file.read_text(encoding='utf-8')
        timeout_ms = args.timeout_ms or int(raw.get('timeout_ms') or 60000)
        baseline = {}
        baseline_error = ''
        try:
            # Fresh connection: no Balsa GEQO/parallel overrides are applied.
            with pg_executor.Cursor() as cur:
                cur.execute('set statement_timeout={}'.format(timeout_ms))
                cur.execute('EXPLAIN (VERBOSE, ANALYZE, FORMAT JSON)\n' + sql)
                baseline = normalize_explain(cur.fetchall())
            status = 'COMPLETED'
            out_path = plans_dir / '{}_repeat_{}_baseline_refresh.json'.format(qid, repeat)
            out_path.write_text(json.dumps(baseline, indent=2) + '\n', encoding='utf-8')
        except Exception as exc:
            status = _status(exc)
            baseline_error = str(exc)

        bp = baseline.get('Planning Time')
        be = baseline.get('Execution Time')
        raw['baseline_status'] = status
        raw['baseline_error'] = baseline_error
        raw['postgres_baseline_planning_ms'] = '' if bp is None else bp
        raw['postgres_baseline_execution_ms'] = '' if be is None else be
        raw['postgres_baseline_end_to_end_ms'] = '' if bp is None or be is None else bp + be

        # Speedup fields are only meaningful when both sides completed.
        raw['speedup_execution'] = ''
        raw['speedup_end_to_end'] = ''
        if status == 'COMPLETED' and raw.get('status') == 'COMPLETED':
            if raw.get('execution_ms') not in ('', None) and float(raw['execution_ms']) > 0:
                raw['speedup_execution'] = be / float(raw['execution_ms'])
            if raw.get('end_to_end_ms') not in ('', None) and float(raw['end_to_end_ms']) > 0:
                raw['speedup_end_to_end'] = (bp + be) / float(raw['end_to_end_ms'])

        pg = pg_by_qid[qid]
        pg['planning_ms'] = '' if bp is None else bp
        pg['execution_ms'] = '' if be is None else be
        pg['end_to_end_ms'] = '' if bp is None or be is None else bp + be
        pg['status'] = 'OK' if status == 'COMPLETED' else status
        pg['timeout_ms'] = str(timeout_ms)
        counts[pg['status']] = counts.get(pg['status'], 0) + 1
        print('{}: {}'.format(qid, pg['status']))

    _write_csv(raw_csv, raw_fields, raw_rows)
    _write_csv(common_pg_csv, pg_fields, pg_rows)

    refresh = dict(
        purpose='native_postgresql_baseline_refresh_only',
        balsa_retrained=False,
        balsa_re_evaluated=False,
        connection='fresh_per_query',
        native_geqo=native_geqo,
        native_max_parallel_workers_per_gather=native_parallel,
        timeout_ms=args.timeout_ms or 'per-row recorded timeout',
        connection_identity=connection[:5],
        data_directory_recorded=bool(connection[5]) if len(connection) > 5 else False,
        status_counts=counts,
    )
    (evaluation_dir / 'postgres_baseline_refresh.json').write_text(
        json.dumps(refresh, indent=2) + '\n', encoding='utf-8')

    run_manifest = json.loads(run_manifest_path.read_text(encoding='utf-8'))
    run_manifest['postgres_baseline_session'] = dict(
        connection='fresh_per_query',
        geqo=native_geqo,
        max_parallel_workers_per_gather=native_parallel,
        statement_timeout_ms=args.timeout_ms or 'per-row recorded timeout',
        source='evaluation/postgres_baseline_refresh.json',
    )
    run_manifest['postgres_status_counts'] = counts
    run_manifest['postgres_baseline_refreshed_without_retraining'] = True
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + '\n', encoding='utf-8')

    print('\nUpdated:')
    print(raw_csv)
    print(common_pg_csv)
    print(run_manifest_path)
    print(evaluation_dir / 'postgres_baseline_refresh.json')
    print('\nNext: rerun analysis/src/merge_balsa_novelty.py and downstream Balsa analyses.')


if __name__ == '__main__':
    main()
