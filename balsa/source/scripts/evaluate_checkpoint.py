#!/usr/bin/env python3
"""Evaluate a trusted evaluation.pt bundle in a fresh process, without training."""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIELDS = '''run_id protocol split_id fold_id seed query_id repeat_id checkpoint_sha256 split_sha256 chosen_plan_id balsa_planning_ms postgres_hinted_planning_ms execution_ms end_to_end_ms wall_clock_ms postgres_baseline_planning_ms postgres_baseline_execution_ms postgres_baseline_end_to_end_ms speedup_execution speedup_end_to_end status timeout_ms result_source hint_respected hinted_sql_path explain_json_path error_message baseline_status baseline_error'''.split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trusted evaluation.pt bundle')
    parser.add_argument('--output-dir', required=True, help='New directory, must not exist')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--timeout-ms', type=int, default=60000)
    parser.add_argument('--run-id', help='Stable common run ID; defaults to a fresh UUID')
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout_ms < 1:
        parser.error('repeats and timeout must be positive')
    import random
    import numpy as np
    import torch
    import pg_executor
    from balsa.optimizer import Optimizer, PlannerConfig
    from balsa.util import plans_lib, postgres
    from balsa.explain_parser import normalize_explain
    from balsa.evaluation_artifacts import file_hash, weight_hash
    import run
    import sim
    def forbidden_train(*args, **kwargs):
        raise RuntimeError('Training is forbidden in independent evaluation')
    run.BalsaAgent.Train = forbidden_train
    sim.Sim.Train = forbidden_train
    if torch.cuda.is_available():
        parser.error('Legacy acceptance evaluator is CPU-only; set CUDA_VISIBLE_DEVICES=""')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_hash = file_hash(args.checkpoint)
    bundle = torch.load(args.checkpoint, map_location='cpu')
    if bundle.get('version') != 1:
        raise ValueError('Requires complete evaluation bundle; raw weights are insufficient')
    manifest = dict(bundle['manifest'])
    split_hash = manifest.pop('sha256')
    if hashlib.sha256(json.dumps(manifest, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest() != split_hash:
        raise ValueError('Split hash mismatch')
    nodes = plans_lib.FilterScansOrJoins(bundle['test_nodes'])
    if any(not re.fullmatch(r'[0-9]+[a-z]+', n.info['query_name']) for n in nodes):
        raise ValueError('Unsafe JOB query identifier')
    test_names = [q[:-4] if q.endswith('.sql') else q for q in manifest['test_queries']]
    train_names = [q[:-4] if q.endswith('.sql') else q for q in manifest['train_queries']]
    if len(nodes) != len(test_names) or {n.info['query_name'] for n in nodes} != set(test_names):
        raise ValueError('Test membership mismatch')
    by_name = {n.info['query_name']: n for n in nodes}
    nodes = [by_name[name] for name in test_names]
    if bundle.get('training_seed') != args.seed:
        raise ValueError('Evaluator seed differs from recorded agent seed')
    if set(test_names) & set(train_names):
        raise ValueError('Train/test overlap')
    os.makedirs(args.output_dir, exist_ok=False)
    plans_dir = os.path.join(args.output_dir, 'plans')
    os.mkdir(plans_dir)
    with open(os.path.join(args.output_dir, 'split.json'), 'w') as f:
        json.dump(bundle['manifest'], f, indent=2)
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        f.write(bundle['params'].ToText())
    model = bundle['model'].eval()
    before = weight_hash(model)
    if before != bundle['weight_sha256']:
        raise ValueError('Weight hash mismatch')
    p = bundle['params']
    planner = Optimizer(bundle['workload_info'], bundle['plan_featurizer'],
                        bundle['pos_featurizer'], bundle['query_featurizer'],
                        bundle['transform'].InvertCost, model, p.tree_conv, p.beam,
                        search_until_n_complete_plans=p.search_until_n_complete_plans,
                        plan_physical=p.plan_physical,
                        use_plan_restrictions=p.real_use_plan_restrictions)
    run_id = args.run_id or uuid.uuid4().hex
    logpath = os.path.join(args.output_dir, 'experiment.log')
    with open(logpath, 'w') as log:
        log.write('===== EXPERIMENT START =====\nModel: Balsa\nRun: {}\nSplit: {}\nFold: {}\nSeed: {}\nTraining status: EVAL_ONLY\n'.format(
            run_id, manifest['split_id'], manifest['fold_id'], args.seed))
    errors = 0
    predictions = []
    with pg_executor.Cursor() as cur:
        cur.execute('select current_database(), current_user, version(), inet_server_addr(), inet_server_port()')
        connection = list(cur.fetchone())
        cur.execute('show data_directory')
        connection.append(cur.fetchone()[0])
    with open(os.path.join(args.output_dir, 'environment.json'), 'w') as f:
        json.dump(dict(connection=connection, cache_policy='uncontrolled warm; no cache drop',
                       seed=args.seed, process_id=os.getpid(),
                       boundary='Balsa encoding/search/selection + PG Planning Time + Execution Time; wall clock separately; baseline same session settings'), f, indent=2)
    with open(os.path.join(args.output_dir, 'balsa_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for node in nodes:
            for repeat in range(args.repeats):
                row = dict(run_id=run_id, protocol=manifest['protocol'],
                           split_id=manifest['split_id'], fold_id=manifest['fold_id'],
                           seed=args.seed, query_id=node.info['query_name'], repeat_id=repeat,
                           checkpoint_sha256=checkpoint_hash, split_sha256=split_hash,
                           result_source='MEASURED', timeout_ms=args.timeout_ms)
                start = time.perf_counter()
                try:
                    with torch.no_grad():
                        _, plan, prediction = planner.plan(node, p.search_method,
                            bushy=p.bushy, planner_config=PlannerConfig.Get(p.planner_config)
                            if p.planner_config else None,
                            avoid_eq_filters=p.avoid_eq_filters, epsilon_greedy=0)
                    row['balsa_planning_ms'] = (time.perf_counter() - start) * 1000
                    hint = plan.hint_str(with_physical_hints=p.plan_physical)
                    row['chosen_plan_id'] = 'plan_' + hashlib.sha256(hint.encode('utf-8')).hexdigest()[:16]
                    predictions.append(dict(query_id=row['query_id'], repeat_id=repeat,
                                            predicted_latency_ms=float(prediction), hint=hint))
                    sql = node.info['sql_str']
                    stem = '{}_repeat_{}'.format(row['query_id'], repeat)
                    sqlpath = os.path.join(plans_dir, stem + '_hinted.sql')
                    jsonpath = os.path.join(plans_dir, stem + '_explain.json')
                    with open(sqlpath, 'w') as out:
                        out.write(hint + '\n' + sql)
                    row['hinted_sql_path'] = sqlpath
                    with pg_executor.Cursor() as cur:
                        cur.execute('set geqo=off')
                        cur.execute('set max_parallel_workers_per_gather=0')
                        cur.execute('set statement_timeout={}'.format(args.timeout_ms))
                        cur.execute(hint + '\n EXPLAIN (VERBOSE, ANALYZE, FORMAT JSON)\n' + sql)
                        document = normalize_explain(cur.fetchall())
                        row['wall_clock_ms'] = (time.perf_counter() - start) * 1000
                        with open(jsonpath, 'w') as out:
                            json.dump(document, out, indent=2)
                        row['explain_json_path'] = jsonpath
                        actual = plans_lib.FilterScansOrJoins(postgres.ParsePostgresPlanJson(document))
                        respected = actual.hint_str(with_physical_hints=p.plan_physical) == hint
                        row.update(hint_respected=respected,
                                   status='COMPLETED' if respected else 'HINT_MISMATCH',
                                   postgres_hinted_planning_ms=document.get('Planning Time'),
                                   execution_ms=document.get('Execution Time'))
                        baseline = {}
                        try:
                            cur.execute('EXPLAIN (VERBOSE, ANALYZE, FORMAT JSON)\n' + sql)
                            baseline = normalize_explain(cur.fetchall())
                            row['baseline_status'] = 'COMPLETED'
                            with open(os.path.join(plans_dir, stem + '_baseline.json'), 'w') as out:
                                json.dump(baseline, out, indent=2)
                        except Exception as exc:
                            row['baseline_status'] = ('TIMEOUT' if getattr(exc, 'pgcode', None) == '57014' else 'ERROR')
                            row['baseline_error'] = str(exc)
                    bp, be = baseline.get('Planning Time'), baseline.get('Execution Time')
                    row.update(postgres_baseline_planning_ms=bp,
                               postgres_baseline_execution_ms=be)
                    if all(row.get(k) is not None for k in
                           ('balsa_planning_ms', 'postgres_hinted_planning_ms', 'execution_ms')):
                        row['end_to_end_ms'] = sum(row[k] for k in
                            ('balsa_planning_ms', 'postgres_hinted_planning_ms', 'execution_ms'))
                    if bp is not None and be is not None:
                        row['postgres_baseline_end_to_end_ms'] = bp + be
                        if respected and row.get('end_to_end_ms') and row.get('execution_ms'):
                            row['speedup_execution'] = be / row['execution_ms']
                            row['speedup_end_to_end'] = (bp + be) / row['end_to_end_ms']
                    if not respected:
                        errors += 1
                except Exception as exc:
                    row.update(status='TIMEOUT' if getattr(exc, 'pgcode', None) == '57014'
                               else 'ERROR', error_message=str(exc))
                    errors += 1
                    print('EVAL ERROR:', row['query_id'], exc, file=sys.stderr)
                writer.writerow(row)
                f.flush()
                with open(logpath, 'a') as log:
                    log.write('\n[4] LQO EVALUATION\nTest query: {}\nRepeat: {}\n'.format(row['query_id'], repeat))
                    for label, key in [('Balsa inference/planning', 'balsa_planning_ms'),
                        ('PostgreSQL hinted planning', 'postgres_hinted_planning_ms'),
                        ('Execution time', 'execution_ms'), ('End-to-End', 'end_to_end_ms'),
                        ('Wall clock', 'wall_clock_ms'), ('Status', 'status'),
                        ('Result source', 'result_source'), ('Plan', 'explain_json_path')]:
                        log.write('{}: {}\n'.format(label, row.get(key, 'N/A')))
                    log.write('[POSTGRESQL]\nPlanning: {}\nExecution: {}\nEnd-to-End: {}\n'.format(
                        row.get('postgres_baseline_planning_ms', 'N/A'),
                        row.get('postgres_baseline_execution_ms', 'N/A'),
                        row.get('postgres_baseline_end_to_end_ms', 'N/A')))
                print('EVAL RESULT:', row)
    after = weight_hash(model)
    unchanged = before == after and checkpoint_hash == file_hash(args.checkpoint)
    with open(os.path.join(args.output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(dict(weights_before=before, weights_after=after,
                       checkpoint_sha256=checkpoint_hash, unchanged=unchanged,
                       errors=errors, process_id=os.getpid(), train_called=False), f, indent=2)
    with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
        json.dump(predictions, f, indent=2)
    if not unchanged:
        raise RuntimeError('Weights changed during evaluation')
    with open(logpath, 'a') as log:
        log.write('Weight hash unchanged: YES\n===== EXPERIMENT END =====\n')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
