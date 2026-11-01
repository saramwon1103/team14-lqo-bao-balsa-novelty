#!/usr/bin/env python3
"""Run one locked split with isolated state. Full runs require explicit approval flag."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from make_splits import validate
from verify_job_templates import signed, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--allow-full', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--expert-data', type=Path,
                        help='Optional trusted workload pickle; validate exact query membership and record hash')
    parser.add_argument('--simulator-smoke', action='store_true', help='One real full-membership fold round with simulator, reduced search/epochs and timeouts')
    parser.add_argument('--iterations', type=int, help='Override full/pilot iteration count; requires --allow-full')
    parser.add_argument('--initial-timeout-ms', type=int, help='Explicit first-round execution guard for full/pilot')
    parser.add_argument('--isolated-runtime', action='store_true', help='Keep working-directory artifacts inside this new run')
    parser.add_argument('--simulator-reuse', type=Path, help='Verified same-split simulator/cache specification; not exact resume')
    parser.add_argument('--simulator-bundle', type=Path, help='Complete, hash-bound simulator initialization metadata')
    parser.add_argument('--simulator-checkpoint', type=Path, help='Verified split-compatible upstream simulator weights; reconstruct encoders from this split only')
    parser.add_argument('--simulator-integration-epochs', type=int, help='Explicit integration-only simulator epoch limit; agent configuration unchanged')
    parser.add_argument('--independent-eval-only', action='store_true', help='Schedule test reporting in fresh evaluator only; train feedback unchanged')
    args = parser.parse_args()
    if args.expert_data:
        args.expert_data = args.expert_data.resolve()
    if not args.smoke and not args.simulator_smoke and not args.allow_full:
        parser.error('Full training requires --allow-full')
    if args.iterations is not None and (not args.allow_full or args.iterations < 1):
        parser.error('--iterations requires --allow-full and a positive count')
    if args.initial_timeout_ms is not None and args.initial_timeout_ms < 1:
        parser.error('Timeout must be positive')
    manifest = validate(json.loads(args.manifest.read_text()))
    os.environ['BALSA_SPLIT_SHA256'] = manifest['sha256']
    if args.simulator_reuse:
        os.environ['BALSA_SIM_REUSE'] = str(args.simulator_reuse.resolve())
    if args.simulator_bundle:
        os.environ['BALSA_SIM_BUNDLE'] = str(args.simulator_bundle.resolve())
    if args.simulator_integration_epochs:
        if args.simulator_integration_epochs < 1:
            parser.error('Simulator epochs must be positive')
        os.environ['BALSA_SIM_INTEGRATION_EPOCHS'] = str(args.simulator_integration_epochs)
    if manifest['protocol'] == 'LOO-template':
        mapping = json.loads((ROOT / 'manifests/job_templates.json').read_text())
        body = dict(mapping)
        mapping_hash = body.pop('sha256')
        if signed(body)['sha256'] != mapping_hash or mapping_hash != manifest['source']['template_map_sha256']:
            raise ValueError('Template mapping hash mismatch')
        for q, info in mapping['queries'].items():
            if sha((ROOT / 'queries/join-order-benchmark' / q).read_bytes()) != info['sql_sha256']:
                raise ValueError('SQL changed after fold was locked: ' + q)
    if args.smoke and manifest['protocol'] != 'Smoke':
        parser.error('--smoke requires the locked Smoke manifest')
    queries = manifest['train_queries'] + manifest['test_queries']
    for q in queries:
        if Path(q).name != q or not (ROOT / 'queries/join-order-benchmark' / q).is_file():
            parser.error('Missing/unsafe query: ' + q)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(str(output if args.isolated_runtime else ROOT))
    started = time.perf_counter()
    os.environ.setdefault('WANDB_MODE', 'offline')
    os.environ['BALSA_CONNECTION_LOG'] = str(output / 'connections.jsonl')
    with (output / 'preflight.log').open('w') as f:
        subprocess.run([sys.executable, str(ROOT / 'scripts/preflight_balsa.py')],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    import random
    import numpy as np
    import torch
    import experiments
    import balsa.params_registry
    from run import BalsaAgent
    from balsa.run_instrumentation import phase
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    p = balsa.params_registry.Get('Balsa_BaseQuery1_Smoke' if args.smoke else
        'Balsa_JOBRandSplit' if manifest['protocol'] == 'Random-52' else 'Balsa_BaseQuery1')
    p.query_glob = queries
    if args.isolated_runtime:
        p.query_dir = str(ROOT / 'queries/join-order-benchmark')
    p.test_query_glob = manifest['test_queries']
    p.structured_split_name = manifest['split_id']
    p.structured_log_dir = str(output)
    # Never share expert labels/replay/simulator across folds. Missing expert
    # data is collected by the real baseline, then used by a separate agent.
    p.init_experience = str(output / 'expert.pkl')
    p.sim_checkpoint = None
    if args.simulator_checkpoint:
        p.sim_checkpoint = str(args.simulator_checkpoint.resolve())
    p.prev_replay_buffers_glob = None
    p.prev_replay_buffers_glob_val = None
    p.agent_checkpoint = None
    p.use_local_execution = True
    p.db = os.environ.get('BALSA_DB', os.environ.get('PGDATABASE', 'imdbload'))
    p.drop_cache = False
    p.sim_cache_dir = str(output / 'simulator')
    if args.smoke or args.simulator_smoke:
        p.skip_training_on_timeouts = True
    if args.iterations is not None:
        p.val_iters = args.iterations
    if args.initial_timeout_ms is not None:
        p.initial_timeout_ms = args.initial_timeout_ms
    if args.independent_eval_only:
        p.test_every_n_iters = 1000000
    if args.simulator_smoke:
        if manifest['protocol'] not in ('LOO-template', 'BaseQuery-1', 'Random-52'):
            parser.error('Simulator smoke requires a real protocol manifest')
        p.sim = True
        p.epochs = 1
        p.val_iters = 1
        p.validate_fraction = 0
        p.sim_skip_geq = 6
        p.skip_training_on_expert = False
        p.beam = 1
        p.search_until_n_complete_plans = 1
        p.checkpoint_every_n_iters = 1
        p.initial_timeout_ms = 2000
        # Test is performed by the independent evaluator, not repeated here.
        p.test_every_n_iters = 1000000
    if args.smoke:
        p.initial_timeout_ms = 10000
    # Persist exact effective settings before baseline/simulator can consume
    # resources or be interrupted. Only val_iters differs from full training.
    (output / 'config.txt').write_text(p.ToText())
    (output / 'split.json').write_text(json.dumps(manifest, indent=2))
    if args.expert_data:
        import pickle
        from balsa.evaluation_artifacts import file_hash
        with args.expert_data.resolve().open('rb') as f:
            workload = pickle.load(f)
        actual = {n.info['query_name'] + '.sql' for n in workload.Queries(split='all')}
        if actual != set(queries):
            raise ValueError('Expert workload membership differs from manifest')
        shutil.copy2(str(args.expert_data.resolve()), p.init_experience)
        (output / 'expert_provenance.json').write_text(json.dumps(dict(
            source=str(args.expert_data.resolve()), sha256=file_hash(p.init_experience),
            result_source='CACHE', purpose='expert initialization, not fresh baseline benchmark'), indent=2))
    else:
        baseline = p.Copy()
        baseline.run_baseline = True
        with phase('baseline_initialization'):
            agent = BalsaAgent(baseline)
        with phase('postgres_expert_baseline'):
            agent.Run()
        agent.wandb_logger.experiment.finish()
        import ray
        ray.shutdown()
    with phase('agent_and_simulator_initialization'):
        agent = BalsaAgent(p)
    agent.locked_manifest = manifest
    agent.run_seed = args.seed
    agent.structured_logger.metadata.update(protocol=manifest['protocol'], fold_id=manifest['fold_id'],
        seed=args.seed, split_sha256=manifest['sha256'])
    (output / 'config.txt').write_text(p.ToText())
    (output / 'split.json').write_text(json.dumps(manifest, indent=2))
    agent.Run()
    for name in ('local_metrics.jsonl', 'postgres-conf.txt'):
        logged = Path(agent.wandb_logger.experiment.dir) / name
        if logged.is_file():
            shutil.copy2(str(logged), str(output / name))
    bundles = list(output.glob('*/checkpoints/evaluation.pt'))
    if len(bundles) != 1:
        raise RuntimeError('Expected exactly one reloadable evaluation bundle')
    (output / 'training_summary.json').write_text(json.dumps(dict(
        wall_clock_seconds=time.perf_counter() - started,
        checkpoint=str(bundles[0]), seed=args.seed,
        smoke=bool(args.smoke or args.simulator_smoke), simulator=bool(p.sim),
        membership_counts=dict(train=len(manifest['train_queries']), test=len(manifest['test_queries']))), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
