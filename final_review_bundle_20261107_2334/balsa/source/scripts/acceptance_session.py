#!/usr/bin/env python3
"""Sequential bounded Fold02/Random logging acceptance; no benchmark sweep."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from pilot_supervisor import (atomic_json, available_memory, command, directory_bytes,
                              service_property, start_service, safe_export, GIB, PYTHON, SOURCE)


def digest(path):
    value = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fuse2fs', type=Path, required=True)
    parser.add_argument('--continue-existing', action='store_true', help='Correct pre-fit infrastructure failure without resetting budgets or recreating image')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    previous = None
    retry = 0
    if args.continue_existing:
        previous = json.loads((root / 'session.json').read_text())
        if any(e.get('did_fit') for s in previous['stages'] for e in s.get('fit_evidence', [])):
            raise ValueError('This continuation only supports pre-fit failures, not training resume')
        retry = previous.get('retry', 0) + 1
        atomic_json(root / ('attempt_{}_before_correction.json'.format(retry)), previous)
    else:
        root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic() - (time.time()-previous['started_at'] if previous else 0)
    global_deadline = began + 4 * 3600
    prior = SOURCE.parent / 'results/pilot_full_fold02_seed0_chunked24g'
    policy = json.loads((prior / 'resource_summary.json').read_text())
    hard = policy['ml_memory_max_bytes']
    guards = policy['runtime_constraints']
    assert hard == 24 * GIB and guards['graceful_stop_memory_bytes'] == 23 * GIB
    assert guards['machine_min_available_bytes'] == 3 * GIB
    state = dict(status='PREPARING', started_at=time.time(), total_budget_seconds=14400,
                 phase_budget_seconds=7200, guards=guards, ml_memory_max_bytes=hard,
                 artifact_allocated_limit_bytes=20 * GIB, stages=[], label='PIPELINE_LOGGING_ACCEPTANCE_NOT_BENCHMARK')
    if previous:
        state['started_at'] = previous['started_at']
        state['retry'] = retry
        state['previous_attempts'] = previous.get('previous_attempts', []) + previous['stages']
    atomic_json(root / 'session.json', state)
    import psycopg2
    conn = psycopg2.connect(host='/tmp', port=5432, user='wings', dbname='imdbload',
                            application_name='balsa-acceptance-supervisor')
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT current_setting('data_directory'),current_setting('shared_buffers'),current_setting('work_mem'),current_setting('max_parallel_workers_per_gather')")
    pgpath, shared, work, parallel = cur.fetchone()
    assert (shared, work, parallel) == ('4GB', '64MB', '2')
    pgpath = Path(pgpath)
    initial_pg = directory_bytes(pgpath)['allocated']
    if available_memory() < 20 * GIB or shutil.disk_usage(str(root)).free < 64 * GIB:
        raise RuntimeError('Insufficient safe RAM/disk headroom')
    ident = Path(previous['cgroup']).name[:-6] if previous else 'balsa-accept-' + str(os.getpid())
    original_pid = ident.rsplit('-', 1)[1]
    slice_name = ident + '.slice'
    mount = Path('/tmp/opencode/ac' + original_pid)
    mount.mkdir(exist_ok=bool(previous))
    old_mount = Path('/tmp/opencode/ai' + original_pid)
    old_mount.mkdir(exist_ok=bool(previous))
    image = root / 'session.ext2'
    # Allocate real SSD blocks up front; never use sparse logical size as a safety claim.
    if not previous:
        command(['fallocate', '-l', str(17 * GIB), str(image)])
        command(['/usr/sbin/mkfs.ext2', '-F', '-q', '-E', 'nodiscard', '-m', '0', str(image)])
    elif not image.is_file() or image.stat().st_size != 17 * GIB:
        raise ValueError('Existing shared image missing or resized')
    fsunit = ident + '-fs.service'
    inputunit = ident + '-input.service'
    start_service(fsunit, [str(args.fuse2fs), '-f', '-o', 'fakeroot', str(image), str(mount)],
                  {}, slice_name, str(root / 'filesystem.log'), 24)
    start_service(inputunit, [str(args.fuse2fs), '-f', '-o', 'ro,fakeroot',
                             str(prior / 'artifacts.ext2'), str(old_mount)],
                  {}, slice_name, str(root / 'input_filesystem.log'), 24)
    for _ in range(100):
        if os.path.ismount(str(mount)) and os.path.ismount(str(old_mount)):
            break
        time.sleep(.1)
    assert os.path.ismount(str(mount)) and os.path.ismount(str(old_mount))
    command(['systemctl', '--user', 'set-property', '--runtime', slice_name,
             'MemoryMax=24G', 'MemoryHigh=23G', 'MemorySwapMax=0'])
    cg = Path('/sys/fs/cgroup') / service_property(slice_name, 'ControlGroup').lstrip('/')
    assert int((cg / 'memory.max').read_text()) == hard
    assert int((cg / 'memory.high').read_text()) == guards['graceful_stop_memory_bytes']
    assert int((cg / 'memory.swap.max').read_text()) == 0
    state['cgroup'] = str(cg)
    state['real_allocated_image_bytes'] = image.stat().st_blocks * 512
    state['source_image_sha256_expected'] = json.loads((prior / 'artifact_manifest.json').read_text())['artifacts.ext2']
    timers = []
    live_units = []
    for name, seconds in [('total', int(global_deadline - time.monotonic() - 60))]:
        timer = ident + '-' + name
        command(['systemd-run', '--user', '--unit=' + timer, '--on-active={}s'.format(seconds),
                 '--timer-property=AccuracySec=1s', '/usr/bin/systemctl', '--user', 'stop', slice_name])
        timers.append(timer + '.timer')
    peaks = dict(memory=0, min_available=available_memory(), postgres_growth=0, live_temp=0)
    if previous:
        for key in peaks:
            if key in previous.get('peaks', {}):
                peaks[key] = (min(peaks[key], previous['peaks'][key]) if key=='min_available'
                              else max(peaks[key], previous['peaks'][key]))
    stop_safety = None

    def cancel_queries():
        cur.execute("SELECT pid,pg_cancel_backend(pid) FROM pg_stat_activity WHERE application_name=%s AND state='active' AND pid<>pg_backend_pid()", (ident,))

    def observe(unit, stage, deadline, reserve=0):
        nonlocal stop_safety
        while True:
            now = time.monotonic()
            memory = int((cg / 'memory.current').read_text())
            peak = int((cg / 'memory.peak').read_text())
            available = available_memory()
            pg_growth = max(0, directory_bytes(pgpath)['allocated'] - initial_pg)
            temp = sum(directory_bytes(p)['allocated'] for p in pgpath.glob('base/**/pgsql_tmp'))
            art = directory_bytes(root)
            free = shutil.disk_usage(str(root)).free
            current = dict(elapsed=now-began, stage=stage['name'], unit=unit,
                memory_bytes=memory, kernel_peak_bytes=peak, available_bytes=available,
                memory_stat=(cg / 'memory.stat').read_text(),
                pg_growth_bytes=pg_growth, live_temp_bytes=temp, artifact=art,
                host_free_bytes=free, image_free_bytes=shutil.disk_usage(str(mount)).free)
            with (root / 'resource_samples.jsonl').open('a') as f:
                f.write(json.dumps(current) + '\n')
            peaks['memory'] = max(peaks['memory'], peak)
            peaks['min_available'] = min(peaks['min_available'], available)
            peaks['postgres_growth'] = max(peaks['postgres_growth'], pg_growth)
            peaks['live_temp'] = max(peaks['live_temp'], temp)
            stage['peak_memory_bytes'] = max(stage.get('peak_memory_bytes', 0), memory)
            sub = service_property(unit, 'SubState')
            active = service_property(unit, 'ActiveState')
            if sub == 'exited' or active in ('inactive', 'failed'):
                return dict(exit_code=service_property(unit, 'ExecMainStatus'), result=service_property(unit, 'Result'))
            reason = None
            if now >= min(deadline - 90, global_deadline - 90):
                reason = 'WALL_CLOCK_CLEANUP_RESERVE'
            elif reserve and now >= deadline - reserve:
                reason = 'TRAIN_RESERVE_FOR_FRESH_EVAL'
            elif memory >= guards['graceful_stop_memory_bytes']:
                reason = 'ML_MEMORY_GUARD'
            elif available <= guards['machine_min_available_bytes']:
                reason = 'MACHINE_AVAILABLE_GUARD'
            elif pg_growth >= guards['postgres_growth_stop_bytes'] or temp >= guards['live_temp_stop_bytes']:
                reason = 'POSTGRES_GROWTH_TEMP_GUARD'
            elif free < 64 * GIB or current['image_free_bytes'] < 256 * 1024**2 or art['allocated'] >= 19 * GIB:
                reason = 'DISK_OR_SESSION_ARTIFACT_GUARD'
            if reason:
                if reason not in ('TRAIN_RESERVE_FOR_FRESH_EVAL', 'WALL_CLOCK_CLEANUP_RESERVE'):
                    stop_safety = reason
                cancel_queries()
                command(['systemctl', '--user', 'stop', unit])
                cancel_queries()
                return dict(stop_reason=reason, exit_code=service_property(unit, 'ExecMainStatus'), result=service_property(unit, 'Result'))
            time.sleep(1)

    def service(suffix, argv, stage, env, deadline, reserve=0):
        unit = ident + '-' + suffix + '.service'
        logfile = mount / stage['name'] / (suffix + '.log')
        live_units.append(unit)
        start_service(unit, argv, env, slice_name, str(logfile), 24)
        result = observe(unit, stage, deadline, reserve)
        command(['systemctl', '--user', 'stop', unit])
        live_units.remove(unit)
        return result

    try:
        state['status'] = 'RUNNING'
        for name, manifest, reserve in [('A_fold02', 'loo_template/fold_02.json', 900), ('B_random52', 'random52.json', 1800)]:
            is_fold = name == 'A_fold02'
            if retry:
                name += '_retry' + str(retry)
            stage_began = began if is_fold else time.monotonic()
            deadline = min(stage_began + 7200, global_deadline)
            stage = dict(name=name, started_at=time.time(), agent_seed=0,
                         manifest=str(SOURCE / 'manifests' / manifest), eval_reserve_seconds=reserve,
                         label='PARTIAL_SIM_FULL_AGENT_PILOT' if is_fold else 'INTEGRATION_SIMULATOR_1_EPOCH_FULL_AGENT')
            state['stages'].append(stage)
            atomic_json(root / 'session.json', state)
            stage_dir = mount / name
            stage_dir.mkdir()
            runtime = mount / ('rA' if is_fold else 'rB')
            runtime.mkdir(exist_ok=True)
            tmp = stage_dir / 'tmp'
            tmp.mkdir()
            env = dict(CUDA_VISIBLE_DEVICES='', WANDB_MODE='offline', PGHOST='/tmp', PGPORT='5432',
                PGUSER='wings', BALSA_DB='imdbload', PGDATABASE='imdbload', PGAPPNAME=ident,
                PYTHONDONTWRITEBYTECODE='1', BALSA_PHASE_LOG=str(stage_dir / 'phases.jsonl'),
                BALSA_RUNTIME_DIR=str(runtime), RAY_TMPDIR=str(runtime), TMPDIR=str(tmp),
                XDG_CACHE_HOME=str(stage_dir / 'cache'), WANDB_CACHE_DIR=str(stage_dir / 'wandb-cache'),
                WANDB_CONFIG_DIR=str(stage_dir / 'wandb-config'))
            timer = ident + '-' + name + '-deadline'
            command(['systemd-run', '--user', '--unit=' + timer,
                '--on-active={}s'.format(max(1, int(deadline-time.monotonic()-60))),
                '--timer-property=AccuracySec=1s', '/usr/bin/systemctl', '--user', 'stop', slice_name])
            timers.append(timer + '.timer')
            argv = [PYTHON, '-u', str(SOURCE / 'scripts/train_split.py'),
                '--manifest', stage['manifest'], '--output-dir', str(stage_dir / 'training'),
                '--allow-full', '--iterations', '3', '--seed', '0', '--initial-timeout-ms', '10000',
                '--isolated-runtime', '--independent-eval-only']
            if is_fold:
                prepared = mount / 'A_fold02/input/simulator_initialization.json'
                if not previous:
                    stage['prepare'] = service('prepare-A', [PYTHON, '-u', str(SOURCE / 'scripts/prepare_fold02_simulator.py'),
                        '--input-mount', str(old_mount), '--output-dir', str(stage_dir / 'input')], stage, env, deadline, reserve)
                    if stage['prepare'].get('exit_code') != '0' or stage['prepare'].get('stop_reason'):
                        state['status'] = 'BLOCKED_PREPARATION_OR_CORRECTNESS'
                        break
                elif not prepared.is_file():
                    raise ValueError('Verified prepared simulator bundle missing')
                argv += ['--simulator-bundle', str(prepared),
                         '--expert-data', str(old_mount / 'training/expert.pkl')]
                shutil.copy2(str(old_mount / 'training/gkyar5oy/postgres_baseline.csv'), str(stage_dir / 'postgres_baseline.csv'))
            else:
                argv += ['--simulator-integration-epochs', '1']
            stage['train_command'] = argv
            stage['train'] = service('train-' + name, argv, stage, env, deadline, reserve)
            checkpoints = list(stage_dir.glob('training/*/checkpoints/evaluation.pt'))
            if len(checkpoints) == 1 and not stop_safety and time.monotonic() < deadline - 120:
                argv_eval = [PYTHON, '-u', str(SOURCE / 'scripts/evaluate_checkpoint.py'),
                    '--checkpoint', str(checkpoints[0]), '--output-dir', str(stage_dir / 'evaluation'),
                    '--seed', '0', '--repeats', '1', '--timeout-ms', '30000']
                stage['eval_command'] = argv_eval
                stage['eval'] = service('eval-' + name, argv_eval, stage, env, deadline)
            else:
                stage['eval'] = dict(status='NOT_RUN', reason='no completed bundle or safety/time guard')
            stage['elapsed_seconds'] = time.monotonic() - stage_began
            events = [json.loads(line) for line in (stage_dir / 'phases.jsonl').read_text().splitlines()]
            stage['completed_iterations'] = [e['value_iter'] for e in events if e['stage']=='value_iteration' and e['event']=='END' and e.get('completed')]
            stage['fit_evidence'] = [json.loads(p.read_text()) for p in stage_dir.glob('training/*/fit_evidence_*.json')]
            es = stage_dir / 'evaluation/evaluation_summary.json'
            if es.exists():
                stage['evaluation_summary'] = json.loads(es.read_text())
            stage['accepted'] = bool(any(e['did_fit'] and e['changed_tensors'] and e['global_step'] for e in stage['fit_evidence'])
                and stage.get('evaluation_summary', {}).get('unchanged') and not stage['evaluation_summary'].get('errors'))
            command(['systemctl', '--user', 'stop', timer + '.timer'])
            atomic_json(root / 'session.json', state)
            # Correctness failures must be fixed before starting Random; timeouts alone are recorded, not hidden.
            if stop_safety or (stage['train'].get('exit_code') != '0' and not stage['train'].get('stop_reason')):
                state['status'] = 'BLOCKED_SAFETY_OR_TRAIN_FAILURE'
                break
            csvpath = stage_dir / 'evaluation/balsa_results.csv'
            if csvpath.exists():
                rows = list(csv.DictReader(csvpath.open()))
                stage['test_statuses'] = {status: sum(r['status']==status for r in rows)
                    for status in sorted({r['status'] for r in rows})}
                if any(r['status'] in ('ERROR', 'HINT_MISMATCH') for r in rows):
                    state['status'] = 'BLOCKED_EVALUATOR_CORRECTNESS_REVIEW_REQUIRED'
                    break
        else:
            state['status'] = ('LOGGING_ACCEPTED_BOTH_PROTOCOLS' if all(s.get('accepted') for s in state['stages']) else 'PARTIAL_NOT_ACCEPTED')
    except BaseException as exc:
        state.update(status='SUPERVISOR_ERROR', error=repr(exc))
    finally:
        for unit in live_units:
            cancel_queries()
            subprocess.run(['systemctl', '--user', 'stop', unit], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        state['peaks'] = peaks
        state['memory_events'] = (cg / 'memory.events').read_text()
        state['export'] = safe_export(mount, root / 'accessible_artifacts')
        for p in mount.rglob('*.json'):
            if p.is_symlink() or p.stat().st_size > 32 * 1024**2:
                continue
            out = root / 'accessible_artifacts' / p.relative_to(mount)
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(p), str(out))
        for path in (mount, old_mount):
            command(['fusermount', '-u', str(path)])
        for unit in [fsunit, inputunit] + timers:
            subprocess.run(['systemctl', '--user', 'stop', unit], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        state['elapsed_seconds'] = time.monotonic() - began
        state['artifact_usage'] = directory_bytes(root)
        atomic_json(root / 'session.json', state)
        cur.close()
        conn.close()
    print(json.dumps(state, indent=2))
    return 0 if state['status']=='LOGGING_ACCEPTED_BOTH_PROTOCOLS' else 1


if __name__ == '__main__':
    sys.exit(main())
