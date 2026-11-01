#!/usr/bin/env python3
"""One approved full-config fold-02 pilot, bounded cgroup + fixed-size artifact filesystem.

Requires user systemd/cgroup-v2, fuse2fs and PostgreSQL already configured safely.
No privilege escalation, database restart, deletion, or automatic retries.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

GIB = 1024**3
PYTHON = '/home/wings/miniconda3/envs/balsa/bin/python'
SOURCE = Path(__file__).resolve().parents[1]


def command(args, **kwargs):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, **kwargs).stdout.strip()


def atomic_json(path, data):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(path))


def available_memory():
    return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                    if s.startswith('MemAvailable:'))) * 1024


def directory_bytes(path):
    allocated = logical = 0
    for root, dirs, files in os.walk(str(path)):
        for name in files:
            try:
                st = os.stat(os.path.join(root, name), follow_symlinks=False)
                allocated += st.st_blocks * 512
                logical += st.st_size
            except FileNotFoundError:
                continue
    return dict(allocated=allocated, logical=logical)


def service_property(unit, prop):
    return command(['systemctl', '--user', 'show', unit, '-p', prop, '--value'])


def start_service(unit, args, env, slice_name, stdout='/dev/null', memory_gib=16):
    cmd = ['systemd-run', '--user', '--unit=' + unit, '--slice=' + slice_name,
           '--property=MemoryMax={}G'.format(memory_gib), '--property=MemorySwapMax=0',
           '--property=KillMode=control-group', '--property=TimeoutStopSec=25s',
           '--property=RuntimeMaxSec=7200s', '--property=OOMPolicy=stop',
           '--property=LimitCORE=0',
           '--property=RemainAfterExit=yes',
           '--property=StandardOutput=file:' + stdout,
           '--property=StandardError=inherit']
    for key, value in env.items():
        cmd += ['--setenv={}={}'.format(key, value)]
    return command(cmd + args)


def safe_export(mount, destination):
    """Keep accessible essentials after unmount; leave all larger data in image."""
    budget = 512 * 1024**2
    used = 0
    omitted = []
    priority = [p for p in mount.rglob('*') if p.is_file() and (
        p.name in ('evaluation.pt', 'evaluation.pt.json', 'simulator.ckpt', 'provenance.json',
                   'run_provenance.json', 'config.txt', 'split.json', 'training_summary.json', 'evaluation_summary.json',
                   'balsa_results.csv', 'postgres_baseline.csv', 'predictions.json', 'environment.json')
        or p.name.startswith('fit_evidence_'))]
    priority += [p for p in mount.rglob('*') if p.is_file() and p not in priority and (
        p.suffix in ('.log', '.sql') or 'evaluation/plans/' in str(p.relative_to(mount))
        or p.name in ('phases.jsonl', 'connections.jsonl', 'local_metrics.jsonl'))]
    for p in priority:
        relative = p.relative_to(mount)
        size = p.stat().st_size
        if used + size > budget or size > 128 * 1024**2:
            omitted.append(str(relative))
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(p), str(target))
        used += size
    # Original paths remain in the retained image; exported CSV must point at
    # the accessible copies after the filesystem is unmounted.
    for path in destination.rglob('balsa_results.csv'):
        with path.open() as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        for row in rows:
            for key in ('hinted_sql_path', 'explain_json_path'):
                value = row.get(key)
                if value:
                    try:
                        relative = Path(value).relative_to(mount)
                    except ValueError:
                        continue
                    row[key] = str(destination / relative)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return dict(exported_bytes=used, omitted_present_in_image=omitted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--fuse2fs', required=True, type=Path)
    parser.add_argument('--split-type', required=True, choices=['LOO-1', 'Random-1', 'BaseQuery-1'])
    parser.add_argument('--ml-memory-gib', type=int, default=16,
                        help='Whole ML slice hard limit; leave headroom for PostgreSQL and OS')
    args = parser.parse_args()
    if not 16 <= args.ml_memory_gib <= 24:
        parser.error('Highest defensible ML hard limit on this 30.6 GiB host is 24 GiB '
                     '(PostgreSQL shared_buffers 4 GiB + OS/apps must remain resident)')
    fusermount = shutil.which('fusermount3') or shutil.which('fusermount')
    if not fusermount:
        raise RuntimeError('fuse2fs requires fusermount3 or fusermount')
    memory_max = args.ml_memory_gib * GIB
    memory_stop = memory_max - GIB
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 7200
    code_files = ['run.py', 'sim.py', 'train_utils.py', 'experiments.py',
                  'balsa/models/treeconv.py', 'balsa/experience.py',
                  'scripts/pilot_supervisor.py', 'scripts/train_split.py',
                  'scripts/evaluate_checkpoint.py', 'balsa/evaluation_artifacts.py',
                  'balsa/run_instrumentation.py']
    atomic_json(root / 'code_sha256.json', {p: hashlib.sha256((SOURCE / p).read_bytes()).hexdigest()
                                          for p in code_files})
    for path in code_files:
        target = root / 'executed_source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(SOURCE / path), str(target))
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get('PGHOST', '/tmp'),
                            dbname=os.environ.get('BALSA_DB', 'imdbload'),
                            user=os.environ.get('PGUSER', 'wings'),
                            application_name='balsa-pilot-supervisor')
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT name,setting,unit FROM pg_settings WHERE name IN ('shared_buffers','work_mem','max_parallel_workers_per_gather','temp_buffers','maintenance_work_mem','temp_file_limit')")
    settings = {name: dict(value=value, unit=unit) for name, value, unit in cur.fetchall()}
    assert settings['shared_buffers']['value'] == '524288', settings
    assert settings['work_mem']['value'] == '65536', settings
    assert settings['max_parallel_workers_per_gather']['value'] == '2', settings
    cur.execute("SELECT current_setting('data_directory'),pg_database_size(current_database())")
    pg_dir, db_initial = cur.fetchone()
    pg_dir = Path(pg_dir)
    initial_pg = directory_bytes(pg_dir)
    if available_memory() < 20 * GIB or shutil.disk_usage(str(root)).free < 64 * GIB:
        raise RuntimeError('Insufficient machine memory/disk headroom; pilot not started')
    unit_base = 'balsa-v2-run-' + str(os.getpid())
    # systemd nests this run slice under balsa-v2.slice; the run-local timer
    # protects the two-hour limit while the outer slice enforces six hours.
    slice_name = unit_base + '.slice'
    fs_unit = unit_base + '-fs.service'
    mount = Path('/tmp/opencode/bp' + str(os.getpid()))
    mount.mkdir(exist_ok=False)
    image = root / 'artifacts.ext2'
    with image.open('xb') as f:
        f.truncate(19 * GIB)
    command(['/usr/sbin/mkfs.ext2', '-F', '-q', '-m', '0', str(image)])
    start_service(fs_unit, [str(args.fuse2fs.resolve()), '-f', '-o', 'fakeroot',
                           str(image), str(mount)], {}, slice_name,
                  str(root / 'filesystem.log'), memory_gib=args.ml_memory_gib)
    for _ in range(100):
        if os.path.ismount(str(mount)):
            break
        time.sleep(0.1)
    if not os.path.ismount(str(mount)):
        raise RuntimeError('Artifact filesystem did not mount')
    command(['systemctl', '--user', 'set-property', '--runtime', slice_name,
             'MemoryMax={}G'.format(args.ml_memory_gib),
             'MemoryHigh={}G'.format(args.ml_memory_gib - 1), 'MemorySwapMax=0'])
    cgroup = Path('/sys/fs/cgroup') / service_property(slice_name, 'ControlGroup').lstrip('/')
    assert int((cgroup / 'memory.max').read_text()) == memory_max
    # Independent aggregate timer is installed before any ML process starts;
    # train and eval never receive separate two-hour budgets.
    watchdog = unit_base + '-aggregate-deadline'
    remaining = max(1, int(deadline - time.monotonic() - 60))
    command(['systemd-run', '--user', '--unit=' + watchdog,
             '--on-active={}s'.format(remaining), '--timer-property=AccuracySec=1s',
             '/usr/bin/systemctl', '--user', 'stop', slice_name])
    atomic_json(root / 'aggregate_deadline_guard.json', dict(
        timer=watchdog + '.timer', slice=slice_name,
        remaining_seconds_when_installed=remaining, stop_grace_seconds=25))
    (mount / 'tmp').mkdir()
    (mount / 'runtime').mkdir()
    env = dict(CUDA_VISIBLE_DEVICES='', WANDB_MODE='offline', PGHOST='/tmp', PGPORT='5432',
               PGUSER=os.environ.get('PGUSER', 'wings'), BALSA_DB='imdbload',
               PGAPPNAME=unit_base, PYTHONDONTWRITEBYTECODE='1',
               BALSA_PHASE_LOG=str(mount / 'phases.jsonl'),
               BALSA_CONNECTION_LOG=str(mount / 'connections.jsonl'),
               BALSA_RUNTIME_DIR=str(mount / 'runtime'), RAY_TMPDIR=str(mount / 'runtime'),
               TMPDIR=str(mount / 'tmp'), XDG_CACHE_HOME=str(mount / 'cache'),
               WANDB_CACHE_DIR=str(mount / 'wandb-cache'),
               WANDB_CONFIG_DIR=str(mount / 'wandb-config'))
    state = dict(status='RUNNING', started_timestamp=time.time(), wall_limit_seconds=7200,
                 ml_memory_max_bytes=memory_max, artifact_limit_bytes=20 * GIB,
                 machine_available_before_ml_bytes=available_memory(),
                 artifact_image_bytes=19 * GIB, export_limit_bytes=512 * 1024**2,
                 postgres_settings=settings, postgres_database_initial_bytes=db_initial,
                 postgres_directory_initial=initial_pg, cgroup=str(cgroup),
                 cgroup_contains=['ML driver', 'Ray workers/daemons', 'eval process', 'artifact filesystem daemon'],
                 configuration=dict(split_type=args.split_type, seed=0, val_iters=3, mode='full',
                                    agent_epochs='Balsa configuration', simulator_integration_epochs=1,
                                    simulator='fresh, split-local, train-only'),
                 runtime_constraints=dict(ray_object_store_bytes=2 * GIB,
                     filesystem='19 GiB FUSE ext2; disk I/O timing affected',
                     graceful_stop_memory_bytes=memory_stop,
                     machine_min_available_bytes=3 * GIB,
                     postgres_growth_stop_bytes=20 * GIB, live_temp_stop_bytes=16 * GIB,
                     train_stop_for_eval_reserve_seconds=600))
    atomic_json(root / 'resource_summary.json', state)
    maximum_memory = maximum_pg_growth = maximum_temp = 0
    maximum_pg_pss = 0
    memory_component_peaks = {}
    minimum_available = available_memory()
    samples = 0

    def cancel_own_queries():
        # Cancel only this pilot's tagged queries, never terminate DB sessions
        # or interfere with queries from other applications.
        cur.execute('SELECT pid,pg_cancel_backend(pid) FROM pg_stat_activity '
                    'WHERE application_name=%s AND pid<>pg_backend_pid() AND state=\'active\'',
                    (unit_base,))
        state.setdefault('controlled_query_cancellations', []).extend(cur.fetchall())

    def postgres_pss():
        pids = [int((pg_dir / 'postmaster.pid').read_text().splitlines()[0])]
        total = 0
        for pid in pids:
            try:
                children = Path('/proc/{}/task/{}/children'.format(pid, pid)).read_text().split()
                pids.extend(int(p) for p in children)
                text = Path('/proc/{}/smaps_rollup'.format(pid)).read_text()
                total += int(next(s.split()[1] for s in text.splitlines() if s.startswith('Pss:'))) * 1024
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        return total

    def observe(unit, phase, stop_for_reserve=False):
        nonlocal maximum_memory, maximum_pg_growth, maximum_temp, maximum_pg_pss, minimum_available, samples
        reason = None
        last_pg = initial_pg
        last_temp = 0
        last_pss = 0
        while True:
            now = time.monotonic()
            memory = int((cgroup / 'memory.current').read_text())
            peak = int((cgroup / 'memory.peak').read_text())
            memory_stat = dict(line.split() for line in (cgroup / 'memory.stat').read_text().splitlines())
            components = {key: int(memory_stat.get(key, 0))
                          for key in ('anon', 'file', 'kernel', 'shmem', 'file_dirty', 'file_writeback')}
            for key, value in components.items():
                memory_component_peaks[key] = max(memory_component_peaks.get(key, 0), value)
            avail = available_memory()
            disk_free = shutil.disk_usage(str(root)).free
            if samples % 5 == 0:
                last_pg = directory_bytes(pg_dir)
                last_temp = sum(directory_bytes(Path(p))['allocated'] for p in
                    glob.glob(str(pg_dir / 'base/*/pgsql_tmp')) + glob.glob(str(pg_dir / 'base/pgsql_tmp')))
                last_pss = postgres_pss()
                cur.execute('SELECT pg_database_size(current_database()),temp_bytes,temp_files FROM pg_stat_database WHERE datname=current_database()')
                database_size, temp_bytes, temp_files = cur.fetchone()
                state['postgres_latest_database_bytes'] = database_size
                state['postgres_cumulative_temp_bytes'] = temp_bytes
                state['postgres_cumulative_temp_files'] = temp_files
            growth = max(0, last_pg['allocated'] - initial_pg['allocated'])
            maximum_memory = max(maximum_memory, memory, peak)
            maximum_pg_growth = max(maximum_pg_growth, growth)
            maximum_temp = max(maximum_temp, last_temp)
            maximum_pg_pss = max(maximum_pg_pss, last_pss)
            minimum_available = min(minimum_available, avail)
            active = service_property(unit, 'ActiveState')
            substate = service_property(unit, 'SubState')
            record = dict(elapsed_seconds=now - started, phase=phase, ml_cgroup_memory_bytes=memory,
                          ml_cgroup_peak_bytes=peak, machine_available_bytes=avail,
                          ml_memory_components_bytes=components,
                          postgres_directory_allocated_bytes=last_pg['allocated'],
                          postgres_growth_bytes=growth, live_temp_bytes=last_temp,
                          postgres_pss_bytes=last_pss,
                          disk_free_bytes=disk_free, filesystem_free_bytes=shutil.disk_usage(str(mount)).free,
                          active_state=active, substate=substate)
            with (root / 'resource_samples.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            samples += 1
            if active in ('inactive', 'failed') or substate == 'exited':
                return service_property(unit, 'ExecMainStatus'), service_property(unit, 'Result'), reason
            if now >= deadline - 90:
                reason = 'WALL_CLOCK_LIMIT'
            elif stop_for_reserve and now >= deadline - 600:
                reason = 'TRAIN_STOPPED_TO_RESERVE_INDEPENDENT_EVAL'
            elif memory >= memory_stop:
                reason = 'ML_MEMORY_GUARD_BEFORE_{}_GIB_HARD_LIMIT'.format(args.ml_memory_gib)
            elif avail <= 3 * GIB:
                reason = 'MACHINE_MEMORY_HEADROOM'
            elif disk_free < 64 * GIB:
                reason = 'HOST_DISK_HEADROOM'
            elif growth >= 20 * GIB or last_temp >= 16 * GIB:
                reason = 'POSTGRES_GROWTH_OR_TEMP_SAFETY_GUARD'
            elif shutil.disk_usage(str(mount)).free < 256 * 1024**2:
                reason = 'ARTIFACT_FILESYSTEM_NEAR_CAPACITY'
            if reason:
                cancel_own_queries()
                command(['systemctl', '--user', 'stop', unit])
                cancel_own_queries()
                return service_property(unit, 'ExecMainStatus'), service_property(unit, 'Result'), reason
            time.sleep(1)

    train_unit = unit_base + '-train.service'
    train = [PYTHON, '-u', str(SOURCE / 'scripts/train_split.py'),
             '--split-master', str(SOURCE.parents[1] / 'analysis/data/split_master.csv'),
             '--split-type', args.split_type,
             '--output-dir', str(mount / 'training'), '--allow-full', '--iterations', '3',
             '--seed', '0', '--initial-timeout-ms', '30000', '--isolated-runtime',
             '--fresh-simulator', '--simulator-integration-epochs', '1']
    train_end = eval_end = None
    try:
        start_service(train_unit, train, env, slice_name, str(mount / 'raw_train.log'), args.ml_memory_gib)
        train_end = observe(train_unit, 'TRAIN', stop_for_reserve=True)
        state['train_service'] = dict(zip(('exit_code', 'result', 'stop_reason'), train_end))
        command(['systemctl', '--user', 'stop', train_unit])
        checkpoint = list(mount.glob('training/*/checkpoints/evaluation.pt'))
        if len(checkpoint) == 1 and deadline - time.monotonic() > 45 and available_memory() > 3 * GIB:
            state['checkpoint'] = str(checkpoint[0].relative_to(mount))
            eval_unit = unit_base + '-eval.service'
            evaluate = [PYTHON, '-u', str(SOURCE / 'scripts/evaluate_checkpoint.py'),
                        '--checkpoint', str(checkpoint[0]), '--output-dir', str(mount / 'evaluation'),
                        '--seed', '0', '--repeats', '1', '--timeout-ms', '30000',
                        '--run-id', unit_base]
            start_service(eval_unit, evaluate, env, slice_name, str(mount / 'raw_test.log'), args.ml_memory_gib)
            eval_end = observe(eval_unit, 'INDEPENDENT_EVAL')
            state['eval_service'] = dict(zip(('exit_code', 'result', 'stop_reason'), eval_end))
            command(['systemctl', '--user', 'stop', eval_unit])
        else:
            state['eval_service'] = dict(status='NOT_RUN', reason='no complete bundle or insufficient safe time/headroom')
        if (state.get('eval_service', {}).get('exit_code') == '0'
                and state.get('eval_service', {}).get('result') == 'success'
                and state.get('train_service', {}).get('exit_code') == '0'
                and state.get('train_service', {}).get('result') == 'success'):
            split_dir = args.split_type
            run_id = unit_base
            common_dir = SOURCE.parents[0] / 'runs' / split_dir / run_id
            export = [PYTHON, str(SOURCE / 'scripts/export_common_results.py'),
                '--training-dir', str(mount / 'training'), '--evaluation-dir', str(mount / 'evaluation'),
                '--split-master', str(SOURCE.parents[1] / 'analysis/data/split_master.csv'),
                '--sql-dir', str(SOURCE / 'queries/join-order-benchmark'), '--output-dir', str(common_dir),
                '--run-id', run_id, '--code-commit', subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=str(SOURCE.parents[1]), text=True).strip(),
                '--raw-training-log', str(mount / 'raw_train.log'), '--raw-eval-log', str(mount / 'raw_test.log')]
            command(export)
            state['common_results'] = str(common_dir)
        events = []
        if (mount / 'phases.jsonl').exists():
            for line in (mount / 'phases.jsonl').read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
        state['completed_value_iterations'] = sorted({e['value_iter'] for e in events
            if e['stage'] == 'value_iteration' and e['event'] == 'END' and e.get('completed')})
        state['phase_timings'] = [e for e in events if e['event'] == 'END']
        state['fit_evidence'] = [json.loads(p.read_text()) for p in mount.glob('training/*/fit_evidence_*.json')]
        state['status'] = ('COMPLETED_3_ITERATIONS_AND_EVAL' if len(state['completed_value_iterations']) == 3
            and train_end == ('0', 'success', None) and eval_end == ('0', 'success', None)
            else 'STOPPED_OR_FAILED_PARTIAL')
    except BaseException as exc:
        state.update(status='SUPERVISOR_ERROR', error=repr(exc))
        cancel_own_queries()
        subprocess.run(['systemctl', '--user', 'stop', train_unit], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        subprocess.run(['systemctl', '--user', 'stop', unit_base + '-eval.service'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    finally:
        state.update(elapsed_seconds=time.monotonic() - started, ml_peak_bytes=maximum_memory,
                     machine_minimum_available_bytes=minimum_available,
                     postgres_maximum_growth_bytes=maximum_pg_growth,
                     postgres_live_temp_peak_bytes=maximum_temp, samples=samples)
        state['postgres_sampled_peak_pss_bytes'] = maximum_pg_pss
        state['ml_memory_component_sampled_peaks_bytes'] = memory_component_peaks
        state['memory_events'] = (cgroup / 'memory.events').read_text()
        state['filesystem_usage_before_export'] = dict(total=shutil.disk_usage(str(mount)).total,
                                                        used=shutil.disk_usage(str(mount)).used)
        state['export'] = safe_export(mount, root / 'accessible_artifacts')
        state['ml_peak_bytes'] = max(maximum_memory, int((cgroup / 'memory.peak').read_text()))
        command([fusermount, '-u', str(mount)])
        subprocess.run(['systemctl', '--user', 'stop', fs_unit], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        subprocess.run(['systemctl', '--user', 'stop', watchdog + '.timer'],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        state['outer_artifacts'] = directory_bytes(root)
        common_dir = Path(state['common_results']) if state.get('common_results') else None
        common_bytes = directory_bytes(common_dir) if common_dir and common_dir.exists() else dict(allocated=0, logical=0)
        state['common_results_artifacts'] = common_bytes
        total_allocated = state['outer_artifacts']['allocated'] + common_bytes['allocated']
        state['total_new_artifact_allocated_bytes'] = total_allocated
        if total_allocated > 20 * GIB:
            state['status'] = 'ARTIFACT_LIMIT_EXCEEDED'
        state['elapsed_seconds'] = time.monotonic() - started
        atomic_json(root / 'resource_summary.json', state)
        cur.close()
        conn.close()
    print(json.dumps(state, indent=2))
    return 0 if state['status'] == 'COMPLETED_3_ITERATIONS_AND_EVAL' else 1


if __name__ == '__main__':
    sys.exit(main())
