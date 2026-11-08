#!/usr/bin/env python3
"""Train process exits, then launch fresh eval; common entry for all locked protocols."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--simulator-smoke', action='store_true')
    mode.add_argument('--allow-full', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--timeout-ms', type=int, default=60000)
    parser.add_argument('--expert-data', type=Path, help='Trusted baseline workload for a retry; exact membership checked')
    parser.add_argument('--iterations', type=int, help='Full-configuration pilot iteration override (requires --allow-full)')
    parser.add_argument('--initial-timeout-ms', type=int)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = dict(status='TRAINING', seed=args.seed, manifest=str(args.manifest.resolve()))
    mode_arg = '--smoke' if args.smoke else '--simulator-smoke' if args.simulator_smoke else '--allow-full'
    train = [sys.executable, str(scripts / 'train_split.py'), '--manifest', str(args.manifest.resolve()),
             '--output-dir', str(root / 'training'), mode_arg, '--seed', str(args.seed)]
    if args.expert_data:
        train += ['--expert-data', str(args.expert_data.resolve())]
    if args.iterations is not None:
        train += ['--iterations', str(args.iterations)]
    if args.initial_timeout_ms is not None:
        train += ['--initial-timeout-ms', str(args.initial_timeout_ms)]
    with (root / 'raw_train.log').open('w') as log:
        code = subprocess.run(train, stdout=log, stderr=subprocess.STDOUT).returncode
    status['train_exit_code'] = code
    if code == 0:
        checkpoint = json.loads((root / 'training/training_summary.json').read_text())['checkpoint']
        status.update(status='EVALUATING', checkpoint=checkpoint)
        evaluate = [sys.executable, str(scripts / 'evaluate_checkpoint.py'), '--checkpoint', checkpoint,
                    '--output-dir', str(root / 'evaluation'), '--seed', str(args.seed),
                    '--repeats', str(args.repeats), '--timeout-ms', str(args.timeout_ms)]
        with (root / 'raw_test.log').open('w') as log:
            code = subprocess.run(evaluate, stdout=log, stderr=subprocess.STDOUT).returncode
        status['eval_exit_code'] = code
    status.update(status='COMPLETED' if code == 0 else 'FAILED',
                  wall_clock_seconds=time.perf_counter() - started)
    (root / 'run_summary.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    return code


if __name__ == '__main__':
    sys.exit(main())
