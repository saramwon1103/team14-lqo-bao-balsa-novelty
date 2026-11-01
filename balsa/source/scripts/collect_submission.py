#!/usr/bin/env python3
"""Validate eval artifacts and copy only selected results, with SHA-256 checksums."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument('--raw-train-log', type=Path)
    parser.add_argument('--raw-test-log', type=Path)
    parser.add_argument('--report', type=Path, help='Optional explicit handoff report')
    args = parser.parse_args()
    root = args.evaluation.resolve()
    summary = json.loads((root / 'evaluation_summary.json').read_text())
    if not summary['unchanged'] or summary['errors']:
        raise ValueError('Evaluation did not pass')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(args.checkpoint) != summary['checkpoint_sha256']:
        raise ValueError('Wrong checkpoint')
    training = args.checkpoint.resolve().parent.parent
    evidence_files = list(training.glob('fit_evidence_*.json'))
    if not evidence_files:
        raise ValueError('Training evidence missing next to checkpoint')
    if not any((lambda e: e['did_fit'] and e['training_batches'] > 0
                and e['global_step'] > 0 and e['changed_tensors'] > 0
                and e['weights_before'] != e['weights_after'])(json.loads(p.read_text()))
               for p in evidence_files):
        raise ValueError('No evidence of real training')
    with (root / 'balsa_results.csv').open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty results')
    for row in rows:
        if row['status'] != 'COMPLETED' or row['result_source'] != 'MEASURED':
            raise ValueError('Incomplete/non-measured results')
        for key in ('hinted_sql_path', 'explain_json_path'):
            p = Path(row[key]).resolve()
            if root not in p.parents or not p.is_file():
                raise ValueError('Missing or unsafe artifact path')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    shutil.copytree(str(root), str(args.output_dir / 'evaluation'))
    shutil.copy2(str(args.checkpoint), str(args.output_dir / 'evaluation.pt'))
    target = args.output_dir / 'training'
    target.mkdir()
    for p in evidence_files + [training / 'experiment.log', training / 'split.json']:
        shutil.copy2(str(p), str(target / p.name))
    shutil.copytree(str(training / 'iterations'), str(target / 'iterations'))
    for name in ('preflight.log', 'connections.jsonl', 'expert_provenance.json', 'config.txt', 'training_summary.json', 'local_metrics.jsonl', 'postgres-conf.txt'):
        p = training.parent / name
        if p.is_file():
            shutil.copy2(str(p), str(target / name))
    simulator = training.parent / 'simulator'
    if simulator.exists():
        (target / 'simulator').mkdir()
        for name in ('simulator.ckpt', 'provenance.json'):
            shutil.copy2(str(simulator / name), str(target / 'simulator' / name))
    sidecar = Path(str(args.checkpoint) + '.json')
    if sidecar.is_file():
        shutil.copy2(str(sidecar), str(args.output_dir / 'evaluation.pt.json'))
    if args.raw_train_log:
        shutil.copy2(str(args.raw_train_log), str(target / 'raw_train.log'))
    if args.raw_test_log:
        shutil.copy2(str(args.raw_test_log), str(args.output_dir / 'evaluation/raw_test.log'))
    if args.report:
        shutil.copy2(str(args.report), str(args.output_dir / 'HANDOFF.md'))
    for row in rows:
        for key in ('hinted_sql_path', 'explain_json_path'):
            row[key] = str(Path(row[key]).resolve().relative_to(root))
    with (args.output_dir / 'evaluation/balsa_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Explicit allowlist: never collect DB data, environments, credentials,
    # expert pickles, W&B state or arbitrary contents of the working directory.
    for name in ('balsa', 'pg_executor', 'queries', 'scripts', 'tests', 'manifests', 'conf'):
        shutil.copytree(str(args.source_root / name), str(args.output_dir / 'source' / name),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '*.egg-info'))
    for name in ('run.py', 'sim.py', 'train_utils.py', 'experiments.py',
                 'setup.py', 'requirements.txt', 'LICENSE', 'README.md', 'README_BASEQUERY1.md'):
        shutil.copy2(str(args.source_root / name), str(args.output_dir / 'source' / name))
    checksums = {str(p.relative_to(args.output_dir)): digest(p)
                 for p in args.output_dir.rglob('*') if p.is_file()}
    (args.output_dir / 'artifact_manifest.json').write_text(json.dumps(checksums, indent=2))
    print('Collected:', args.output_dir)


if __name__ == '__main__':
    main()
