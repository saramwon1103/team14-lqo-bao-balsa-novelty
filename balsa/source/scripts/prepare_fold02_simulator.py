#!/usr/bin/env python3
"""Recover complete same-fold encoders from trusted pilot caches without training."""
import argparse
import gc
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))


def vector_digest(vectors):
    digest = hashlib.sha256()
    for vec in vectors:
        digest.update(vec.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-mount', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import sim
    from balsa.util import plans_lib, postgres, dataset
    from balsa.evaluation_artifacts import file_hash, weight_hash
    old = SOURCE.parent / 'results/pilot_full_fold02_seed0_chunked24g'
    manifest = json.loads((SOURCE / 'manifests/loo_template/fold_02.json').read_text())
    if json.loads((old / 'accessible_artifacts/training/split.json').read_text()) != manifest:
        raise ValueError('Source manifest differs')
    checks = json.loads((old / 'artifact_manifest.json').read_text())
    for relative, digest in checks.items():
        if file_hash(str(old / relative)) != digest:
            raise ValueError('Pilot artifact checksum mismatch: ' + relative)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    mount = args.input_mount
    raw_path = mount / 'training/simulator/sim-data-6af55cf7.pkl'
    feat_path = mount / 'training/simulator/sim-featurized-7456ee24.pkl'
    ckpt = mount / 'training/tensorboard_logs_balsa/1_j2vdgg2l/checkpoints/epoch=10.ckpt'
    expected = json.loads((old / 'accessible_artifacts/simulator_best_checkpoint_verification.json').read_text())
    if file_hash(str(ckpt)) != expected['checkpoint_sha256']:
        raise ValueError('Simulator checkpoint does not match recorded metadata')
    with (mount / 'training/expert.pkl').open('rb') as handle:
        expert = pickle.load(handle)
    wi = expert.workload_info.Copy()
    wi.table_num_rows = postgres.GetAllTableNumRows(wi.rel_names)
    qf = sim.SimQueryFeaturizer(wi)  # Exact saved config: sim_query_featurizer=True; stateless.
    qf.Fit([n for n in expert.Queries(split='all') if n.info['query_name'] + '.sql' in manifest['train_queries']])
    with raw_path.open('rb') as handle:
        points = pickle.load(handle)
    paths = sorted({p.goal.info['path'] for p in points})
    if not {Path(p).name for p in paths} <= set(manifest['train_queries']):
        raise ValueError('Simulator data contains held-out SQL')
    goals = [p.goal for p in points]
    costs_digest = hashlib.sha256(np.asarray([p.cost for p in points]).tobytes()).hexdigest()
    plans_lib.GatherUnaryFiltersInfo(goals)
    postgres.EstimateFilterRows(goals)
    query_digest = vector_digest(qf(node) for node in goals)
    count = len(points)
    del goals, points
    gc.collect()
    with feat_path.open('rb') as handle:
        data = torch.load(handle, map_location='cpu')
    if len(data[0]) != count or vector_digest(data[0]) != query_digest:
        raise ValueError('Full ordered query feature reconstruction differs from cached simulator features')
    if hashlib.sha256(np.asarray(data[3]).tobytes()).hexdigest() != costs_digest:
        raise ValueError('Cached labels changed or samples reordered')
    transform = dataset.PlansDataset(*data[:3], data[3], tree_conv=True,
                                    transform_cost=['log1p', 'standardize'])
    old_log = (old / 'accessible_artifacts/raw_train.log').read_text()
    mean, std = re.search(r'costs stats mean ([\d.e+-]+) std ([\d.e+-]+)', old_log).groups()
    if not (transform.mean == float(mean) and transform.std == float(std)):
        raise ValueError('Recovered normalization differs from recorded simulator training')
    dims = (data[0][0].shape[0], data[1].shape[1])
    model = sim.SimModel.load_from_checkpoint(str(ckpt), map_location='cpu')
    if weight_hash(model) != expected['loaded_model_weight_sha256']:
        raise ValueError('Loaded simulator weight hash differs')
    if dims != (model.hparams.query_feat_dims, model.hparams.plan_feat_dims):
        raise ValueError('Recovered vocabulary does not match model dimensions')
    transform.FreeData()
    del data
    gc.collect()
    path = args.output_dir / 'simulator_initialization.pt'
    torch.save(dict(model=model, query_featurizer=qf, workload_info=wi,
                    transform=transform), str(path) + '.tmp')
    Path(str(path) + '.tmp').replace(path)
    metadata = dict(bundle_path=str(path.resolve()), bundle_sha256=file_hash(str(path)),
        split_sha256=manifest['sha256'], checkpoint_sha256=expected['checkpoint_sha256'],
        weight_sha256=weight_hash(model), checkpoint_epoch=expected['epoch'],
        checkpoint_optimizer_steps=expected['global_step'], simulator_converged=False,
        purpose='fresh agent initialized from partially trained same-fold simulator; not exact resume',
        num_points=count, collected_query_paths=paths, query_features_ordered_sha256=query_digest,
        labels_ordered_sha256=costs_digest, all_query_features_bit_identical=True,
        label_mean=float(transform.mean), label_std=float(transform.std),
        rel_ids=wi.rel_ids.tolist(), scan_types=wi.scan_types.tolist(), join_types=wi.join_types.tolist(),
        sim_data_sha256=file_hash(str(raw_path)), sim_featurized_sha256=file_hash(str(feat_path)))
    (args.output_dir / 'simulator_initialization.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
