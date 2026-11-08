"""Trusted, hash-bound simulator initialization; not training-state resume."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


def restore_simulator(params, metadata_path):
    import torch
    import sim
    from balsa.evaluation_artifacts import file_hash, weight_hash
    with open(metadata_path) as handle:
        meta = json.load(handle)
    if meta['split_sha256'] != os.environ.get('BALSA_SPLIT_SHA256'):
        raise ValueError('Simulator bundle split mismatch')
    if not os.path.isabs(meta['bundle_path']):
        meta['bundle_path'] = str(Path(metadata_path).resolve().parent / meta['bundle_path'])
    if file_hash(meta['bundle_path']) != meta['bundle_sha256']:
        raise ValueError('Simulator bundle hash mismatch')
    bundle = torch.load(meta['bundle_path'], map_location='cpu')
    obj = object.__new__(sim.Sim)
    obj.params = params.Copy()
    obj.training_workload_info = bundle['workload_info']
    obj.query_featurizer = bundle['query_featurizer']
    obj.model = bundle['model'].eval()
    if weight_hash(obj.model) != meta['weight_sha256']:
        raise ValueError('Restored simulator weight mismatch')
    obj.train_dataset = SimpleNamespace(dataset=bundle['transform'])
    obj.train_loader = obj.val_loader = None
    obj.simulation_data = None
    obj.planner = None
    obj.model.freeze()
    obj.fit_evidence = dict(status='VERIFIED_PARTIAL_SIMULATOR_INITIALIZATION_NOT_RESUME', **meta)
    print('RESTORED_SIMULATOR:', json.dumps(meta))
    return obj
