"""Trusted local evaluation bundles; never load pickle from untrusted sources."""
import copy
import hashlib
import json
import os


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def weight_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_bundle(agent, model, dataset):
    import torch
    logger = agent.structured_logger
    if logger is None:
        return
    transform = copy.copy(dataset)
    for key in ('query_feats', 'plans', 'indexes', 'costs'):
        setattr(transform, key, None)
    manifest = dict(protocol='CUSTOM', split_id=logger.split_name,
                    fold_id=None, seed=None,
                    source='Local experiment configuration; not an official split',
                    train_queries=[n.info['query_name'] for n in agent.train_nodes],
                    test_queries=[n.info['query_name'] for n in agent.test_nodes])
    if hasattr(agent, 'locked_manifest'):
        manifest = agent.locked_manifest
    else:
        manifest['sha256'] = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    payload = dict(version=1, model=copy.deepcopy(model.model).cpu(),
                   params=agent.params, transform=transform,
                   workload_info=(agent.sim._GetPlanner().workload_info
                       if agent.sim is not None and agent.sim.IsPlanPhysicalButUseGenericOps()
                       else agent.exp.workload_info),
                   plan_featurizer=agent.exp.featurizer,
                   pos_featurizer=agent.exp.pos_featurizer,
                   query_featurizer=agent.exp.query_featurizer,
                   test_nodes=agent.test_nodes, manifest=manifest,
                   value_iter=agent.curr_value_iter,
                   training_seed=getattr(agent, 'run_seed', None),
                   upstream_commit='a0b45f2371c8c569bb90ad611f39911e5b850852',
                   local_code_commit=None,
                   run_py_sha256=file_hash(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'run.py')),
                   weight_sha256=weight_hash(model.model))
    path = os.path.join(logger.checkpoints_dir, 'evaluation.pt')
    torch.save(payload, path + '.tmp')
    os.replace(path + '.tmp', path)
    metadata = dict(checkpoint_sha256=file_hash(path),
                    weight_sha256=payload['weight_sha256'], value_iter=agent.curr_value_iter,
                    seed=payload['training_seed'], upstream_commit=payload['upstream_commit'],
                    local_code_commit=None, run_py_sha256=payload['run_py_sha256'],
                    type='evaluation only; not an exact resume checkpoint',
                    split_sha256=manifest['sha256'])
    with open(path + '.json.tmp', 'w') as handle:
        json.dump(metadata, handle, indent=2)
    os.replace(path + '.json.tmp', path + '.json')
    logger.metadata = dict(protocol=manifest['protocol'], split_id=manifest['split_id'],
                           fold_id=manifest['fold_id'], seed=getattr(agent, 'run_seed', None),
                           split_sha256=manifest['sha256'], checkpoint_sha256=file_hash(path))
    with open(os.path.join(logger.output_dir, 'split.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2)
    print('Evaluation bundle:', path)
