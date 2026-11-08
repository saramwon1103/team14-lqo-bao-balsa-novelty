"""Trusted local evaluation bundles; never load pickle from untrusted sources."""
import copy
import hashlib
import json
import os
import shutil
import subprocess


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
    try:
        code_commit = os.environ.get('BALSA_CODE_COMMIT') or subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.dirname(__file__)),
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = None
    portable_params = copy.deepcopy(agent.params)
    if hasattr(portable_params, 'query_dir'):
        portable_params.query_dir = 'balsa/source/queries/join-order-benchmark'
    for name in ('structured_log_dir','sim_cache_dir','init_experience','agent_checkpoint',
                 'sim_checkpoint','prev_replay_buffers_glob','prev_replay_buffers_glob_val'):
        if hasattr(portable_params,name): setattr(portable_params,name,None)
    portable_test_nodes = copy.deepcopy(agent.test_nodes)
    for node in portable_test_nodes:
        if isinstance(node.info, dict) and node.info.get('query_name'):
            node.info['path'] = node.info['query_name'] + '.sql'
    payload = dict(version=1, model=copy.deepcopy(model.model).cpu(),
                   params=portable_params, transform=transform,
                   workload_info=(agent.sim._GetPlanner().workload_info
                       if agent.sim is not None and agent.sim.IsPlanPhysicalButUseGenericOps()
                       else agent.exp.workload_info),
                   plan_featurizer=agent.exp.featurizer,
                   pos_featurizer=agent.exp.pos_featurizer,
                   query_featurizer=agent.exp.query_featurizer,
                    test_nodes=portable_test_nodes, manifest=manifest,
                   value_iter=agent.curr_value_iter,
                   training_seed=getattr(agent, 'run_seed', None),
                   upstream_commit='a0b45f2371c8c569bb90ad611f39911e5b850852',
                    local_code_commit=code_commit,
                    model_source=getattr(agent, 'model_source', 'team_trained'),
                    simulator_source=getattr(agent, 'simulator_source', 'unknown'),
                   run_py_sha256=file_hash(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'run.py')),
                   weight_sha256=weight_hash(model.model))
    path = os.path.join(logger.checkpoints_dir, 'evaluation.pt')
    torch.save(payload, path + '.tmp')
    os.replace(path + '.tmp', path)
    metadata = dict(checkpoint_sha256=file_hash(path),
                    weight_sha256=payload['weight_sha256'], value_iter=agent.curr_value_iter,
                     seed=payload['training_seed'], upstream_commit=payload['upstream_commit'],
                     local_code_commit=payload['local_code_commit'],
                     model_source=payload['model_source'], simulator_source=payload['simulator_source'],
                     run_py_sha256=payload['run_py_sha256'],
                    type='evaluation only; not an exact resume checkpoint',
                    split_sha256=manifest['sha256'])
    with open(path + '.json.tmp', 'w') as handle:
        json.dump(metadata, handle, indent=2)
    os.replace(path + '.json.tmp', path + '.json')
    # Preserve milestone bundles from a single training run. The regular
    # evaluation.pt remains the latest bundle for backwards compatibility.
    snapshot_root = os.environ.get('BALSA_CHECKPOINT_SNAPSHOTS')
    milestone = agent.curr_value_iter + 1
    milestones = {int(value) for value in os.environ.get(
        'BALSA_CHECKPOINT_MILESTONES', '3,5,10').split(',') if value.strip()}
    if snapshot_root and milestone in milestones:
        destination = os.path.join(snapshot_root, 'iteration-{}'.format(milestone))
        os.makedirs(destination, exist_ok=True)
        shutil.copy2(path, os.path.join(destination, 'evaluation.pt'))
        shutil.copy2(path + '.json', os.path.join(destination, 'evaluation.pt.json'))
    logger.metadata = dict(protocol=manifest['protocol'], split_id=manifest['split_id'],
                           fold_id=manifest['fold_id'], seed=getattr(agent, 'run_seed', None),
                           split_sha256=manifest['sha256'], checkpoint_sha256=file_hash(path))
    with open(os.path.join(logger.output_dir, 'split.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2)
    print('Evaluation bundle:', path)
