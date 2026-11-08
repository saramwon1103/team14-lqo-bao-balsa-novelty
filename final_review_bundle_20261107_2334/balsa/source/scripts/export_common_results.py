#!/usr/bin/env python3
"""Export a V2 Balsa train/eval run into the team's common CSV contract."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

from split_master_adapter import EXPECTED, load_and_validate

SNAPSHOT_FIELDS = ['model', 'split_type', 'split_id', 'query_id', 'family', 'role', 'sql_file']
TRACE_FIELDS = ['run_id','model','split_type','split_id','phase','step','current','total',
                'metric_name','metric_value','metric_unit','status','message','timestamp']
CHECKPOINT_FIELDS = ['run_id','model','split_type','split_id','checkpoint_id','step',
                     'metric_name','metric_value','path','is_final','status']
EVAL_FIELDS = ['run_id','model','split_type','split_id','query_id','family','checkpoint_id',
               'planning_ms','inference_ms','execution_ms','end_to_end_ms','status','timeout_ms',
               'chosen_plan_id','error']
PG_FIELDS = ['split_type','split_id','query_id','family','planning_ms','execution_ms',
             'end_to_end_ms','status','timeout_ms']


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def read_text(path, errors='strict'):
    with Path(path).open('r',encoding='utf-8',errors=errors) as f: return f.read()


def csv_write(path, fields, rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows)


def validate_novelty_join(novelty_csv, split_type, split_id, evaluation_rows):
    with Path(novelty_csv).open(newline='',encoding='utf-8-sig') as f:
        novelty=list(csv.DictReader(f))
    keys=[(r['query_id'],r['family'],r['split_type'],str(r['split_id'])) for r in novelty]
    if len(keys)!=len(set(keys)):
        raise ValueError('novelty.csv has duplicate composite join keys')
    lookup=set(keys)
    result_keys=[(r['query_id'],r['family'],r['split_type'],str(r['split_id'])) for r in evaluation_rows]
    if len(result_keys)!=len(set(result_keys)):
        raise ValueError('Balsa evaluation has duplicate novelty join keys')
    missing=[key for key in result_keys if key not in lookup]
    if missing: raise ValueError('Balsa results missing novelty keys: '+repr(missing[:5]))
    novelty_group={key for key in lookup if key[2]==split_type and key[3]==str(split_id)}
    if set(result_keys)!=novelty_group:
        raise ValueError('Balsa test coverage differs from novelty.csv for this split')
    return dict(novelty_rows=len(novelty),unique_novelty_keys=len(lookup),
                matched_rows=len(result_keys),join_key=['query_id','family','split_type','split_id'],
                merge_cardinality='one_to_one')


def validate_exported_run(run_dir, split_type):
    root=Path(run_dir)
    expected_test=EXPECTED[split_type][1]
    def read(name, fields):
        with (root/name).open(newline='',encoding='utf-8-sig') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames!=fields: raise ValueError('{} schema mismatch'.format(name))
            return list(reader)
    snap=read('split_snapshot.csv',SNAPSHOT_FIELDS)
    train={r['query_id'] for r in snap if r['role']=='train'}
    test={r['query_id'] for r in snap if r['role']=='test'}
    if len(snap)!=113 or len(train)!=EXPECTED[split_type][0] or len(test)!=expected_test or train&test:
        raise ValueError('Exported split snapshot membership/count failure')
    evaluation=read('evaluation_results.csv',EVAL_FIELDS)
    postgres=read('postgres_results.csv',PG_FIELDS)
    eval_ids=[r['query_id'] for r in evaluation]
    pg_ids=[r['query_id'] for r in postgres]
    if len(evaluation)!=expected_test or set(eval_ids)!=test or len(eval_ids)!=len(set(eval_ids)):
        raise ValueError('Evaluation output does not have one observation per test query')
    if len(postgres)!=expected_test or set(pg_ids)!=test or len(pg_ids)!=len(set(pg_ids)):
        raise ValueError('PostgreSQL output does not cover exact test set')
    for row in evaluation:
        if row['status']=='TIMEOUT' and any(row[k] for k in ('planning_ms','inference_ms','execution_ms','end_to_end_ms')):
            raise ValueError('Timeout contains fabricated timing')
        if row['status']=='OK':
            vals=[float(row[k]) for k in ('planning_ms','inference_ms','execution_ms')]
            if abs(float(row['end_to_end_ms'])-sum(vals))>1e-6:
                raise ValueError('E2E mapping differs from V2 timing contract')
    for row in postgres:
        timing_keys=('planning_ms','execution_ms','end_to_end_ms')
        if row['status']=='OK':
            if not all(row[k] for k in timing_keys):
                raise ValueError('OK PostgreSQL row missing timing')
            if abs(float(row['end_to_end_ms'])-
                   (float(row['planning_ms'])+float(row['execution_ms'])))>1e-6:
                raise ValueError('PostgreSQL E2E differs from planning + execution')
        elif row['status'] in ('TIMEOUT','ERROR','NOT_RUN'):
            if any(row[k] for k in timing_keys):
                raise ValueError('Unmeasured/failed PostgreSQL row contains fabricated timing')
        else:
            raise ValueError('Unknown PostgreSQL status: '+row['status'])
    checkpoint=read('checkpoint_manifest.csv',CHECKPOINT_FIELDS)
    if len(checkpoint)!=1 or not (root/checkpoint[0]['path']).is_file():
        raise ValueError('Checkpoint manifest path does not resolve to artifact')
    if not (root/'raw_training.log').is_file() or not (root/'training_trace.csv').is_file():
        raise ValueError('Required common artifacts missing')
    return dict(snapshot_rows=len(snap),evaluation_rows=len(evaluation),postgres_rows=len(postgres),
                statuses={s:sum(r['status']==s for r in evaluation) for s in sorted({r['status'] for r in evaluation})})


def v2_split_name(protocol):
    if protocol not in ('LOO-1','Random-1','BaseQuery-1'):
        raise ValueError('Not a V2 official split: '+str(protocol))
    return protocol.split('-')[0]


def status_v2(value):
    return {
        'COMPLETED':'OK',
        'HINT_MISMATCH':'ERROR',
        'OK':'OK',
        'TIMEOUT':'TIMEOUT',
        'ERROR':'ERROR',
        'NOT_RUN':'NOT_RUN',
    }.get(value,'ERROR')


def baseline_status_v2(value):
    """Normalize PostgreSQL baseline status without inventing an error.

    Legacy Balsa rows left ``baseline_status`` blank when the hinted Balsa
    execution timed out before the baseline statement was reached.  Blank in
    that situation means "not measured", not "PostgreSQL error".
    """
    if value in ('', None):
        return 'NOT_RUN'
    return status_v2(value)


def query_stem(value):
    return value[1:] if value.startswith('q') else value


def extract_hint_sql(sql_text):
    """Recover the complete Balsa hint, including multiline block hints."""
    if sql_text.startswith('/*+'):
        end=sql_text.find('*/')
        if end < 0:
            raise ValueError('Hinted SQL has an unterminated Balsa hint')
        return sql_text[:end+2]
    return sql_text.splitlines()[0] if sql_text.splitlines() else ''


def export_run(training_dir, evaluation_dir, source_csv, sql_dir, output_dir,
               run_id, code_commit, pg_version=None, novelty_csv=None,
               raw_training_log=None, raw_eval_log=None):
    training_dir=Path(training_dir).resolve(); evaluation_dir=Path(evaluation_dir).resolve()
    output=Path(output_dir).resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    evaluation=json.loads((evaluation_dir/'evaluation_summary.json').read_text())
    if not evaluation.get('unchanged') or evaluation.get('train_called') is not False:
        raise ValueError('Independent evaluation evidence failed')
    bundle_paths=list(training_dir.glob('*/checkpoints/evaluation.pt'))
    if len(bundle_paths)!=1: raise ValueError('Expected one agent evaluation checkpoint')
    import torch
    bundle=torch.load(str(bundle_paths[0]),map_location='cpu')
    manifest=bundle['manifest']; protocol=manifest['protocol']; split_type=v2_split_name(protocol)
    locked=load_and_validate(source_csv,sql_dir,manifest['source']['source_commit'])
    source_split=locked['splits'][protocol]
    if source_split['manifest']['sha256']!=manifest['sha256']:
        raise ValueError('Checkpoint manifest does not equal split_master membership')
    if bundle.get('local_code_commit') != code_commit:
        raise ValueError('Checkpoint code commit differs from requested provenance')
    if bundle.get('model_source')!='team_trained' or not bundle.get('simulator_source'):
        raise ValueError('Checkpoint provenance must identify team-trained model and simulator source')
    provenance_path=training_dir/'run_provenance.json'
    if not provenance_path.is_file(): raise ValueError('V2 train provenance is missing')
    provenance=json.loads(provenance_path.read_text())
    if provenance.get('code_commit')!=code_commit or provenance.get('split_manifest_sha256')!=manifest['sha256']:
        raise ValueError('Train provenance commit/split mismatch')
    test_ids={'q'+Path(p).stem for p in manifest['test_queries']}
    train_ids={'q'+Path(p).stem for p in manifest['train_queries']}
    if train_ids & test_ids or len(train_ids|test_ids)!=113: raise ValueError('Invalid checkpoint membership')
    rows=source_split['snapshot']
    snapshot=[dict(model='Balsa',split_type=split_type,split_id='1',query_id=r['query_id'],
                   family=r['family'],role=r['role'],sql_file=r['sql_file']) for r in rows]
    if len(snapshot)!=113: raise ValueError('Split snapshot must contain 113 rows')
    csv_write(output/'split_snapshot.csv',SNAPSHOT_FIELDS,snapshot)

    checkpoint_id='balsa_'+split_type.lower()+'_'+digest(bundle_paths[0])[:16]
    artifact=output/'artifacts'/'evaluation.pt';artifact.parent.mkdir()
    shutil.copy2(str(bundle_paths[0]),str(artifact))
    shutil.copy2(str(bundle_paths[0])+'.json',str(artifact)+'.json')
    checkpoint_row=dict(run_id=run_id,model='Balsa',split_type=split_type,split_id='1',
        checkpoint_id=checkpoint_id,step=bundle.get('value_iter'),metric_name='',metric_value='',
        path='artifacts/evaluation.pt',is_final='true',status='OK')
    csv_write(output/'checkpoint_manifest.csv',CHECKPOINT_FIELDS,[checkpoint_row])

    eval_file=evaluation_dir/'balsa_results.csv'
    with eval_file.open(newline='',encoding='utf-8-sig') as f: measured=list(csv.DictReader(f))
    if len(measured)!=len(test_ids) or {('q'+query_stem(r['query_id'])) for r in measured}!=test_ids:
        raise ValueError('Evaluation coverage differs from exact split test membership')
    if len({r['query_id'] for r in measured})!=len(test_ids): raise ValueError('Duplicate evaluation observation')
    if any(r.get('run_id')!=run_id or r.get('checkpoint_sha256')!=digest(bundle_paths[0]) for r in measured):
        raise ValueError('Evaluation run/checkpoint provenance mismatch')
    by_query={query_stem(r['query_id']):r for r in measured}
    base=[]; evaluation_rows=[]
    for row in source_split['snapshot']:
        if row['role']!='test': continue
        qid=row['query_id'][1:]; raw=by_query[qid]
        status=status_v2(raw.get('status'))
        timeout=raw.get('timeout_ms') or ''
        vals={key:(float(raw[key]) if raw.get(key) not in ('',None) else None) for key in
              ('balsa_planning_ms','postgres_hinted_planning_ms','execution_ms')}
        complete=status=='OK'
        if complete and not all(v is not None for v in vals.values()): raise ValueError('OK row missing timing: '+row['query_id'])
        if not complete: vals={k:None for k in vals}
        plan_id=raw.get('chosen_plan_id')
        planpath=raw.get('hinted_sql_path')
        if status=='OK':
            if not plan_id or not planpath or not Path(planpath).is_file():
                raise ValueError('Successful query missing chosen plan artifact')
            hint=extract_hint_sql(Path(planpath).read_text())
            expected='plan_'+hashlib.sha256(hint.encode()).hexdigest()[:16]
            if plan_id!=expected: raise ValueError('chosen_plan_id does not identify hinted SQL')
        e2e=sum(vals[k] for k in vals) if complete else None
        evaluation_rows.append(dict(run_id=run_id,model='Balsa',split_type=split_type,split_id='1',
            query_id=row['query_id'],family=row['family'],checkpoint_id=checkpoint_id,
            planning_ms=vals['postgres_hinted_planning_ms'],inference_ms=vals['balsa_planning_ms'],
            execution_ms=vals['execution_ms'],end_to_end_ms=e2e,status=status,timeout_ms=timeout,
            chosen_plan_id=plan_id or '',error=raw.get('error_message','')))
        baseline_status=baseline_status_v2(raw.get('baseline_status'))
        bp=raw.get('postgres_baseline_planning_ms')
        be=raw.get('postgres_baseline_execution_ms')
        baseline_ok=baseline_status=='OK' and bp not in ('',None) and be not in ('',None)
        base.append(dict(split_type=split_type,split_id='1',query_id=row['query_id'],family=row['family'],
            planning_ms=float(bp) if baseline_ok else '',execution_ms=float(be) if baseline_ok else '',
            end_to_end_ms=float(bp)+float(be) if baseline_ok else '',status='OK' if baseline_ok else baseline_status,
            timeout_ms=timeout))
    csv_write(output/'evaluation_results.csv',EVAL_FIELDS,evaluation_rows)
    csv_write(output/'postgres_results.csv',PG_FIELDS,base)
    if len(evaluation_rows)!=EXPECTED[split_type][1] or len(base)!=len(evaluation_rows):
        raise ValueError('V2 test output count mismatch')
    novelty_path=Path(novelty_csv) if novelty_csv else Path(source_csv).resolve().parent/'novelty.csv'
    novelty_audit=validate_novelty_join(novelty_path,split_type,1,evaluation_rows) if novelty_path.is_file() else None

    trace=[]; step=0
    phasefile=training_dir.parent/'phases.jsonl'
    if not phasefile.exists(): phasefile=training_dir/'phases.jsonl'
    if phasefile.exists():
        for line in phasefile.read_text().splitlines():
            event=json.loads(line); step+=1
            trace.append(dict(run_id=run_id,model='Balsa',split_type=split_type,split_id='1',
                phase=event.get('stage'),step=step,current=event.get('value_iter',''),total='',
                metric_name='duration',metric_value=event.get('seconds',''),metric_unit='s',
                status='OK' if event.get('completed') else event.get('event',''),
                message=event.get('event',''),timestamp=event.get('timestamp','')))
    for evidence_path in sorted(training_dir.rglob('fit_evidence_*.json')):
        evidence=json.loads(evidence_path.read_text()); iteration=evidence.get('value_iter')
        metrics=[('optimizer_steps',evidence.get('global_step',0),'steps'),
                 ('training_batches',evidence.get('training_batches',0),'batches'),
                 ('changed_tensors',evidence.get('changed_tensors',0),'tensors')]
        for name,value,unit in metrics:
            step+=1; trace.append(dict(run_id=run_id,model='Balsa',split_type=split_type,split_id='1',
                phase='agent_fit',step=step,current=iteration,total='',metric_name=name,metric_value=value,
                metric_unit=unit,status='COMPLETED' if evidence.get('did_fit') else 'SKIPPED_EXPERT_INIT',
                message='weights_before={} weights_after={}'.format(evidence.get('weights_before',''),evidence.get('weights_after','')),
                timestamp=''))
    csv_write(output/'training_trace.csv',TRACE_FIELDS,trace)

    rawlogs=[]
    for p in sorted(training_dir.rglob('experiment.log')):
        rawlogs.append('\n===== SOURCE {} =====\n'.format(p.relative_to(training_dir))+read_text(p,errors='replace'))
    for p in sorted(training_dir.parent.glob('train-*.log')):
        rawlogs.append('\n===== RAW {} =====\n'.format(p.name)+read_text(p,errors='replace'))
    if raw_training_log:
        rawlogs.append('\n===== TRAIN PROCESS STDOUT =====\n'+read_text(raw_training_log,errors='replace'))
    (output/'raw_training.log').write_text('\n'.join(rawlogs))
    raw_training=output/'raw_training.log'
    raw_training.write_text(read_text(raw_training,errors='replace').replace('/home/wings/','<machine-local>/').replace('/tmp/opencode/','<runtime>/'))
    # Keep evaluator artifacts and raw eval output under portable run-local paths.
    shutil.copytree(str(evaluation_dir),str(output/'evaluation'),ignore=shutil.ignore_patterns('__pycache__'))
    raw_eval=output/'raw_eval.log'
    if raw_eval_log:
        raw_eval.write_text(read_text(raw_eval_log,errors='replace'))
    elif (evaluation_dir/'experiment.log').is_file():
        raw_eval.write_text(read_text(evaluation_dir/'experiment.log',errors='replace'))
    else:
        raw_eval.write_text('No evaluator console log supplied; see evaluation/experiment.log and evaluation/evaluation_summary.json.\n')
    raw_eval.write_text(read_text(raw_eval,errors='replace').replace('/home/wings/','<machine-local>/').replace('/tmp/opencode/','<runtime>/'))
    copied_csv=output/'evaluation'/'balsa_results.csv'
    if copied_csv.is_file():
        with copied_csv.open(newline='',encoding='utf-8-sig') as f: raw_rows=list(csv.DictReader(f)); fields=list(raw_rows[0]) if raw_rows else []
        for raw in raw_rows:
            for field in ('hinted_sql_path','explain_json_path'):
                value=raw.get(field)
                if value:
                    path=Path(value)
                    if not path.is_absolute() or not path.is_file(): raise ValueError('Evaluation artifact path invalid: '+value)
                    raw[field]=str(Path('evaluation')/'plans'/path.name)
        csv_write(copied_csv,fields,raw_rows)
    environment_path=output/'evaluation'/'environment.json'
    if environment_path.is_file():
        env=json.loads(environment_path.read_text())
        connection=env.get('connection',[])
        if len(connection)>5:
            env['server_data_directory_recorded']=bool(connection[5])
            connection[5]='<redacted-machine-local-path>'
            env['connection']=connection
        environment_path.write_text(json.dumps(env,indent=2)+'\n')
    eval_log=output/'evaluation'/'experiment.log'
    if eval_log.is_file():
        text=read_text(eval_log,errors='replace').replace(str(evaluation_dir), 'evaluation')
        text=text.replace('/home/wings/', '<machine-local>/').replace('/tmp/opencode/', '<runtime>/')
        eval_log.write_text(text)
    config_path=output/'evaluation'/'config.txt'
    if config_path.is_file():
        text=read_text(config_path,errors='replace').replace('/home/wings/','<machine-local>/').replace('/tmp/opencode/','<runtime>/')
        config_path.write_text(text)
    for path in (output/'evaluation').rglob('*'):
        if path.is_file() and path.suffix in ('.json','.txt','.log','.csv'):
            try: text=path.read_text()
            except UnicodeDecodeError: continue
            if '/home/wings/' in text or '/tmp/opencode/' in text:
                raise ValueError('Absolute machine path leaked into exported artifact: '+str(path))
    env=json.loads((evaluation_dir/'environment.json').read_text())
    if pg_version is None:
        pg_version=env.get('connection',[None,None,'unknown'])[2]
    postgres_baseline_session=env.get('postgres_baseline_session')
    if postgres_baseline_session is None:
        postgres_baseline_session=dict(
            connection='legacy_same_session',
            note=('Evaluator artifact predates baseline isolation. PostgreSQL baseline '
                  'may inherit Balsa-only session overrides; refresh/re-evaluate before '
                  'claiming a native PostgreSQL baseline.'))
    evidence_paths=list(training_dir.rglob('fit_evidence_*.json'))
    fit_steps=[];real_update=False
    for path in evidence_paths:
        item=json.loads(path.read_text())
        if item.get('did_fit'):
            fit_steps.append(int(item.get('global_step',0)))
            real_update = real_update or (int(item.get('global_step',0))>0 and int(item.get('changed_tensors',0))>0
                and item.get('weights_before')!=item.get('weights_after'))
    if not real_update: raise ValueError('No weight-changing agent optimizer update evidenced')
    checkpoint_step=sum(fit_steps)
    if checkpoint_step <= 0:
        raise ValueError('No real agent optimizer update evidenced for this run')
    checkpoint_row['step']=checkpoint_step
    csv_write(output/'checkpoint_manifest.csv',CHECKPOINT_FIELDS,[checkpoint_row])
    phase_timestamps=[]
    if phasefile.exists():
        for line in phasefile.read_text().splitlines():
            item=json.loads(line)
            if isinstance(item.get('timestamp'),(int,float)): phase_timestamps.append(item['timestamp'])
    actual_command=' '.join(provenance.get('training_command',[])) if provenance else ''
    run_manifest=dict(run_id=run_id,model='Balsa',split_type=split_type,split_id=1,
        model_source=bundle['model_source'],
        simulator_source=bundle['simulator_source'],
        train_query_count=len(train_ids),test_query_count=len(test_ids),seed=bundle.get('training_seed'),
        git_repo='https://github.com/saramwon1103/team14-lqo-bao-balsa-novelty',
        git_commit=code_commit,training_command=actual_command.replace('/home/wings/','<machine-local>/').replace('/tmp/opencode/','<runtime>/'),
        postgres_version=pg_version,start_time=min(phase_timestamps) if phase_timestamps else '',
        end_time=max(phase_timestamps) if phase_timestamps else '',checkpoint_path='artifacts/evaluation.pt',
        checkpoint_sha256=digest(artifact),status='COMPLETED',
        split_source='analysis/data/split_master.csv',split_source_commit=manifest['source']['source_commit'],
        split_source_sha256=manifest['source']['sha256'],membership_sha256=manifest['source']['membership_sha256'],
        split_manifest_sha256=manifest['sha256'],evaluation_pid=evaluation.get('process_id'),
        train_called_during_eval=evaluation.get('train_called'),weights_unchanged=evaluation.get('unchanged'),
        training_trace_source='team_run',timing_contract='V2: inference + hinted PostgreSQL planning + execution',
        postgres_baseline_session=postgres_baseline_session,
        novelty_join_audit=novelty_audit,
        evaluation_status_counts={s:sum(r['status']==s for r in evaluation_rows) for s in sorted({r['status'] for r in evaluation_rows})},
        postgres_status_counts={s:sum(r['status']==s for r in base) for s in sorted({r['status'] for r in base})})
    if any(r['status']!='OK' for r in evaluation_rows): run_manifest['status']='COMPLETED_WITH_QUERY_ERRORS_OR_TIMEOUTS'
    (output/'run_manifest.json').write_text(json.dumps(run_manifest,indent=2)+'\n')
    run_manifest['output_validation']=validate_exported_run(output,split_type)
    (output/'run_manifest.json').write_text(json.dumps(run_manifest,indent=2)+'\n')
    return run_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-dir',type=Path,required=True);p.add_argument('--evaluation-dir',type=Path,required=True)
    p.add_argument('--split-master',type=Path,required=True);p.add_argument('--sql-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--code-commit',required=True);p.add_argument('--postgres-version')
    p.add_argument('--novelty-csv',type=Path)
    p.add_argument('--raw-training-log',type=Path);p.add_argument('--raw-eval-log',type=Path)
    a=p.parse_args()
    result=export_run(a.training_dir,a.evaluation_dir,a.split_master,a.sql_dir,a.output_dir,
                      a.run_id,a.code_commit,a.postgres_version,a.novelty_csv,
                      a.raw_training_log,a.raw_eval_log)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
