#!/usr/bin/env python3
"""Audit and package completed/partial logging acceptance without training or SQL."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def usage(root):
    return dict(allocated=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file()),
                logical=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()
    state = json.loads((session / 'session.json').read_text())
    if state['status'] in ('RUNNING', 'PREPARING'):
        raise ValueError('Cannot certify a running session')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if usage(session)['allocated'] >= 19 * 1024**3:
        raise ValueError('Insufficient artifact headroom for a separate package')
    shutil.copytree(str(session/'executed_source'), str(output/'source'))
    corrections = session/'code_corrections'
    if corrections.exists():
        shutil.copytree(str(corrections), str(output/'code_corrections'))
        for correction in sorted(corrections.iterdir(), key=lambda p: int(p.name)):
            for path in correction.rglob('*.py'):
                dest = output/'source'/path.relative_to(correction)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(path), str(dest))
    for name in ('session.json', 'resource_samples.jsonl', 'executed_source_sha256.json',
                  'unit_tests.log', 'pip_check.log', 'preflight.log', 'correctness_crosscheck.json',
                  'README_ACCEPTANCE.md', 'POLICY.md', 'final_unit_tests_verified.log', 'filesystem_readonly_check.log'):
        if (session/name).is_file():
            shutil.copy2(str(session/name), str(output/name))
    report = []
    all_accepted = len(state['stages']) == 2
    for stage in state['stages']:
        source = session/'accessible_artifacts'/stage['name']
        target = output/stage['name']
        if source.is_dir():
            shutil.copytree(str(source),str(target))
        else:
            target.mkdir()
        entry = dict(name=stage['name'], label=stage['label'], completed_iterations=stage.get('completed_iterations', []),
                     elapsed_seconds=stage.get('elapsed_seconds'), evidence=stage.get('fit_evidence', []), accepted=False)
        if stage['name'].startswith('A_'):
            initialization = session/'accessible_artifacts/A_fold02/input'
            if (initialization/'simulator_initialization.pt').is_file():
                dest = target/'simulator_initialization'
                shutil.copytree(str(initialization), str(dest))
                meta = json.loads((dest/'simulator_initialization.json').read_text())
                meta['bundle_path'] = 'simulator_initialization.pt'
                (dest/'simulator_initialization.json').write_text(json.dumps(meta,indent=2))
        evaluation = target/'evaluation'
        csvpath = evaluation/'balsa_results.csv'
        if csvpath.exists():
            rows = list(csv.DictReader(csvpath.open()))
            fields = list(rows[0]) if rows else []
            manifest = json.loads((evaluation/'split.json').read_text())
            expected = {q[:-4] for q in manifest['test_queries']}
            if {row['query_id'] for row in rows} != expected or len(rows) != len(expected):
                raise ValueError('Missing/duplicate test query rows: ' + stage['name'])
            bundles = list(target.glob('training/*/checkpoints/evaluation.pt'))
            if len(bundles) != 1:
                raise ValueError('Expected exactly one agent checkpoint')
            ckpt = bundles[0]
            ckpt_hash = digest(ckpt)
            for row in rows:
                if row['checkpoint_sha256'] != ckpt_hash or row['split_sha256'] != manifest['sha256']:
                    raise ValueError('CSV checkpoint/split hash mismatch')
                if row['result_source'] != 'MEASURED':
                    raise ValueError('Cache masquerading as fresh evaluation')
                for key in ('hinted_sql_path','explain_json_path'):
                    if row[key]:
                        p = Path(row[key])
                        relative = p.relative_to(source)
                        destination = target/relative
                        if not destination.is_file():
                            raise ValueError('Missing plan: ' + str(destination))
                        row[key] = str(destination.relative_to(output))
                if row['status']=='COMPLETED':
                    measured = sum(float(row[k]) for k in ('balsa_planning_ms','postgres_hinted_planning_ms','execution_ms'))
                    if abs(measured-float(row['end_to_end_ms'])) > 1e-5 or row['hint_respected'] != 'True':
                        raise ValueError('Incorrect E2E or hint check')
                elif row['status'] in ('TIMEOUT','ERROR') and row['execution_ms']:
                    raise ValueError('Uncompleted query has fabricated execution time')
                if row.get('baseline_status') != 'COMPLETED' and (row['speedup_execution'] or row['speedup_end_to_end']):
                    raise ValueError('Invalid baseline speedup')
            with csvpath.open('w', newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            baseline_fields = ['query_id','repeat_id','planning_ms','execution_ms','end_to_end_ms','status']
            with (evaluation/'postgres_baseline.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=baseline_fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(query_id=row['query_id'],repeat_id=row['repeat_id'],
                        planning_ms=row['postgres_baseline_planning_ms'],execution_ms=row['postgres_baseline_execution_ms'],
                        end_to_end_ms=row['postgres_baseline_end_to_end_ms'],status=row.get('baseline_status','UNKNOWN')))
            logs = [p.read_text() for p in sorted(target.glob('training/*/experiment.log'))]
            logs += [(evaluation/'experiment.log').read_text()]
            (target/'experiment.log').write_text('\n'.join(logs))
            evidence=stage.get('fit_evidence',[])
            evalsummary=json.loads((evaluation/'evaluation_summary.json').read_text())
            real_fit=any(e['did_fit'] and e['changed_tensors']>0 and e['global_step']>0 and e['weights_before']!=e['weights_after'] for e in evidence)
            statuses={s:sum(row['status']==s for row in rows) for s in sorted({row['status'] for row in rows})}
            coverage=sum(row['status']=='COMPLETED' and row.get('baseline_status')=='COMPLETED' for row in rows)
            entry.update(agent_checkpoint=str(ckpt.relative_to(output)), checkpoint_sha256=ckpt_hash,
                csv=str(csvpath.relative_to(output)), statuses=statuses, paired_speedup_coverage=coverage,
                expected_test_queries=len(expected), weights_unchanged=evalsummary['unchanged'],
                train_called=evalsummary['train_called'], eval_process_id=evalsummary['process_id'],
                accepted=bool(real_fit and evalsummary['unchanged'] and not evalsummary['train_called']
                    and any(row['status']=='COMPLETED' for row in rows)
                    and not any(row['status'] in ('ERROR','HINT_MISMATCH') for row in rows)))
        all_accepted = all_accepted and entry['accepted']
        report.append(entry)
    audit=dict(status='LOGGING_LOO_AND_RANDOM_ACCEPTED' if all_accepted else 'PARTIAL_LOGGING_NOT_ACCEPTED',
        protocols=report, session_status=state['status'],
        limitations=['not 33-fold benchmark','not 500 iterations','simulator convergence unproven',
            'Random integration uses one simulator epoch, full agent hyperparameters',
            'checkpoint load is initialization, not exact training resume', 'uncontrolled warm-cache CPU measurement'],
        source_image=dict(path=str(session/'session.ext2'),sha256=digest(session/'session.ext2')))
    (output/'AUDIT.json').write_text(json.dumps(audit,indent=2))
    lines=['# Nghiệm thu logging Fold 02 và Random', '', '**'+audit['status']+'**', '',
           '| Giao thức | Nhãn | Vòng hoàn tất | Kết quả test | Cặp baseline hợp lệ | Thời gian s |',
           '|---|---|---|---|---|---|']
    for p in report:
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(p['name'],p['label'],p['completed_iterations'],p.get('statuses','NOT_RUN'),p.get('paired_speedup_coverage','N/A'),p['elapsed_seconds']))
    lines += ['', '## Tài nguyên', json.dumps(state.get('peaks',{})), '',
        'Artifact của phiên được tính theo block thực cấp, gồm image preallocated 17 GiB và gói nộp; không xóa kết quả cũ.', '',
        '## Đính chính báo cáo pilot trước',
        'Không có bằng chứng rằng thêm 3 epoch chắc chắn kích hoạt early stopping hay simulator hội tụ. Các dự toán 15–25 epoch, thời gian agent/vòng và storage dài hạn trong báo cáo trước chỉ là giả định chưa kiểm chứng.',
        'Chỉ optimizer steps đọc trực tiếp từ fit evidence/checkpoint được xác nhận; progress bar không tự chứng minh optimizer step đã hoàn tất.', '',
        'Chi tiết checkpoint, seed, hash, coverage và phần còn thiếu: AUDIT.json, session.json và metadata từng run. Không kết luận Balsa tăng tốc từ pipeline acceptance.']
    (output/'HANDOFF.md').write_text('\n'.join(lines)+'\n')
    checks={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    (output/'artifact_manifest.json').write_text(json.dumps(checks,indent=2))
    for relative,value in checks.items():
        if digest(output/relative)!=value:
            raise ValueError('Packaged checksum verification failed')
    print(json.dumps(audit,indent=2))
    print('Session usage:',usage(session),'Package usage:',usage(output))
    if usage(session)['allocated'] + usage(output)['allocated'] > 20 * 1024**3:
        raise ValueError('Combined new artifact allocation exceeds session budget')


if __name__=='__main__':
    main()
