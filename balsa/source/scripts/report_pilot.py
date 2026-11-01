#!/usr/bin/env python3
"""Report only persisted observations from a finished bounded pilot (no ML imports)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    summary = json.loads((run / 'resource_summary.json').read_text())
    if summary['status'] == 'RUNNING':
        raise RuntimeError('Pilot still running; no completion report generated')
    artifacts = run / 'accessible_artifacts'
    iterations = summary.get('completed_value_iterations', [])
    timings = summary.get('phase_timings', [])
    lines = ['# Pilot LOO-template fold 02, seed 0', '',
             '**Trạng thái: {}**'.format(summary['status']),
             'Vòng hoàn tất: {} / tối đa 3 (iteration ID {}).'.format(len(iterations), iterations),
             'Wall-clock pipeline: {:.3f} giây / 7200 giây.'.format(summary['elapsed_seconds']), '',
             '## Cấu hình và điều kiện đo',
             'Run mới, simulator mới; full 100 epoch, beam 20, 10 complete plans, simulator skip >=12 relation; chỉ giảm value iteration xuống 3.',
             'PG được duyệt: shared_buffers=4GiB, work_mem=64MiB, max_parallel_workers_per_gather=2; baseline mới, không so sánh trực tiếp môi trường smoke cũ.',
             'Cgroup {:.0f}GiB cho cả ML/Ray/eval/FUSE, soft guard {:.0f}GiB; swap=0, object store Ray=2GiB. Filesystem FUSE/ext2 có thể ảnh hưởng timing.'.format(
                 summary['ml_memory_max_bytes'] / 1024**3,
                 summary['runtime_constraints']['graceful_stop_memory_bytes'] / 1024**3), '',
             '## Tài nguyên',
             '- Peak cgroup ML (counter kernel): {:.3f} GiB.'.format(summary['ml_peak_bytes'] / 1024**3),
             '- RAM khả dụng máy thấp nhất được lấy mẫu: {:.3f} GiB.'.format(summary['machine_minimum_available_bytes'] / 1024**3),
             '- PostgreSQL PSS peak được lấy mẫu: {:.3f} GiB.'.format(summary.get('postgres_sampled_peak_pss_bytes', 0) / 1024**3),
             '- PG allocated growth peak được lấy mẫu: {:.3f} GiB.'.format(summary['postgres_maximum_growth_bytes'] / 1024**3),
             '- PG live temp peak được lấy mẫu: {:.3f} GiB.'.format(summary['postgres_live_temp_peak_bytes'] / 1024**3),
             '- Artifact logical bytes (image + export + telemetry): {}.'.format(summary['outer_artifacts']['logical']),
             '- Artifact SSD blocks được cấp: {} byte.'.format(summary['outer_artifacts']['allocated']),
             '- memory.events: `{}`.'.format(summary.get('memory_events', '').replace('\n', '; ')),
             '- Sampling: ML/machine/filesystem khoảng 1 giây; PG directory/temp/PSS mỗi 5 mẫu. Peak PG ngắn hơn có thể không quan sát được.', '',
             '## Kết thúc và độc lập đánh giá',
             'Train service: `{}`.'.format(json.dumps(summary.get('train_service', {}))),
             'Eval service: `{}`.'.format(json.dumps(summary.get('eval_service', {}))), '',
             '## Giai đoạn đã đóng trong telemetry',
             '| Stage | Iteration/sample | Thời gian s | Hoàn tất |',
             '|---|---:|---:|---|']
    components = summary.get('ml_memory_component_sampled_peaks_bytes')
    if components:
        lines.insert(lines.index('## Kết thúc và độc lập đánh giá'),
                     'RAM components (sampled peaks, **không cộng các peak không đồng thời**): `{}`. file bao gồm shmem.'.format(json.dumps(components)))
    for event in timings:
        lines.append('| {} | {} | {} | {} |'.format(event['stage'],
            event.get('value_iter', event.get('sample', 'N/A')),
            '{:.3f}'.format(event['seconds']) if 'seconds' in event else 'N/A',
            event.get('completed')))
    lines += ['', 'Các phase chỉ START mà chưa END là bị ngắt; không giả là đã hoàn tất. Thời gian wait client không thay cho Execution Time PG.', '',
              '## Cập nhật trọng số thật', '| Iteration | did_fit | Batch | global_step | Tensor thay đổi |',
              '|---|---|---:|---:|---:|']
    for fit in summary.get('fit_evidence', []):
        lines.append('| {} | {} | {} | {} | {} |'.format(fit['value_iter'], fit['did_fit'],
            fit['training_batches'], fit['global_step'], fit['changed_tensors']))
    sim = artifacts / 'training/simulator/provenance.json'
    if sim.exists():
        provenance = json.loads(sim.read_text())
        lines.append('Simulator completed evidence: `{}`; {} simulated points.'.format(
            json.dumps(provenance.get('evidence', {})), provenance.get('num_simulated_points')))
    else:
        lines.append('Simulator chưa có provenance hoàn tất được xuất; không suy ra số optimizer step từ log trạng thái.')
    ev = artifacts / 'evaluation/evaluation_summary.json'
    if ev.exists():
        lines.append('Fresh-process evaluation summary: `{}`.'.format(ev.read_text().replace('\n', ' ')))
    else:
        lines.append('Chưa có evaluation_summary hoàn tất; không xác nhận eval/hash bất biến thành công.')
    lines += ['', '## Artifact',
              'Accessible copies: `{}`.'.format(artifacts),
              'Full retained filesystem image: `{}` (không tự xóa).'.format(run / 'artifacts.ext2'),
              'Tệp lớn hoặc không được export vẫn nằm trong image; `.tmp` không phải checkpoint hợp lệ.', '',
              '## Hiệu chỉnh dự toán 35 run/seed',
              '35 run = 33 LOO-template + 1 Random + 1 BaseQuery. Không tự chạy tiếp.',
              'Khởi tạo phải tính riêng: baseline, simulator collection, simulator training/featurization. Mỗi vòng phải tách agent fit, planning và PG execution/wait.',
              'Chỉ ngoại suy những giai đoạn đã đo xong; phase bị ngắt cho lower bound tại thời điểm dừng, không phải tốc độ ổn định.',
              'Một fold và tối đa 3 vòng chưa mô tả được tăng trưởng replay, cache hit, timeout thích nghi, validation/early stopping hoặc độ khó 32 template còn lại.',
              'Nếu simulator hoặc vòng đầu chưa hoàn tất trong ngân sách, chưa có căn cứ số học đáng tin cậy cho tổng giờ của 35×500 vòng. Không nhân wall-clock bị cắt cụt với 35×500.',
              'Đây là pilot vận hành/chi phí, không dùng 3 vòng để kết luận cuối cùng về hiệu năng Balsa.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
