# Balsa V2 — BaseQuery-1: 10 value iterations

## Kết luận

Run mới được thực hiện trong một training duy nhất, seed 0, split BaseQuery-1. Các checkpoint ở mốc 3, 5 và 10 được snapshot từ cùng tiến trình training rồi đánh giá bằng ba evaluator process độc lập. Kết quả BaseQuery-1 cũ được giữ nguyên.

## Kết quả theo mốc

| Iteration | Evaluation | Completed | Median Balsa E2E (ms) | Median PostgreSQL baseline E2E (ms) | Median baseline/Balsa E2E | SHA-256 checkpoint |
|---:|---|---:|---:|---:|---:|---|
| 3 | PASS | 19/19 | 3989.528 | 239.535 | 0.1100x | `0e26eb8b4283a111f06d3aa7ffbaeaca06981c7e74a7f3bd99fd7074b803edcb` |
| 5 | PASS | 19/19 | 3686.589 | 238.553 | 0.1024x | `b98bcb549b61927e76b863a6b6340a6b11eb6f6df6b464a355949da9f0837cfe` |
| 10 | PASS | 19/19 | 3316.508 | 239.397 | 0.1073x | `374e25f14402a59504960a158fd84a3e6334ac3a0cc80ed5b426b291c4b66577` |

Từng evaluation có `errors=0`, `unchanged=true`, `train_called=false`; mỗi evaluation có đủ 19 query rows và không có timeout/failed rows. “Balsa E2E” giữ contract V2: Balsa planning + hinted PostgreSQL planning + execution. Tỷ lệ nhỏ hơn 1 nghĩa là baseline end-to-end thấp hơn Balsa cho median của workload này.

## Training và provenance

- 10/10 value iterations hoàn tất (index trong log: 0–9); một training run, không resume và không tái sử dụng checkpoint, replay hay simulator từ run trước.
- Iteration 0 là expert initialization (fit được skip); iterations 1–9 đều train với 100 optimizer steps, tổng cộng 900 optimizer steps. Evidence theo iteration nằm ở `training/9ry1hl0h/fit_evidence_*.json`.
- Simulator mới, split-local, train-only; 94 train queries / 19 test queries. PostgreSQL 12.5, database `imdbload`, role `wings`, socket `/tmp`.
- Training code commit: `f596e88129fc1d92b74240f0dbf6e6192da74545`; split manifest SHA-256: `6cc7dadb036e85db609ebda0a25931f89c4e838101ea381245a19a218a4b94a7`.
- Mỗi checkpoint bundle và JSON sidecar được giữ tại `milestones/iteration-{3,5,10}/`; evaluation rows, plans, config, logs và prediction artifacts tại `evaluations/iteration-{3,5,10}/`.

## Resource guards

- Elapsed: 4144.8s / 7200s; ML peak: 15.00 GiB / 16 GiB cgroup cap.
- Minimum MemAvailable: 9.75 GiB (minimum guard 3 GiB). PostgreSQL growth peak: 7.68 GiB (<20 GiB); live temp peak: 7.73 GiB (<16 GiB).
- New allocated artifacts: 5.69 GiB (<20 GiB). OOM events: ['max 0', 'oom 0', 'oom_kill 0'].

### Ghi chú về supervisor metadata

`resource_summary.json` được giữ nguyên như evidence thô. Trường `configuration.val_iters=3` bị hardcode cũ trong supervisor; lệnh train lưu trong `training/run_provenance.json`, config và 10 completed iteration events xác nhận yêu cầu thực tế là 10. Summary status là `COMPLETED_10_ITERATIONS_AND_3_MILESTONE_EVALS`, nhưng wrapper trả mã 1 do điều kiện return vẫn so với literal status cũ của pilot 3 iterations. Chi tiết được ghi ở `resource_summary_errata.json`.

## Artifact index

- `run_manifest.json`: metadata/provenance và checkpoint hashes.
- `milestone_comparison.csv`: kết quả query-level theo từng checkpoint.
- `novelty_join.csv`: join novelty ↔ evaluation tại mỗi mốc; join key `(query_id, family, split_type, split_id)`.
- `resource_summary.json`, `phases.jsonl`, `connections.jsonl`, raw logs và training evidence.
- `evaluations/iteration-{3,5,10}/balsa_results.csv`: kết quả evaluator đầy đủ cho mỗi milestone.
- `milestones/iteration-{3,5,10}/evaluation.pt`: bundles độc lập có SHA-256 sidecar.
