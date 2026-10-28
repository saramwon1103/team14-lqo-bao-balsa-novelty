# TODO - Thắng / Balsa

## Đã có

- Source overlay
- experiments.py
- run.py
- requirements.txt
- structured logger
- preflight script
- run_basequery1.sh
- unit tests
- SOURCE_PROVENANCE.md
- environment.txt
- README_BASEQUERY1.md

## Cần bổ sung

- [ ] balsa/checkpoint/latest.pt hoặc link + checksum
- [ ] balsa/results/balsa_results.csv
- [ ] balsa/results/postgres_baseline.csv
- [ ] balsa/logs/experiment.log
- [ ] per-query / iteration logs cần cho demo
- [ ] xác nhận nguồn BaseQuery-1
- [ ] lệnh training
- [ ] lệnh evaluation
- [ ] xác nhận upstream Balsa commit

## Mục tiêu

Người khác phải có thể:

1. Dựng upstream Balsa.
2. Áp dụng source overlay.
3. Load checkpoint.
4. Chạy evaluation.
5. Sinh lại CSV và log.
