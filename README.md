# Team 14 - Bao, Balsa & Query Novelty Analysis

Đồ án môn Hệ Cơ sở Dữ liệu Tiên tiến.

Repo tái lập một phần benchmark Learned Query Optimizer từ bài báo
"Is Your Learned Query Optimizer Behaving As You Expect? A Machine Learning Perspective"
và mở rộng bằng phân tích Query Novelty ở cấp từng truy vấn.

## Thành phần

- ao/: kết quả và artifact của Bao
- alsa/: source overlay và artifact của Balsa
- splits/: thông tin Train/Test split
- nalysis/: Novelty Score và Data Mining
- demo/: kịch bản demo
- docs/: schema, trạng thái và phân công bổ sung

## Pipeline

Official Train/Test Split
→ Bao / Balsa
→ Query-level Result
→ Query Novelty
→ PostgreSQL Baseline
→ Data Mining
→ Analysis

## Trạng thái hiện tại

### Bao
- BaseQuery-1: có kết quả
- Random-1: có kết quả
- LOO-1: có kết quả
- Cần bổ sung source/config/checkpoint provenance

### Balsa
- Có source overlay và hướng dẫn BaseQuery-1
- Cần bổ sung checkpoint, result và execution log của lần chạy thật

### Novelty Analysis
- Đang triển khai

## Tham khảo

Benchmark repository:
https://github.com/edualc/lqo_ml_perspective

Balsa upstream:
https://github.com/balsa-project/balsa
