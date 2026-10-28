# Novelty Score & Data Mining

Phần mở rộng của Nhóm 14 nhằm phân tích mức độ mới của truy vấn kiểm thử
so với tập huấn luyện tương ứng.

## Pipeline

Official Train/Test Split
→ Trích đặc trưng SQL
→ Tables / Joins / Filters
→ Jaccard Distance
→ Nearest Training Query
→ Novelty Score
→ Ghép Bao / Balsa / PostgreSQL
→ Data Mining

## Dữ liệu đầu ra dự kiến

analysis/data/
- query_features.csv
- novelty.csv
- master_results.csv
- analysis_summary.csv

## Source

analysis/src/
- extract_query_features.py
- compute_novelty.py
- merge_results.py
- analyze_novelty.py
