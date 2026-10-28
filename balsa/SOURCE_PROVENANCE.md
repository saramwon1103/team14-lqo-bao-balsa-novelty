# Nguồn mã

- Kho thượng nguồn: `https://github.com/balsa-project/balsa.git`
- Commit được clone: `a0b45f2371c8c569bb90ad611f39911e5b850852`
- Ngày chuẩn bị gói: 2026-10-27
- Giấy phép thượng nguồn: Apache License 2.0, xem `LICENSE`

Các thay đổi cục bộ phục vụ thí nghiệm:

- thêm split `BaseQuery-1` và cấu hình smoke test trong `experiments.py`
- thêm logger văn bản, CSV và JSON trong `balsa/structured_logging.py`
- nối logger và checkpoint vào vòng chạy thật trong `run.py`
- thêm kiểm tra môi trường và script chạy trong `scripts/`
- thêm kiểm thử chỉ dùng thư viện chuẩn trong `tests/`
- thêm hướng dẫn tiếng Việt trong `README_BASEQUERY1.md`

Không thay thế số liệu thật bằng kết quả mô phỏng hoặc số mẫu.
