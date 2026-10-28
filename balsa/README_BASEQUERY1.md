# Balsa chính thức với log BaseQuery-1

Bộ mã này là bản clone của kho Balsa chính thức tại commit
`a0b45f2371c8c569bb90ad611f39911e5b850852`. Phần mở rộng chỉ bổ sung cấu hình
BaseQuery-1, kiểm tra môi trường và cơ chế ghi log. Thuật toán lập kế hoạch,
huấn luyện và thực thi truy vấn vẫn dùng mã Balsa gốc.

## Điểm cần hiểu trước khi chạy

- Tên mô hình trong log là `Balsa`, không phải `Bao`.
- BaseQuery-1 có 19 truy vấn kiểm thử và 94 truy vấn huấn luyện.
- `q16b` thuộc tập huấn luyện.
- `q17a` cũng thuộc tập huấn luyện của BaseQuery-1. Vì vậy log đánh giá thật sẽ
  dùng `q2a`, `q2b`, ... theo danh sách trong `experiments.py`, không ép `q17a`
  thành truy vấn kiểm thử.
- BaseQuery-1 là split tùy chỉnh của bài tập này. Kết quả chính thức trong bài
  báo Balsa dùng `Balsa_JOBRandSplit` hoặc `Balsa_JOBSlowSplit`.

## 1. Chuẩn bị môi trường tương thích

Balsa gốc dùng Python 3.7 và các phiên bản thư viện cũ được ghim trong
`requirements.txt`. Cách ít xung đột nhất là tạo môi trường Conda riêng:

```bash
conda create -n balsa python=3.7 -y
conda activate balsa
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e pg_executor
```

Việc tái lập chính xác cần PostgreSQL 12.5, `pg_hint_plan` 1.3.7 và dữ liệu JOB.
Các lệnh cài đặt đầy đủ vẫn có trong `README.md` gốc. Sau khi nạp dữ liệu, kiểm
tra tối thiểu:

```bash
python scripts/preflight_balsa.py --db imdbload
```

Kiểm tra chỉ đọc sẽ xác nhận phiên bản Python, thư viện, 113 tệp SQL, kết nối
PostgreSQL, 21 bảng JOB, 2.528.312 dòng trong bảng `title` và `pg_hint_plan`.

Nếu PostgreSQL dùng host, cổng hoặc người dùng riêng, có thể đặt các biến chuẩn:

```bash
export PGHOST=127.0.0.1
export PGPORT=5432
export PGUSER=postgres
export BALSA_DB=imdbload
```

Không ghi mật khẩu vào script. Dùng `~/.pgpass`, `PGPASSFILE` hoặc cơ chế xác
thực do PostgreSQL cung cấp.

## 2. Chạy kiểm tra nhanh

Chế độ này chỉ kiểm tra toàn bộ pipeline với sáu truy vấn, một vòng lặp và một
epoch. Số liệu của nó không phải kết quả benchmark:

```bash
bash scripts/run_basequery1.sh smoke
```

## 3. Chạy đầy đủ BaseQuery-1

Lệnh `full` tự chạy PostgreSQL baseline nếu chưa có dữ liệu expert, sau đó chạy
Balsa 500 vòng lặp theo cấu hình gốc:

```bash
WANDB_MODE=offline bash scripts/run_basequery1.sh full
```

Có thể tách hai giai đoạn:

```bash
bash scripts/run_basequery1.sh baseline
bash scripts/run_basequery1.sh train
```

Để chạy split Random của bài báo Balsa thay vì BaseQuery-1:

```bash
bash scripts/run_basequery1.sh paper
```

## 4. Đầu ra

Mỗi lần chạy có một thư mục riêng theo mã run của Weights & Biases:

```text
structured_logs/<run_id>/
├── experiment.log
├── balsa_results.csv
├── postgres_baseline.csv
├── checkpoints/
│   ├── latest.pt
│   └── latest.json
└── iterations/
    ├── value_iter_0000.log
    ├── value_iter_0000.json
    └── ...
```

- `experiment.log` ghép toàn bộ vòng huấn luyện theo đúng định dạng yêu cầu.
- `iterations/*.log` là một tệp dễ đọc cho mỗi vòng huấn luyện.
- `iterations/*.json` lưu cùng dữ liệu ở dạng máy đọc được.
- `balsa_results.csv` chứa từng truy vấn kiểm thử, thời gian Balsa, thời gian
  PostgreSQL và speedup.
- `postgres_baseline.csv` chứa cost, cardinality, planning và execution thật từ
  `EXPLAIN (ANALYZE, FORMAT JSON)`.
- `raw_logs/` giữ toàn bộ stdout và stderr của chương trình gốc.
- Checkpoint chính thức vẫn được Balsa lưu định kỳ trong thư mục run của W&B với
  tên `checkpoint.pt`. Một bản mới nhất được đồng bộ sang
  `structured_logs/<run_id>/checkpoints/latest.pt`. Trường `Model/state` trong
  log trỏ tới bản này sau mỗi lần lưu.

Log được sắp theo bốn nhóm nội dung mà đề bài yêu cầu. Về trình tự chạy nội bộ,
Balsa huấn luyện trên replay buffer hiện có trước, sau đó lập kế hoạch và thực
thi các truy vấn train để bổ sung dữ liệu cho vòng kế tiếp. Vì vậy phần `[1]`
của vòng `n` mô tả dữ liệu mới thu trong vòng `n`, còn phần `[3]` mô tả lần cập
nhật mô hình đã dùng dữ liệu có sẵn đến đầu vòng đó.

Xem log đang chạy:

```bash
tail -f "$(find structured_logs -name experiment.log -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
```

## 5. Nguồn của từng con số

| Trường | Nguồn trong lần chạy thật |
|---|---|
| `Cost` | `Plan.Total Cost` của PostgreSQL |
| `Cardinality` | `Plan.Plan Rows` của PostgreSQL |
| `Execution time` | `Execution Time` trả về khi thực thi kế hoạch |
| `Balsa inference/planning` | Thời gian `planner.plan()` của Balsa |
| `PostgreSQL Planning` | `Planning Time` của kế hoạch baseline |
| `PostgreSQL Execution` | `Execution Time` của kế hoạch baseline |
| `End-to-End` | planning cộng execution |

Không có số liệu mẫu nào được chèn vào kết quả. Nếu truy vấn timeout hoặc chưa
được đánh giá, log ghi rõ `TIMEOUT`, `SKIPPED` hoặc `N/A`.

## 6. Kiểm thử phần logger

Phần logger chỉ dùng thư viện chuẩn nên có thể kiểm thử mà chưa cần cài toàn bộ
PyTorch và Ray:

```bash
python -m unittest tests/test_structured_logging.py
python -m py_compile balsa/structured_logging.py experiments.py run.py
```

Chạy đầy đủ Balsa tốn đáng kể thời gian, CPU, RAM và dung lượng PostgreSQL. Cấu
hình `conf/balsa-postgresql.conf` của nhóm tác giả giả định máy có nhiều RAM.
Hãy điều chỉnh các tham số bộ nhớ theo máy nếu chỉ chạy thử, đồng thời ghi lại
cấu hình đó khi báo cáo kết quả.
