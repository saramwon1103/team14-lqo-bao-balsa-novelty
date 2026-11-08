# Bao — Đánh giá trên JOB (LOO-1, Random-1, BaseQuery-1)

> Phụ trách: **Tuấn** · Team 14
> Mã nguồn gốc: [edualc/lqo_ml_perspective](https://github.com/edualc/lqo_ml_perspective) (VLDB 2024)
> `model_source = author_checkpoint`

Thư mục này chứa toàn bộ kết quả của **Bao** trên 3 split chính thức của Join Order Benchmark (JOB), theo schema chung của nhóm. Các file trong `results/` dùng để merge trực tiếp với `split_master.csv`, `novelty.csv`, PostgreSQL baseline và kết quả Balsa.

---

## 1. Kết quả tóm tắt

| Split | Test | Timeout Bao | Timeout PG | Speedup (chỉ query cả 2 xong) | Bao nhanh hơn | Speedup (timeout = 180 s) |
|---|---:|---|---|---:|---:|---:|
| LOO-1 | 33 | — | q26c | **1.59x** (32 query) | 12/32 | 2.24x |
| Random-1 | 22 | q25c | q17b, q25c | 0.97x (20 query) | 3/20 | 1.43x |
| BaseQuery-1 | 19 | q25a, q25c | q25a, q25c | **2.43x** (17 query) | 8/17 | 1.06x |

- **Speedup** = Σ End-to-End của PostgreSQL / Σ End-to-End của Bao. Giá trị > 1 nghĩa là Bao nhanh hơn.
- **End-to-End** = `planning_ms + execution_ms`, lấy trung vị (median) của 3 lần đo.
- Cột cuối tính mỗi query timeout là 180 s. Đây là **cận dưới**, vì thời gian thật còn lâu hơn.

**Nhận xét nhanh:**
- **BaseQuery:** Bao nhanh hơn rõ ở các query chạy xong.
- **Random:** ở query thường, Bao ngang PostgreSQL, nhưng tránh được một timeout (q17b).
- **LOO:** Bao không bị timeout nào. Tuy nhiên phần lợi tập trung ở một vài query nặng (Bao chỉ nhanh hơn ở 12/32 query).
- Mỗi split chỉ có 1 lượt chạy và khoảng 20–33 query, nên **chưa đủ để khái quát**.

---

## 2. Phương pháp và giới hạn

| Hạng mục | Thực hiện |
|---|---|
| Split | Dùng đúng split chính thức `bao/queries/job__{leave_one_out,random,base_query}_split_1`. Train ∩ Test = ∅, Train ∪ Test = 113 query |
| Huấn luyện | **Không tái lập được** trên máy nhóm (xem bên dưới), nên dùng **checkpoint của tác giả** |
| Kiểm thử Bao | **Nhóm tự đo**: nạp checkpoint, chạy `run_test_queries.py --use_bao` |
| PostgreSQL baseline | **Nhóm tự đo** trên cùng máy, cùng tập test, cùng cấu hình |
| Số lần đo | 3 lần mỗi query, lấy median |
| Timeout | `statement_timeout = 180000 ms` (3 phút) |

**Vì sao dùng checkpoint của tác giả:** nhóm đã chạy huấn luyện đúng giao thức của bài báo (500 mẫu, retrain sau mỗi 25 query, timeout 3 phút). Sau 35 phút chỉ hoàn thành 40/1500 lần chạy, có query bị timeout. Ước tính mất khoảng **22 giờ cho mỗi split** trên laptop 8 GB RAM. Log của nỗ lực này được lưu tại `logs/team_training_attempt_bq1.txt`.

**Hệ quả khi diễn giải số liệu:**
- `training_trace.csv` và mục [1]–[3] trong `logs/experiment__*.log` là **log huấn luyện của tác giả** (timestamp năm 2023, phần cứng của tác giả). Các dòng này đều được ghi chú `author log` hoặc `(author checkpoint, paper hardware)`.
- `evaluation_results.csv`, `postgres_results.csv`, cùng mục [4] và [POSTGRESQL] là **số đo thật của nhóm**.
- Checkpoint được train trên phần cứng khác. Vì vậy lựa chọn hint của Bao có thể không tối ưu cho máy của nhóm.

---

## 3. Môi trường

| Thành phần | Giá trị |
|---|---|
| Máy | Laptop 8 GB RAM, 8 nhân CPU |
| Hệ điều hành | Windows → WSL2 Ubuntu 22.04 (khoảng 3.7 GB RAM) → Docker Desktop |
| PostgreSQL | 12.5 kèm extension Bao (`pg_bao`, build với `with_llvm=no`), GEQO bật |
| Bao server | Python 3, Torch CPU, `bao_num_arms = 5` |
| Dữ liệu | IMDB/JOB nạp từ CSV gốc: `aka_name` 901,343 · `title` 2,528,311 · `cast_info` 36,244,344 |

Commit của repo gốc và phiên bản PostgreSQL chính xác được ghi trong `results/<Split>/<run_id>/run_manifest.json`.

---

## 4. Cấu trúc thư mục

```
bao/
├── README.md
├── checkpoints/                      # checkpoint Bao CỦA TÁC GIẢ (khoảng 544 KB mỗi split)
│   ├── bao__job__leave_one_out_split_1/   (nn_weights, x_transform, y_transform, channels, n)
│   ├── bao__job__random_split_1/
│   ├── bao__job__base_query_split_1/
│   └── LICENSE_BAO
├── config/                           # cấu hình môi trường đã dùng
│   ├── docker-compose.yml
│   ├── Dockerfile__pg_bao_img
│   └── *.conf                        # cấu hình PostgreSQL
├── scripts/
│   ├── test_ckpt.sh                  # test 1 split: nạp checkpoint → PostgreSQL → Bao → xuất log
│   ├── run_one.sh                    # train + test đầy đủ theo giao thức bài báo (cần máy mạnh)
│   ├── build_bao_outputs.py          # chuyển log thô sang schema chung của nhóm → results/
│   ├── make_bao_log.py               # tạo experiment log theo template
│   └── make_summary_log.py           # rút gọn experiment log (1 khối cho mỗi mục)
├── logs/
│   ├── test__{bao,postgres}__{loo1,rnd1,bq1}_ckpt.txt   # log thô khi test (3 lần/query)
│   ├── server__{loo1,rnd1,bq1}_ckpt.txt                 # log Bao server
│   ├── console_evaluation.log                           # console lúc test
│   ├── experiment__{loo1,rnd1,bq1}_ckpt.log             # log theo template (đầy đủ)
│   ├── experiment__{loo1,rnd1,bq1}_ckpt_summary.log     # log theo template (rút gọn)
│   └── team_training_attempt_bq1.txt                    # nỗ lực tự train (đã dừng)
└── results/                          # ★ DÙNG ĐỂ MERGE
    ├── ALL_evaluation_results.csv    # 74 dòng (33 + 22 + 19)
    ├── ALL_postgres_results.csv
    ├── VALIDATION.txt
    ├── *_ckpt_per_query.csv          # bảng Bao vs PostgreSQL theo từng query
    ├── LOO-1/bao_loo1_run01/
    ├── Random-1/bao_rnd1_run01/
    └── BaseQuery-1/bao_bq1_run01/
        ├── run_manifest.json
        ├── split_snapshot.csv
        ├── training_trace.csv
        ├── checkpoint_manifest.csv
        ├── evaluation_results.csv
        ├── postgres_results.csv
        └── raw_training.log
```

---

## 5. Schema trong `results/`

### `evaluation_results.csv` (Bao)
```
run_id,model,split_type,split_id,query_id,family,checkpoint_id,planning_ms,inference_ms,execution_ms,end_to_end_ms,status,timeout_ms,chosen_plan_id,error
```

### `postgres_results.csv`
```
split_type,split_id,query_id,family,planning_ms,execution_ms,end_to_end_ms,status,timeout_ms
```

### Quy ước

| Trường | Quy ước |
|---|---|
| `split_type` | `LOO` · `Random` · `BaseQuery` |
| `query_id` / `family` | `q16b` / `q16` (tên file `16b.sql`) |
| `planning_ms` | `Planning Time` khi bật Bao, **đã gồm thời gian Bao suy luận** |
| `inference_ms` | **Để trống**, vì đã nằm trong `planning_ms`. Không cộng hai lần |
| `end_to_end_ms` | `median(planning_ms) + median(execution_ms)` của 3 lần đo |
| `status` | `OK` · `TIMEOUT` · `ERROR` |
| Timeout | `status = TIMEOUT`, `timeout_ms = 180000`, các cột thời gian **để trống**, `error = timeout`. Giá trị phạt `360000` trong log thô **không** được ghi thành thời gian thực thi |
| `chosen_plan_id` | Arm (bộ hint) mà Bao chọn, xem bảng bên dưới |
| `training_trace.csv` | `metric_name = execution_time` (ms), tức reward thật của Bao. Bao không có epoch hay loss nên không ghi các trường này |

**Ánh xạ `chosen_plan_id`** (theo `pg_extension/bao_planner.h`):

| Arm | Hint |
|---|---|
| `arm_0` | (no hint): optimizer mặc định của PostgreSQL |
| `arm_1` | `SET enable_nestloop TO off` |
| `arm_2` | `SET enable_mergejoin TO off; SET enable_indexscan TO off` |
| `arm_3` | `SET enable_nestloop TO off; SET enable_mergejoin TO off; SET enable_indexscan TO off` |
| `arm_4` | `SET enable_mergejoin TO off` |

### Template experiment log
```
===== EXPERIMENT START =====
Split / Model
[1] TRAINING DATA GENERATION   Train query, Plan collected, Cost, Cardinality, Execution time
[2] QUERY & PLAN ENCODING      Query, Query/Plan encoded, Model input generated
[3] LQO TRAINING               Model, Split, Training status, Training time, Model/state
[4] LQO EVALUATION             Test query, Bao inference/planning, Execution time, End-to-End, Status
[POSTGRESQL]                   Planning, Execution, End-to-End
===== EXPERIMENT END =====
```
Cost và Cardinality ở mục [1] là giá trị ước lượng (`Total Cost`, `Plan Rows`) lấy bằng `EXPLAIN` trên database của nhóm.

---

## 6. Chạy lại

Các script ở đây **chạy bên trong repo gốc** `lqo_ml_perspective`, vì chúng dùng `run_test_queries.py`, Bao server và `pg_extension` của repo đó.

```bash
# 1. Repo gốc + môi trường Docker (xem config/ và docker/postgres_bao/README.md của repo gốc)
git clone https://github.com/edualc/lqo_ml_perspective.git && cd lqo_ml_perspective
cp <team-repo>/bao/scripts/* bao/
cd docker/postgres_bao && docker compose up -d && cd ../..
# (nạp JOB vào database imdbload, tạo venv trong container bao_server)

# 2. Test 3 split bằng checkpoint (khoảng 1.5 giờ trên máy nhóm)
docker exec bao_server bash -c "cd /app/bao && \
  bash test_ckpt.sh loo1_ckpt leave_one_out_split_1 && \
  bash test_ckpt.sh rnd1_ckpt random_split_1 && \
  bash test_ckpt.sh bq1_ckpt base_query_split_1" | tee ~/bao_ckpt.out

# 3. Xuất dữ liệu theo schema nhóm → bao/runs/
sudo python3 bao/build_bao_outputs.py --console ~/bao_ckpt.out \
  --git-commit "$(git rev-parse HEAD)" \
  --pg-version "$(docker exec -u postgres pg_bao psql -Atc 'SELECT version();')"

# 4. Log rút gọn
sudo python3 bao/make_summary_log.py bao/logs/experiment__*_ckpt.log
```

**Lưu ý:**
- `test_ckpt.sh` từ chối ghi đè log cũ. Khi muốn chạy lượt mới, hãy đổi tag, ví dụ `bq1_ckpt2`.
- Nếu muốn tự train (`model_source = team_trained`), dùng `scripts/run_one.sh <tag> <split>`. Cần máy có từ 16 GB RAM trở lên, và mất khoảng 1–2 giờ mỗi split.
- **Không commit:** `bao.db`, `pgdata/`, CSV dữ liệu IMDB, `env/`.

---

## 7. Trích dẫn

- C. Lehmann, P. Sulimov, K. Stockinger. *Is Your Learned Query Optimizer Behaving As You Expect? A Machine Learning Perspective.* VLDB 2024.
- R. Marcus et al. *Bao: Making Learned Query Optimization Practical.* SIGMOD 2021.
- V. Leis et al. *How Good Are Query Optimizers, Really?* VLDB 2015 (JOB).

Checkpoint và mã Bao thuộc về các tác giả gốc, giấy phép xem `checkpoints/LICENSE_BAO`.
