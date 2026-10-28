# TUẤN TODO — BAO TRAINING & EVALUATION

## 1. Mục tiêu

Phụ trách chạy **Bao** trên đúng 3 split chính thức của JOB:

- `LOO-1`
- `Random-1`
- `BaseQuery-1`

Mục tiêu chính là tạo ra kết quả training/evaluation theo **format chung của nhóm**, để sau này có thể merge trực tiếp với:

- `split_master.csv`
- `query_features.csv`
- `novelty.csv`
- PostgreSQL baseline
- kết quả Balsa

> Hiện tại chưa cần làm giao diện Streamlit. Chỉ cần chạy đúng pipeline và xuất đúng format dữ liệu.

---

# 2. Split bắt buộc

Sử dụng đúng official split từ repository:

```text
D:\lqo_ml_perspective\bao\queries\
```

Các split:

```text
job__leave_one_out_split_1
job__random_split_1
job__base_query_split_1
```

Expected counts:

| Split | Train | Test | Total |
|---|---:|---:|---:|
| LOO-1 | 80 | 33 | 113 |
| Random-1 | 91 | 22 | 113 |
| BaseQuery-1 | 94 | 19 | 113 |

Không tự tạo lại random split hoặc đổi seed của split.

---

# 3. Yêu cầu quan trọng về checkpoint

Phải xác định rõ checkpoint được sử dụng là:

```text
team_trained
```

hay:

```text
author_checkpoint
```

Nếu Bao được train lại bởi nhóm thì mới được ghi:

```text
model_source = team_trained
```

Nếu sử dụng checkpoint của tác giả:

```text
model_source = author_checkpoint
```

Không được ghi là nhóm train nếu thực tế chỉ load checkpoint có sẵn.

---

# 4. Cấu trúc output

Mỗi split tạo một thư mục riêng:

```text
bao/runs/
├── LOO-1/
│   └── <run_id>/
│       ├── run_manifest.json
│       ├── split_snapshot.csv
│       ├── training_trace.csv
│       ├── checkpoint_manifest.csv
│       ├── evaluation_results.csv
│       ├── postgres_results.csv
│       └── raw_training.log
│
├── Random-1/
│   └── <run_id>/
│       └── ...
│
└── BaseQuery-1/
    └── <run_id>/
        └── ...
```

Ví dụ:

```text
bao/runs/LOO-1/bao_loo1_run01/
```

---

# 5. `run_manifest.json`

Mỗi run phải có file mô tả nguồn gốc thí nghiệm.

Format:

```json
{
  "run_id": "bao_loo1_run01",
  "model": "Bao",
  "split_type": "LOO",
  "split_id": 1,

  "model_source": "team_trained",

  "train_query_count": 80,
  "test_query_count": 33,

  "seed": 42,

  "git_repo": "edualc/lqo_ml_perspective",
  "git_commit": "ACTUAL_COMMIT",

  "training_command": "ACTUAL_COMMAND_USED",

  "postgres_version": "ACTUAL_VERSION",

  "start_time": "YYYY-MM-DDTHH:MM:SS",
  "end_time": "YYYY-MM-DDTHH:MM:SS",

  "checkpoint_path": "ACTUAL_CHECKPOINT_PATH",

  "status": "completed"
}
```

Các giá trị phải là giá trị thực tế của run.

---

# 6. `split_snapshot.csv`

Format:

```csv
model,split_type,split_id,query_id,family,role,sql_file
```

Ví dụ:

```csv
Bao,LOO,1,q1a,q1,train,1a.sql
Bao,LOO,1,q1c,q1,test,1c.sql
```

File này phải chứa toàn bộ 113 query của split.

Validation:

```text
Train ∩ Test = ∅
Train ∪ Test = 113 JOB queries
```

---

# 7. `training_trace.csv`

Đây là file dùng để ghi lại **quá trình training thực tế của Bao**.

Không ép Bao phải có `epoch` hoặc `loss`.

Sử dụng format chung:

```csv
run_id,model,split_type,split_id,phase,step,current,total,metric_name,metric_value,metric_unit,status,message,timestamp
```

Ví dụ:

```csv
bao_loo1_run01,Bao,LOO,1,training,1,1,50,reward,0.421,,RUNNING,"training iteration",2026-10-28T20:01:00
bao_loo1_run01,Bao,LOO,1,training,2,2,50,reward,0.473,,RUNNING,"training iteration",2026-10-28T20:01:10
```

## Quy tắc

- `metric_name` phải là metric thực tế Bao sử dụng.
- Nếu Bao không có `loss`, không tự tạo `loss`.
- Nếu không có khái niệm epoch thì sử dụng `step` hoặc `iteration`.
- `current` là bước hiện tại.
- `total` là tổng số bước nếu biết.
- Nếu không xác định được `total`, có thể để trống.
- Không giả lập dữ liệu chỉ để file trông đẹp.

---

# 8. `checkpoint_manifest.csv`

Format:

```csv
run_id,model,split_type,split_id,checkpoint_id,step,metric_name,metric_value,path,is_final,status
```

Ví dụ:

```csv
bao_loo1_run01,Bao,LOO,1,bao_loo1_final,50,reward,0.812,checkpoints/bao_loo1_final,true,OK
```

Phải xác định được checkpoint cuối cùng dùng cho evaluation.

---

# 9. `evaluation_results.csv`

Đây là output quan trọng nhất để merge với Novelty.

Format bắt buộc:

```csv
run_id,model,split_type,split_id,query_id,family,checkpoint_id,planning_ms,inference_ms,execution_ms,end_to_end_ms,status,timeout_ms,chosen_plan_id,error
```

Ví dụ:

```csv
bao_loo1_run01,Bao,LOO,1,q1c,q1,bao_loo1_final,8.3,4.1,122.5,134.9,OK,300000,p017,
```

## Quy tắc timing

Nếu inference đã được tính bên trong planning:

```text
inference_ms = blank
```

Không cộng inference lần thứ hai vào `end_to_end_ms`.

Nếu đo riêng được:

```text
end_to_end_ms
=
planning_ms
+
inference_ms
+
execution_ms
```

theo đúng cách Bao thực tế chạy.

---

# 10. Timeout / Failure

Nếu query timeout:

```csv
status=TIMEOUT
```

và:

```csv
timeout_ms=300000
```

hoặc timeout thực tế đang cấu hình.

Không ghi:

```text
execution_ms = 300000
```

như thể query hoàn thành trong 300 giây.

Các timing không hoàn thành có thể để trống.

Ví dụ:

```csv
bao_loo1_run01,Bao,LOO,1,qXX,qX,bao_loo1_final,,,,,TIMEOUT,300000,,timeout
```

---

# 11. `postgres_results.csv`

PostgreSQL baseline cần dùng cùng query Test và cùng môi trường benchmark.

Format:

```csv
split_type,split_id,query_id,family,planning_ms,execution_ms,end_to_end_ms,status,timeout_ms
```

Ví dụ:

```csv
LOO,1,q1c,q1,1.52,155.80,157.32,OK,300000
```

---

# 12. `raw_training.log`

Giữ nguyên console/log gốc của quá trình:

```text
training
checkpoint loading/saving
evaluation
errors
timeouts
```

Không chỉnh sửa log gốc.

File này dùng để truy vết khi dữ liệu CSV có vấn đề.

---

# 13. Expected evaluation count

Mỗi split cần có đúng số Test query:

```text
LOO-1       = 33
Random-1    = 22
BaseQuery-1 = 19
```

Tổng:

```text
74 evaluation rows
```

trước khi phân loại:

```text
OK
TIMEOUT
ERROR
```

---

# 14. Output cuối cần bàn giao

Sau khi hoàn tất, Bao phải có:

```text
Bao
├── LOO-1
│   ├── training evidence
│   ├── checkpoint
│   └── 33 evaluation rows
│
├── Random-1
│   ├── training evidence
│   ├── checkpoint
│   └── 22 evaluation rows
│
└── BaseQuery-1
    ├── training evidence
    ├── checkpoint
    └── 19 evaluation rows
```

---

# 15. Checklist hoàn thành

- [ ] Dùng đúng official LOO-1.
- [ ] Dùng đúng official Random-1.
- [ ] Dùng đúng official BaseQuery-1.
- [ ] Train/checkpoint riêng cho từng split.
- [ ] Ghi rõ `team_trained` hay `author_checkpoint`.
- [ ] Có `run_manifest.json`.
- [ ] Có `split_snapshot.csv`.
- [ ] Có `training_trace.csv`.
- [ ] Có `checkpoint_manifest.csv`.
- [ ] Có `evaluation_results.csv`.
- [ ] Có `postgres_results.csv`.
- [ ] Có `raw_training.log`.
- [ ] LOO có 33 Test rows.
- [ ] Random có 22 Test rows.
- [ ] BaseQuery có 19 Test rows.
- [ ] Timeout không bị giả thành execution time.
- [ ] Timing không bị double-count inference.
- [ ] Tất cả output dùng đúng schema chung của nhóm.

---

# 16. Không cần làm

Tuấn **không cần làm Streamlit/UI**.

Chỉ cần đảm bảo:

```text
Official Split
→ Bao Training
→ Checkpoint
→ Evaluation
→ PostgreSQL Baseline
→ Standardized Output Files
```

Phần giao diện và tích hợp Novelty sẽ được thực hiện sau từ các file output này.