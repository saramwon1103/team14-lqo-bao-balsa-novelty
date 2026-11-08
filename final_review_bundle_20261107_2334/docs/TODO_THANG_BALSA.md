# THẮNG TODO — BALSA TRAINING & EVALUATION

## 1. Mục tiêu

Phụ trách chạy **Balsa** trên đúng 3 split chính thức của JOB:

- `LOO-1`
- `Random-1`
- `BaseQuery-1`

Hiện tại BaseQuery-1 đã có phần setup trước đó, nhưng cần đảm bảo cuối cùng **Balsa có đủ cả ba split** theo cùng format với Bao.

Mục tiêu chính là tạo kết quả training/evaluation theo **format chung của nhóm**, để sau này merge trực tiếp với:

- `split_master.csv`
- `query_features.csv`
- `novelty.csv`
- PostgreSQL baseline
- kết quả Bao

> Hiện tại chưa cần làm giao diện Streamlit. Chỉ cần chạy đúng pipeline và xuất đúng format dữ liệu.

---

# 2. Split bắt buộc

Phải sử dụng đúng split chính thức tương ứng với benchmark.

Expected counts:

| Split | Train | Test | Total |
|---|---:|---:|---:|
| LOO-1 | 80 | 33 | 113 |
| Random-1 | 91 | 22 | 113 |
| BaseQuery-1 | 94 | 19 | 113 |

Các split phải khớp với official definitions trong repository.

Không tự tạo Random split mới.

Không thay đổi membership Train/Test.

---

# 3. Yêu cầu về checkpoint

Mỗi split phải có checkpoint riêng:

```text
Balsa_LOO
Balsa_RANDOM
Balsa_BASE
```

Phải xác định checkpoint là:

```text
team_trained
```

hay:

```text
author_checkpoint
```

Nếu train lại từ dữ liệu Train tương ứng:

```text
model_source = team_trained
```

Nếu dùng checkpoint tác giả:

```text
model_source = author_checkpoint
```

Không được gọi checkpoint tác giả là checkpoint nhóm train.

---

# 4. Cấu trúc output

Mỗi split tạo một thư mục riêng:

```text
balsa/runs/
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
balsa/runs/LOO-1/balsa_loo1_run01/
```

---

# 5. `run_manifest.json`

Format:

```json
{
  "run_id": "balsa_loo1_run01",
  "model": "Balsa",
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

Phải ghi giá trị thực tế của run.

---

# 6. `split_snapshot.csv`

Format chung:

```csv
model,split_type,split_id,query_id,family,role,sql_file
```

Ví dụ:

```csv
Balsa,LOO,1,q1a,q1,train,1a.sql
Balsa,LOO,1,q1c,q1,test,1c.sql
```

Validation:

```text
Train ∩ Test = ∅
Train ∪ Test = 113 JOB queries
```

---

# 7. `training_trace.csv`

Đây là file ghi lại quá trình training thực tế của Balsa.

Format chung với Bao:

```csv
run_id,model,split_type,split_id,phase,step,current,total,metric_name,metric_value,metric_unit,status,message,timestamp
```

Ví dụ nếu Balsa thực tế có training loss:

```csv
balsa_loo1_run01,Balsa,LOO,1,training,1,1,100,loss,1.823,,RUNNING,"training step",2026-10-28T20:01:00
balsa_loo1_run01,Balsa,LOO,1,training,2,2,100,loss,1.412,,RUNNING,"training step",2026-10-28T20:01:10
balsa_loo1_run01,Balsa,LOO,1,training,3,3,100,loss,1.105,,RUNNING,"training step",2026-10-28T20:01:20
```

## Quy tắc

- `metric_name` phải là metric thật Balsa sử dụng.
- Nếu Balsa dùng loss thì ghi loss thật.
- Nếu training được tổ chức theo iteration thay vì epoch thì dùng iteration.
- Không ép dữ liệu Balsa phải giống nội bộ Bao.
- Chỉ cần cùng schema output.
- Không giả lập metric nếu framework không có.

---

# 8. `checkpoint_manifest.csv`

Format:

```csv
run_id,model,split_type,split_id,checkpoint_id,step,metric_name,metric_value,path,is_final,status
```

Ví dụ:

```csv
balsa_loo1_run01,Balsa,LOO,1,balsa_loo1_final,100,loss,0.142,checkpoints/balsa_loo1_final.pt,true,OK
```

Checkpoint cuối phải là checkpoint thực tế dùng để evaluation.

---

# 9. `evaluation_results.csv`

Format bắt buộc giống Bao:

```csv
run_id,model,split_type,split_id,query_id,family,checkpoint_id,planning_ms,inference_ms,execution_ms,end_to_end_ms,status,timeout_ms,chosen_plan_id,error
```

Ví dụ:

```csv
balsa_loo1_run01,Balsa,LOO,1,q1c,q1,balsa_loo1_final,10.4,5.6,118.2,134.2,OK,300000,p023,
```

## Timing

Nếu Balsa đo riêng:

```text
planning_ms
inference_ms
execution_ms
```

thì ghi riêng.

Nếu inference đã được framework tính trong planning:

```text
inference_ms = blank
```

Không double count inference trong `end_to_end_ms`.

---

# 10. Timeout / Failure

Nếu query timeout:

```text
status = TIMEOUT
```

và ghi timeout threshold thực tế:

```text
timeout_ms = ...
```

Không biến timeout threshold thành runtime hoàn thành.

Ví dụ:

```csv
balsa_loo1_run01,Balsa,LOO,1,qXX,qX,balsa_loo1_final,,,,,TIMEOUT,300000,,timeout
```

Nếu crash/error:

```text
status = ERROR
```

và ghi lý do vào `error`.

---

# 11. `postgres_results.csv`

PostgreSQL baseline dùng cùng Test queries.

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

Giữ log nguyên bản của quá trình:

```text
training
checkpoint generation
evaluation
errors
timeouts
```

Không chỉnh sửa nội dung log gốc.

File log dùng để kiểm chứng khi CSV có bất thường.

---

# 13. Expected evaluation count

Balsa phải có:

```text
LOO-1       = 33 Test rows
Random-1    = 22 Test rows
BaseQuery-1 = 19 Test rows
```

Tổng:

```text
74 evaluation rows
```

trước khi phân loại trạng thái:

```text
OK
TIMEOUT
ERROR
```

---

# 14. Output cuối cần bàn giao

```text
Balsa
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

- [ ] Bổ sung LOO-1.
- [ ] Bổ sung Random-1.
- [ ] Kiểm tra lại BaseQuery-1.
- [ ] LOO đúng 80 Train / 33 Test.
- [ ] Random đúng 91 Train / 22 Test.
- [ ] BaseQuery đúng 94 Train / 19 Test.
- [ ] Train/checkpoint riêng từng split.
- [ ] Ghi rõ `team_trained` hay `author_checkpoint`.
- [ ] Có `run_manifest.json`.
- [ ] Có `split_snapshot.csv`.
- [ ] Có `training_trace.csv`.
- [ ] Có `checkpoint_manifest.csv`.
- [ ] Có `evaluation_results.csv`.
- [ ] Có `postgres_results.csv`.
- [ ] Có `raw_training.log`.
- [ ] Tổng 74 Test observations.
- [ ] Timeout được ghi bằng `status`.
- [ ] Không double-count inference.
- [ ] Output đúng schema chung với Bao.

---

# 16. Không cần làm

Thắng **không cần làm Streamlit/UI**.

Chỉ cần hoàn thiện pipeline:

```text
Official Split
→ Balsa Training
→ Checkpoint
→ Evaluation
→ PostgreSQL Baseline
→ Standardized Output Files
```

Phần giao diện, Novelty và Data Mining sẽ được tích hợp sau từ các file output này.