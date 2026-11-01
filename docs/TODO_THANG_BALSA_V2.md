# TODO_THANG_BALSA_V2

## 1. Mục tiêu

Giữ pipeline Balsa hiện tại, nhưng chạy lại đúng protocol chung của đồ án để kết quả có thể so sánh trực tiếp với:

```text
Bao
PostgreSQL
Novelty Score
```

Pipeline hiện tại đã chứng minh được:

```text
Balsa train
→ weights thay đổi
→ checkpoint được tạo
→ evaluator reload checkpoint
→ evaluate độc lập
```

Phần cần sửa chủ yếu là **split data và common output format**.

---

# 2. SINGLE SOURCE OF TRUTH — BẮT BUỘC

Balsa phải sử dụng:

```text
analysis/data/split_master.csv
```

làm nguồn Train/Test membership duy nhất.

Không được tự generate lại `Random-1`.

Không được lấy split native của Balsa thay thế.

Không dùng:

```text
Random-52
LOO-template
```

cho kết quả chính thức của đồ án.

Các run này có thể giữ lại làm evidence/debug.

---

# 3. Ba split chính thức

Balsa phải chạy đúng:

```text
LOO-1
Random-1
BaseQuery-1
```

Expected:

```text
LOO-1
Train = 80
Test  = 33
```

```text
Random-1
Train = 91
Test  = 22
```

```text
BaseQuery-1
Train = 94
Test  = 19
```

Tổng mỗi split:

```text
113 JOB queries
```

---

# 4. Validate trước khi train

Trước mỗi run:

```text
Train ∩ Test = ∅
```

và:

```text
Train ∪ Test = 113 queries
```

Nếu count không đúng:

```text
STOP RUN
```

Không tiếp tục training.

---

# 5. Không dùng Random-52 cho Random-1

Hiện pipeline có:

```text
Random-52
Train = 94
Test = 19
```

Đây là split từ upstream Balsa.

Nó vẫn có giá trị cho Balsa original experiment, nhưng KHÔNG phải `Random-1` của đồ án.

Kết quả `Random-52` không được đổi tên thành `Random-1`.

---

# 6. Không dùng LOO-template cho LOO-1

Hiện có:

```text
LOO-template Fold 02
Train = 109
Test = 4
```

Đây không phải:

```text
LOO-1
Train = 80
Test = 33
```

Run `LOO-template` chỉ giữ làm technical evidence.

---

# 7. BaseQuery-1

Manifest BaseQuery-1 hiện đã có đúng membership.

Tuy nhiên vẫn phải validate lại bằng:

```text
analysis/data/split_master.csv
```

Không hard-code một danh sách riêng mà không kiểm tra với source chung.

---

# 8. Manifest

Nếu pipeline cần manifest riêng, có thể tạo:

```text
balsa/source/manifests/loo1.json
balsa/source/manifests/random1.json
balsa/source/manifests/basequery1.json
```

Nhưng các manifest này phải được:

```text
GENERATE FROM
hoặc
VALIDATE AGAINST

analysis/data/split_master.csv
```

`split_master.csv` luôn là nguồn chuẩn cao nhất.

---

# 9. Training

Cần train riêng:

```text
Balsa_LOO
← Train_LOO-1
```

```text
Balsa_RANDOM
← Train_RANDOM-1
```

```text
Balsa_BASE
← Train_BASEQuery-1
```

Mỗi split phải tạo checkpoint riêng.

Không reuse checkpoint giữa các split.

---

# 10. Checkpoint provenance

Nếu checkpoint được pipeline của Thắng train thật:

```text
model_source = team_trained
```

Bằng chứng hiện tại như:

```text
fit_evidence_*.json
weights_before
weights_after
optimizer_steps
```

nên tiếp tục giữ.

Ví dụ evidence hợp lệ:

```text
weights_before != weights_after
optimizer_steps > 0
```

---

# 11. Independent evaluation

Sau khi training:

```text
checkpoint
→ evaluator mới
→ load checkpoint
→ evaluate Test
```

Evaluator không được gọi training lại.

Giữ evidence:

```text
train_called = false
```

nếu pipeline hiện tại đã có.

---

# 12. Tổng evaluation

Kết quả cuối:

```text
LOO-1       = 33 rows
Random-1    = 22 rows
BaseQuery-1 = 19 rows
```

Tổng:

```text
74 test observations
```

trước khi phân loại:

```text
OK
TIMEOUT
ERROR
```

---

# 13. Common output structure

Mỗi split phải export:

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

---

# 14. `run_manifest.json`

Fields:

```text
run_id
model
split_type
split_id
model_source
train_query_count
test_query_count
seed
git_repo
git_commit
training_command
postgres_version
start_time
end_time
checkpoint_path
status
```

Ví dụ:

```json
{
  "model": "Balsa",
  "split_type": "LOO",
  "split_id": 1,
  "model_source": "team_trained",
  "train_query_count": 80,
  "test_query_count": 33
}
```

Có thể thêm:

```text
split_source = analysis/data/split_master.csv
```

Khuyến nghị thêm field này.

---

# 15. `split_snapshot.csv`

Schema:

```text
model
split_type
split_id
query_id
family
role
sql_file
```

File phải được tạo từ:

```text
analysis/data/split_master.csv
```

hoặc validate 100% against file đó.

Mỗi split:

```text
113 rows
```

---

# 16. `training_trace.csv`

Có thể convert từ:

```text
phases.jsonl
fit_evidence_*.json
training/iterations/*.json
```

Schema:

```text
run_id
model
split_type
split_id
phase
step
current
total
metric_name
metric_value
metric_unit
status
message
timestamp
```

Không fake epoch/loss.

Dùng đúng metric Balsa thực sự có.

---

# 17. `checkpoint_manifest.csv`

Schema:

```text
run_id
model
split_type
split_id
checkpoint_id
step
metric_name
metric_value
path
is_final
status
```

Path phải trỏ đúng checkpoint thật trong repo.

---

# 18. Timing mapping

Pipeline Balsa hiện đo:

```text
balsa_planning_ms
postgres_hinted_planning_ms
execution_ms
```

Khi export common schema:

```text
inference_ms = balsa_planning_ms
```

```text
planning_ms = postgres_hinted_planning_ms
```

```text
execution_ms = execution_ms
```

```text
end_to_end_ms =
balsa_planning_ms
+ postgres_hinted_planning_ms
+ execution_ms
```

Có thể giữ thêm raw columns để debug.

---

# 19. `evaluation_results.csv`

Schema:

```text
run_id
model
split_type
split_id
query_id
family
checkpoint_id
planning_ms
inference_ms
execution_ms
end_to_end_ms
status
timeout_ms
chosen_plan_id
error
```

---

# 20. Timeout

Nếu timeout:

```text
status = TIMEOUT
timeout_ms = threshold
```

Timing của query không hoàn tất phải để trống.

Không dùng:

```text
execution_ms = 180000
```

chỉ vì threshold là 180000 ms.

---

# 21. PostgreSQL baseline

Export:

```text
postgres_results.csv
```

Schema:

```text
split_type
split_id
query_id
family
planning_ms
execution_ms
end_to_end_ms
status
timeout_ms
```

PostgreSQL phải evaluate đúng cùng Test set với Balsa.

---

# 22. Giữ evidence hiện tại

Không xóa:

```text
Random-52
LOO-template
fit_evidence_*.json
phases.jsonl
training/iterations/*.json
```

Các file này vẫn hữu ích để chứng minh pipeline Balsa hoạt động.

Chỉ không sử dụng `Random-52` và `LOO-template` làm kết quả chính của RQ2.

---

# 23. Điều kiện hoàn thành

```text
✅ split source = analysis/data/split_master.csv

✅ LOO-1 = 80/33
✅ Random-1 = 91/22
✅ BaseQuery-1 = 94/19

✅ 3 independent training runs
✅ 3 corresponding checkpoints

✅ 33 + 22 + 19 = 74 Test observations

✅ run_manifest.json
✅ split_snapshot.csv
✅ training_trace.csv
✅ checkpoint_manifest.csv
✅ evaluation_results.csv
✅ postgres_results.csv
✅ raw_training.log

✅ checkpoint provenance rõ
✅ timeout đúng
✅ timing đúng
✅ ready to merge novelty.csv
```

Sau đó Balsa mới được xem là ready cho RQ2 chính thức.