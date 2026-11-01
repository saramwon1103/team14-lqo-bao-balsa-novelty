# TODO_TUAN_BAO_V2

## 1. Mục tiêu

Hoàn thiện phần Bao để có thể merge trực tiếp với:

```text
analysis/data/novelty.csv
```

và sử dụng cho phân tích:

```text
Novelty
↔
Bao Performance
↔
PostgreSQL Performance
```

Phần Bao hiện tại đã có kết quả evaluation đủ 3 split. V2 chủ yếu yêu cầu chuẩn hóa provenance, checkpoint metadata và output.

---

# 2. SINGLE SOURCE OF TRUTH — SPLIT DATA

Từ thời điểm này, Bao phải xem file:

```text
analysis/data/split_master.csv
```

là nguồn Train/Test membership duy nhất của nhóm.

Không tự sinh lại split.

Không lấy một split khác chỉ vì có sẵn trong Bao/upstream.

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

Trước khi training/evaluation phải validate:

```text
Train ∩ Test = ∅
Train ∪ Test = 113 JOB queries
```

Bao hiện tại đã khớp 3 split này, cần tiếp tục giữ nguyên.

---

# 3. Trạng thái hiện tại

Phần Bao đã đạt:

```text
✅ LOO-1 đúng 80/33
✅ Random-1 đúng 91/22
✅ BaseQuery-1 đúng 94/19

✅ đủ 74 test observations
✅ có PostgreSQL baseline
✅ timeout handling đúng
✅ End-to-End timing hợp lệ
✅ provenance ghi author_checkpoint
```

Hiện tại checkpoint dùng để evaluation là:

```text
model_source = author_checkpoint
```

Không đổi thành:

```text
team_trained
```

nếu team chưa thực sự train Bao hoàn chỉnh.

---

# 4. Sửa checkpoint metadata

Checkpoint artifact hiện được lưu trong repo nhóm tại:

```text
bao/checkpoints/
├── bao__job__leave_one_out_split_1
├── bao__job__random_split_1
└── bao__job__base_query_split_1
```

Metadata phải trỏ đúng vào artifact thực tế này.

Không nên chỉ ghi path kiểu:

```text
bao/bao_server/...
```

nếu đó không phải vị trí artifact hiện tại trong repo.

Đề nghị trong `run_manifest.json` ghi:

```json
{
  "model_source": "author_checkpoint",
  "source_checkpoint_path": "...",
  "artifact_checkpoint_path": "bao/checkpoints/bao__job__leave_one_out_split_1"
}
```

Tương tự cho Random và BaseQuery.

---

# 5. File cần sửa

Kiểm tra và cập nhật:

```text
bao/results/LOO-1/<run_id>/run_manifest.json
bao/results/Random-1/<run_id>/run_manifest.json
bao/results/BaseQuery-1/<run_id>/run_manifest.json
```

và:

```text
checkpoint_manifest.csv
```

`checkpoint_manifest.csv` phải chứa path đúng tới checkpoint trong repo nhóm.

---

# 6. Training provenance

Nếu `training_trace.csv` được reconstruct từ log của tác giả thì phải ghi rõ:

```text
training_trace_source = author_log
```

Có thể thêm vào `run_manifest.json`:

```json
{
  "model_source": "author_checkpoint",
  "training_trace_source": "author_log"
}
```

Không để UI/report hiểu rằng đây là training trace do team vừa chạy.

---

# 7. `hint_changed_between_runs`

Hiện không nên có trường hợp:

```text
status = OK
error = hint_changed_between_runs
```

Nếu query vẫn chạy thành công thì `error` phải trống.

Đề nghị thêm:

```text
warning
```

và lưu:

```text
status = OK
error =
warning = hint_changed_between_runs
```

---

# 8. `chosen_plan_id`

Nếu nhiều evaluation runs chọn các hint khác nhau, không nên lấy plan của một run rồi gắn cho median timing của tất cả runs.

Ví dụ:

```text
run 1 → arm_0
run 2 → arm_4
run 3 → arm_0
```

Nên lưu:

```text
chosen_plan_id = MULTIPLE
```

và thêm:

```text
chosen_plan_ids = arm_0|arm_4|arm_0
```

Nếu cả các run dùng cùng plan thì có thể lưu plan đó bình thường.

---

# 9. Output folder cuối cùng

Mỗi split giữ structure:

```text
bao/results/
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

# 10. `split_snapshot.csv`

Bao không tự tạo split mới.

File này phải được sinh hoặc validate dựa trên:

```text
analysis/data/split_master.csv
```

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

Expected mỗi split:

```text
113 rows
```

---

# 11. `evaluation_results.csv`

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

Có thể thêm:

```text
warning
chosen_plan_ids
```

Nếu Bao inference đã nằm trong planning thì:

```text
inference_ms =
```

và không cộng inference lần thứ hai.

---

# 12. Timeout

Nếu timeout:

```text
status = TIMEOUT
timeout_ms = threshold
```

và:

```text
planning_ms =
execution_ms =
end_to_end_ms =
```

Không ghi:

```text
execution_ms = timeout threshold
```

---

# 13. Điều kiện Bao V2 hoàn thành

```text
✅ split membership lấy từ split_master.csv
✅ LOO-1 = 80/33
✅ Random-1 = 91/22
✅ BaseQuery-1 = 94/19

✅ 74 evaluation rows
✅ PostgreSQL baseline

✅ checkpoint provenance rõ
✅ artifact checkpoint path đúng
✅ training trace source rõ
✅ warning không nằm trong error
✅ chosen_plan_id xử lý đúng khi hint thay đổi

✅ output có thể merge trực tiếp với novelty.csv
```

Sau khi hoàn thành, Bao được xem là ready cho RQ2.