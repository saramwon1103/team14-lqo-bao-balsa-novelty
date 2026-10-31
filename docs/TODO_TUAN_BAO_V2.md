Tuấn ơi, mình đã review phần Bao. Nhìn chung phần evaluation đã khá đầy đủ và có thể dùng để merge với Novelty, nhưng có vài chỗ cần fix trước khi khóa kết quả.

✅ **Phần đã làm ổn**
- Đủ 3 split chính thức:
  - LOO-1: 80 Train / 33 Test
  - Random-1: 91 Train / 22 Test
  - BaseQuery-1: 94 Train / 19 Test
- Evaluation đủ tổng 74 test observations.
- Có PostgreSQL baseline.
- Timeout đang xử lý đúng, không lấy timeout threshold giả thành execution time.
- Timing E2E hiện tại ổn, không thấy double-count inference.
- Đã ghi đúng checkpoint đang dùng là `author_checkpoint`.

⚠️ **Phần cần fix**

**1. Checkpoint path trong metadata chưa khớp vị trí file checkpoint trong repo nhóm**

Hiện checkpoint thực tế đang nằm ở:

```text
bao/checkpoints/bao__job__leave_one_out_split_1
bao/checkpoints/bao__job__random_split_1
bao/checkpoints/bao__job__base_query_split_1
```

Nhưng trong metadata hiện có chỗ đang ghi path kiểu:

```text
bao/bao_server/...
```

Nhờ Tuấn sửa để metadata trỏ đúng vào file/artifact hiện đang nằm trong repo nhóm.

Cụ thể sửa trong:

```text
bao/results/LOO-1/<run_id>/run_manifest.json
bao/results/Random-1/<run_id>/run_manifest.json
bao/results/BaseQuery-1/<run_id>/run_manifest.json
```

và:

```text
checkpoint_manifest.csv
```

Mình đề nghị ghi rõ 2 field nếu muốn phân biệt nguồn gốc và vị trí hiện tại:

```json
"model_source": "author_checkpoint",
"source_checkpoint_path": "đường dẫn checkpoint gốc của tác giả",
"artifact_checkpoint_path": "bao/checkpoints/bao__job__leave_one_out_split_1"
```

Với Random/BaseQuery thì đổi path tương ứng.

Streamlit và script merge sau này sẽ dùng:

```text
artifact_checkpoint_path
```

để tìm checkpoint trong repo nhóm.

---

**2. `hint_changed_between_runs` không nên nằm trong cột `error` nếu query vẫn chạy OK**

Hiện có trường hợp:

```text
status = OK
error = hint_changed_between_runs
```

Nhờ Tuấn đổi thành:

```text
status = OK
error =
warning = hint_changed_between_runs
```

Tức là thêm cột `warning` trong `evaluation_results.csv`.

Nếu không muốn thêm cột mới thì ít nhất để `error` trống đối với query `status = OK`.

---

**3. `chosen_plan_id` cần xử lý rõ nếu 3 lần chạy Bao chọn hint khác nhau**

Hiện timing đang lấy median của nhiều run, nhưng `chosen_plan_id` có thể đang lấy hint của một run cụ thể.

Nếu ví dụ:

```text
run 1 → arm_0
run 2 → arm_4
run 3 → arm_0
```

thì nên lưu:

```text
chosen_plan_id = MULTIPLE
```

và thêm:

```text
chosen_plan_ids = arm_0|arm_4|arm_0
```

để tránh hiểu nhầm rằng median runtime tương ứng chính xác với một plan duy nhất.

---

**4. Làm rõ phần training**

Hiện 3 run đang ghi:

```text
model_source = author_checkpoint
```

=> phần này đúng.

Nhưng điều đó cũng có nghĩa là hiện tại **chưa có Bao model do team tự train hoàn chỉnh cho cả 3 split**.

Nếu team chưa train thành công thì cứ giữ nguyên:

```text
author_checkpoint
```

không cần đổi.

Training trace hiện tại nếu lấy từ log/checkpoint của tác giả thì nên ghi rõ trong manifest, ví dụ:

```json
"training_trace_source": "author_log"
```

để sau này demo không gây hiểu nhầm rằng trace đó là run training của team.

---

**5. Kiểm tra lại `raw_training.log` trước khi final**

Mỗi run nên có:

```text
bao/results/LOO-1/<run_id>/raw_training.log
bao/results/Random-1/<run_id>/raw_training.log
bao/results/BaseQuery-1/<run_id>/raw_training.log
```

Nếu log đang lấy từ author experiment log thì ghi rõ nguồn trong manifest.

---

Sau khi fix, format cuối mình cần từ Bao là:

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

Và checkpoint artifact chung của nhóm nằm ở:

```text
bao/checkpoints/
├── bao__job__leave_one_out_split_1
├── bao__job__random_split_1
└── bao__job__base_query_split_1
```

Sau khi Tuấn sửa 4–5 điểm trên thì phần Bao có thể xem là đủ sạch để merge với `novelty.csv` và làm Bao ↔ PostgreSQL ↔ Novelty.