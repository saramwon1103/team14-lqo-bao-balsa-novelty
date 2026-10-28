# TODO - Thắng / Balsa

## Mục tiêu

Balsa phải được huấn luyện và đánh giá trên cùng ba cách chia dữ liệu với Bao:

- LOO-1
- Random-1
- BaseQuery-1

Hiện BaseQuery-1 đã được triển khai. Cần bổ sung LOO-1 và Random-1.

## BaseQuery-1

Xác nhận lại:

`Train_BASE → Balsa_BASE → Test_BASE`

Cần bàn giao:

- Train/Test manifest.
- Checkpoint.
- Training log.
- Evaluation log.
- `balsa_results.csv`.
- PostgreSQL baseline.
- Planning / execution / end-to-end time.
- Status của từng query.

Đồng thời xác nhận rõ BaseQuery-1 lấy từ benchmark paper/repository nào, tránh mô tả mơ hồ là custom split nếu thực tế sử dụng split của benchmark.

## LOO-1

Cần bổ sung pipeline:

`Train_LOO`

`→ Training Data Generation`

`→ Query/Plan Encoding`

`→ Train Balsa`

`→ Balsa_LOO checkpoint`

`→ Test_LOO`

`→ per-query results`

Phải sử dụng đúng Train/Test definition tương ứng với benchmark, không tự tạo một LOO mới.

## Random-1

Cần bổ sung pipeline:

`Train_RANDOM`

`→ Training Data Generation`

`→ Query/Plan Encoding`

`→ Train Balsa`

`→ Balsa_RANDOM checkpoint`

`→ Test_RANDOM`

`→ per-query results`

Không được random lại bằng seed khác nếu nhóm đã xác định Random-1 từ benchmark.

## Artifact cần có cho mỗi split

- Checkpoint/model state.
- Training configuration.
- Training log.
- Evaluation command.
- Per-query result.
- PostgreSQL baseline.
- Planning time.
- Execution time.
- End-to-end time.
- Status COMPLETED / TIMEOUT / ERROR.

## Output cuối

Balsa phải có đủ:

`Balsa_LOO`

`Balsa_RANDOM`

`Balsa_BASE`

và kết quả phải giữ toàn bộ official Test set.

Quyên sẽ lấy:

`Test_LOO ∩ Test_RANDOM ∩ Test_BASE`

ở bước Data Mining để xây dựng tập `Q_common`.