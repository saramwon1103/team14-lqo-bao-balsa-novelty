# Balsa: checkpoint và nghiệm thu logging LOO / Random

## Balsa V2 theo protocol chung (đang chuẩn bị, chưa chạy)

Các artifact LOO-template Fold 02/Random-52 bên dưới là evidence kỹ thuật cũ,
không phải kết quả RQ2. Nguồn split V2 duy nhất là
`analysis/data/split_master.csv` trên repo gốc. Adapter từ chối sai counts,
SQL JOB thiếu/thừa, query/family/sql_file bất nhất, duplicate hoặc train/test
overlap; lưu hash CSV, source commit và membership hash. Ba manifest được sinh
tại `balsa/source/manifests/v2/`.

Đã khóa counts: LOO-1 80/33, Random-1 91/22, BaseQuery-1 94/19. `novelty.csv`
có 74 unique keys `(query_id,family,split_type,split_id)` và coverage khớp
33/22/19; exporter kiểm tra one-to-one theo từng split. `analysis/src/merge_results.py`
hiện là file rỗng (0 byte), nên chưa có script merge chính thức để gọi; không
ghi hay tái tính Novelty Score.

TODO V2 và `analysis/data/novelty.csv` đặt join keys/coverage nhất quán. Thời
gian: V2 chỉ định `inference_ms=balsa_planning_ms`, `planning_ms` là PostgreSQL
hinted Planning Time, và E2E là tổng ba thành phần. Bao README cũ dùng
`planning_ms` đã gồm inference và để `inference_ms` trống. Đối với Balsa V2,
chúng tôi giữ contract rõ ràng của TODO V2 (không diễn giải lại dữ liệu Bao).

Chưa có run budget V2 phê duyệt hoặc kết quả train/eval V2. **Không dùng các
lệnh dưới đây trước khi nhóm chốt ngân sách/guard**; mỗi split phải train mới,
không dùng Random-52/LOO-template/BaseQuery cũ làm model/cache/checkpoint.

### Cập nhật nguồn và validate/generate manifest

Trên clone mới nhất của repository nhóm:

```bash
git fetch origin
git status --short
python balsa/source/scripts/split_master_adapter.py \
  --source-csv analysis/data/split_master.csv \
  --sql-dir balsa/source/queries/join-order-benchmark \
  --output-dir /tmp/balsa-v2-manifests \
  --source-commit "$(git log -1 --format=%H -- analysis/data/split_master.csv)"
```

Adapter đã chạy trên analysis commit `70061bad5969e34b999dc0cf0364801d56060c45`,
split-file commit `5c9d6403089289a3188aa05ea75ef33c62b50177`, SHA-256
`804cc4ff7d764173670b65d7b0655037b398356171adcb9555df73e3fa3f8880`. Output
được xác minh với toàn bộ 113 SQL JOB của source tree Balsa.

### Train/eval riêng từng split (chỉ sau khi budget được duyệt)

Mỗi output tạm phải là thư mục mới. Sau train, dùng đúng checkpoint vừa tạo
cho evaluator mới; không truyền checkpoint/replay/simulator từ split khác:

```bash
PY=/path/to/verified-balsa-python
SPLIT=analysis/data/split_master.csv
SQL=balsa/source/queries/join-order-benchmark
RUN_ID="loo1_$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" balsa/source/scripts/train_split.py --split-master "$SPLIT" --split-type LOO-1 \
  --output-dir "/tmp/$RUN_ID/training" --allow-full --iterations 3 --seed 0 \
  --initial-timeout-ms 10000 --isolated-runtime --independent-eval-only \
  --simulator-integration-epochs 1
"$PY" balsa/source/scripts/evaluate_checkpoint.py \
  --checkpoint "/tmp/$RUN_ID/training/<wandb-run>/checkpoints/evaluation.pt" \
  --output-dir "/tmp/$RUN_ID/evaluation" --seed 0 --repeats 1 --timeout-ms 30000
"$PY" balsa/source/scripts/export_common_results.py \
  --training-dir "/tmp/$RUN_ID/training" --evaluation-dir "/tmp/$RUN_ID/evaluation" \
  --split-master "$SPLIT" --sql-dir "$SQL" --novelty-csv analysis/data/novelty.csv \
  --output-dir "balsa/runs/LOO-1/$RUN_ID" --run-id "$RUN_ID" --code-commit "$(git rev-parse HEAD)"
```

Lặp nguyên mẫu với `SPLIT_TYPE=Random-1` hoặc `BaseQuery-1` và đúng thư mục
đích tương ứng. Ví dụ command dùng biến có thể chạy tuần tự cho cả ba:

```bash
declare -A RUN_IDS
set -o pipefail
for SPLIT_TYPE in LOO-1 Random-1 BaseQuery-1; do
  RUN_ID="${SPLIT_TYPE//-/_}_$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="/tmp/$RUN_ID"; mkdir "$RUN_DIR"
  "$PY" balsa/source/scripts/train_split.py --split-master "$SPLIT" --split-type "$SPLIT_TYPE" \
    --output-dir "$RUN_DIR/training" --allow-full --iterations 3 --seed 0 \
    --initial-timeout-ms 10000 --isolated-runtime --independent-eval-only \
    --simulator-integration-epochs 1 2>&1 | tee "$RUN_DIR/raw_training_stdout.log"
  CKPT="$(find "$RUN_DIR/training" -path '*/checkpoints/evaluation.pt' -print -quit)"
  test -n "$CKPT"
  "$PY" balsa/source/scripts/evaluate_checkpoint.py --checkpoint "$CKPT" \
    --output-dir "$RUN_DIR/evaluation" --seed 0 --repeats 1 --timeout-ms 30000 \
    --run-id "$RUN_ID" \
    2>&1 | tee "$RUN_DIR/raw_eval_stdout.log"
  RUN_IDS[$SPLIT_TYPE]="$RUN_ID"
done
# Export only after all three clean-tree training invocations have finished.
for SPLIT_TYPE in LOO-1 Random-1 BaseQuery-1; do
  RUN_ID="${RUN_IDS[$SPLIT_TYPE]}"; RUN_DIR="/tmp/$RUN_ID"
  "$PY" balsa/source/scripts/export_common_results.py --training-dir "$RUN_DIR/training" \
    --evaluation-dir "$RUN_DIR/evaluation" --split-master "$SPLIT" --sql-dir "$SQL" \
    --novelty-csv analysis/data/novelty.csv --output-dir "balsa/runs/$SPLIT_TYPE/$RUN_ID" \
    --raw-training-log "$RUN_DIR/raw_training_stdout.log" --raw-eval-log "$RUN_DIR/raw_eval_stdout.log" \
    --run-id "$RUN_ID" --code-commit "$(git rev-parse HEAD)"
done
```

`split_id=1` không phải random seed. Đây là cấu hình **đề xuất, chưa được
chạy/chưa có V2 budget phê duyệt**: 3 value iterations; tối đa 1 epoch
simulator mới trên train của chính split; max 100 epochs/agent iteration với
early stopping gốc; beam 20/10 complete plans; 10s initial execution guard,
30s eval/query, 1 repeat. Lệnh trực tiếp không tự tạo cgroup guard. Cần duyệt
tổng wall-clock/storage/RAM/PostgreSQL trước khi chạy.

### Tests và kiểm tra analysis contract

```bash
"$PY" -m unittest discover -s balsa/source/tests -v
```

Exporter bắt buộc one row/test, baseline cùng đúng Test IDs, no fabricated
timeout timing, `E2E=inference+planning+execution`, checkpoint path thật, ít
nhất một optimizer update, eval `train_called=false`/weights unchanged, và
one-to-one match với Novelty composite key. Lần kiểm tra hiện tại chỉ xác nhận
contract/keys trên `novelty.csv`, chưa có phép merge từ ba Balsa V2 run vì chưa
train.

Đã chạy **agent train thật → checkpoint → evaluator tiến trình mới → PostgreSQL
thực thi hinted SQL → CSV và EXPLAIN ANALYZE** cho hai giao thức dưới đây.
Đây là nghiệm thu pipeline, **không phải** benchmark 33 fold, 500 vòng hoặc
tái lập kết quả bài báo; không kết luận simulator hội tụ hay Balsa tăng tốc.

| Giao thức | Seed agent | Vòng | Optimizer step agent | Test thành công | Nhãn |
|---|---:|---:|---:|---:|---|
| LOO-template Fold 02 | 0 | 3 | 65 | 4/4 | Agent full, khởi tạo simulator cùng fold đã train một phần |
| Random-52 | 0 | 3 | 200 | 19/19 | Integration: simulator train-only 1 epoch; agent full |

Cả hai: không timeout/error/hint mismatch trong fresh eval, hash trọng số
không đổi, `train_called=false`. Vòng 0 là `SKIPPED_EXPERT_INIT` (0 step).
Agent full: tối đa 100 epoch/vòng, beam 20, 10 complete plans; chỉ giới hạn
3 value iteration và chuyển báo cáo test sang evaluator độc lập.
Random split seed **52** khác seed agent **0**; dùng đúng 94 train/19 test
upstream. LOO-template là giao thức mở rộng của nhóm (109 train/4 test).

## Các tệp cần dùng

Các đường dẫn dưới đây tính từ **repository root**:

| Giao thức | Checkpoint agent | CSV đánh giá |
|---|---|---|
| Fold 02 | `balsa/checkpoint/loo_template_fold02_seed0/evaluation.pt` | `balsa/results/acceptance_loo02_random_seed0/A_fold02_retry2/evaluation/balsa_results.csv` |
| Random | `balsa/checkpoint/random52_seed0/evaluation.pt` | `balsa/results/acceptance_loo02_random_seed0/B_random52_retry2/evaluation/balsa_results.csv` |

- Script test: `balsa/source/scripts/evaluate_checkpoint.py`.
- Toàn bộ mã phụ thuộc, 113 SQL JOB, manifest và tests: `balsa/source/`.
- Mỗi run có config/split, fit evidence, `experiment.log` theo mục [1]–[4],
  log thô train/eval, `evaluation/postgres_baseline.csv`, hinted SQL,
  EXPLAIN ANALYZE và baseline JSON trong `evaluation/plans/`.
- Báo cáo chi tiết: `balsa/results/acceptance_loo02_random_seed0/HANDOFF.md`.
  Một số đường dẫn trong báo cáo/metadata gốc là đường dẫn lịch sử trên máy
  thí nghiệm; bảng trên là vị trí công bố. CSV đã đổi đường dẫn plan thành
  đường dẫn tương đối từ repository root.
- `balsa/PUBLISHED_MANIFEST.json`: checksum của các tệp công bố; không phải
  checksum của Git LFS pointer. Metadata checkpoint ghi hash binary thật.

## Tải checkpoint bằng Git LFS

Checkpoint được lưu bằng **Git LFS**, không đưa binary vào Git thường.

```bash
git lfs install
git clone https://github.com/saramwon1103/team14-lqo-bao-balsa-novelty.git
cd team14-lqo-bao-balsa-novelty
# Checkout nhánh/PR chứa gói này nếu chưa merge vào main.
git lfs pull --include='balsa/checkpoint/**/*.pt'
```

Chỉ nạp checkpoint pickle/PyTorch từ nguồn tin cậy. Đây là bundle **evaluation**,
có model, featurizer, thứ tự đặc trưng, chuẩn hóa nhãn và manifest; không hứa
exact training resume. Simulator/cache lớn và database không được đưa vào Git.

## Môi trường và PostgreSQL

Bộ mã legacy: Python 3.7.12, torch 1.4.0, pytorch-lightning 0.9.0, Ray 1.9.2,
W&B 0.18.7 offline; xem `balsa/source/requirements.txt` và README nguồn để
dựng môi trường. Không tự nâng phiên bản rồi coi tương đương.

PostgreSQL thí nghiệm: 12.5, JOB/IMDB đã nạp, pg_hint_plan phải hoạt động.
Thiết lập kết nối thống nhất cho preflight/trainer/evaluator/worker:

```bash
export PGHOST=/tmp PGPORT=5432 PGUSER=wings BALSA_DB=imdbload
export CUDA_VISIBLE_DEVICES='' WANDB_MODE=offline
# Đổi host/role/database cho đúng máy của bạn; không đưa mật khẩu vào Git.
export PY=/path/to/balsa-env/bin/python
"$PY" balsa/source/scripts/preflight_balsa.py
```

Máy đã đo dùng CPU, PG shared_buffers=4GiB/work_mem=64MiB/parallel-query=2.
Evaluator đặt session GEQO off, parallel-query=0 và timeout 30 giây cho cả
hinted và baseline. Cache warm không kiểm soát, không drop cache/restart PG.

## Test riêng từ checkpoint, không cần trainer/simulator còn sống

Từ repository root, sau khi tải Git LFS và thiết lập môi trường bên trên:

```bash
"$PY" balsa/source/scripts/evaluate_checkpoint.py \
  --checkpoint balsa/checkpoint/loo_template_fold02_seed0/evaluation.pt \
  --output-dir /tmp/balsa_eval_loo02_NEW --seed 0 --repeats 1 --timeout-ms 30000

"$PY" balsa/source/scripts/evaluate_checkpoint.py \
  --checkpoint balsa/checkpoint/random52_seed0/evaluation.pt \
  --output-dir /tmp/balsa_eval_random_NEW --seed 0 --repeats 1 --timeout-ms 30000
```

Output phải chưa tồn tại. Lệnh cùng entry point/flags đã chạy thành công trên
máy thí nghiệm; checkpoint công bố giữ nguyên SHA-256. Các đích `_NEW` là cho
lần đo mới, không phải artifact đã có. Evaluator cấm gọi train, kiểm tra
membership/seed và hash trước/sau, thực thi thật mọi query/repeat.

## Train mới và kiểm thử

```bash
"$PY" -m unittest discover -s balsa/source/tests -v

# Random integration: simulator mới chỉ từ train Random, không dùng Fold 02.
"$PY" balsa/source/scripts/train_split.py \
  --manifest balsa/source/manifests/random52.json --output-dir /tmp/balsa_train_random_NEW \
  --allow-full --iterations 3 --seed 0 --initial-timeout-ms 10000 \
  --isolated-runtime --independent-eval-only --simulator-integration-epochs 1

# Fold 02 từ đầu: sẽ thu dữ liệu và train simulator mới; không dùng checkpoint
# Random cho fold này. Cần ngân sách riêng, không phải lệnh resume pilot cũ.
"$PY" balsa/source/scripts/train_split.py \
  --manifest balsa/source/manifests/loo_template/fold_02.json --output-dir /tmp/balsa_train_loo02_NEW \
  --allow-full --iterations 3 --seed 0 --initial-timeout-ms 10000 \
  --isolated-runtime --independent-eval-only
```

Trainer trực tiếp không tự đặt cgroup/time budget: phải triển khai guard phù
hợp trước khi chạy. Các supervisor nghiệm thu lưu kèm mã nhưng chứa đường
dẫn máy thí nghiệm, không phải script cài đặt đa máy. Lần Fold 02 đã công bố
khởi tạo từ simulator cùng fold một phần (checkpoint step6050), đã phục hồi
và đối chiếu đủ 625035 vector query/nhãn với cache; không pretrain lại trong
run đó. Cache và simulator initialization không nằm trong gói tối thiểu này.

## Đọc số liệu đúng

`end_to_end_ms = balsa_planning_ms + postgres_hinted_planning_ms + execution_ms`;
client wall-clock đo riêng. Timeout/error không có execution hoàn tất phải
để trống/N/A, không lấy ngưỡng timeout làm thời gian thực thi. Speedup chỉ có
trên cặp hợp lệ cùng query/repeat; fresh paired baseline coverage là 4/4 và
19/19. Baseline pilot Fold 02 cũ được tái sử dụng làm expert giữ **110 thành
công/3 timeout**, không đổi thành 113 thành công hoặc gọi là thực thi mới.

Nguồn upstream: `balsa-project/balsa`, commit
`a0b45f2371c8c569bb90ad611f39911e5b850852`, Apache-2.0 (LICENSE trong source).
25 unit tests của gói nghiệm thu PASS; suite trên repository sau ghép được
ghi thêm trong `published_tests.log`. Không có kết luận full benchmark.
