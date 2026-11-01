# Balsa: train thật và test độc lập từ checkpoint

**Nghiệm thu logging mới: LOO-template Fold 02 và Random đã hoàn tất** (mỗi
protocol 3 vòng; fresh eval lần lượt 4/4 và 19/19 query, hint khớp, hash trọng
số không đổi). Random mang nhãn integration với simulator 1 epoch; agent giữ
cấu hình full. Xem `README_ACCEPTANCE.md` và gói mới
`results/acceptance_loo02_random_seed0_v1_submission/`. Đây không phải kết quả
33 fold/500 vòng hoặc bằng chứng simulator hội tụ/Balsa tăng tốc.

Pilot cấu hình full fold 02 đã được phê duyệt riêng với giới hạn 2 giờ,
16 GiB ML và 20 GiB artifact. Chính sách triển khai/ảnh hưởng phép đo:
`results/pilot_setup/OPERATING_POLICY.md`. Run pilot và trạng thái nằm ở
`results/pilot_full_fold02_seed0_3iter/`; không suy ra hoàn tất từ tên thư mục.
**Pilot đã vượt qua hoàn toàn bước featurization simulator (625.035 điểm) sau khi
sửa rò/tích tụ RAM (`anon` đỉnh 14,21 GiB, không OOM/high-throttle), hoàn tất
13 epoch simulator (7.150 step, checkpoint nguyên tử `epoch=10.ckpt`) và dừng có
kiểm soát tại mốc dự trữ wall-clock 6.600 giây (`results/pilot_full_fold02_seed0_chunked24g/`):**
0/3 vòng Balsa agent hoàn tất trong ngân sách 2 giờ. Xem `results/HANDOFF_PILOT.md`;
không tự chạy lại hoặc chạy tiếp 35 split.

## Phạm vi và nguồn

Balsa gốc: https://github.com/balsa-project/balsa, commit tham chiếu
`a0b45f2371c8c569bb90ad611f39911e5b850852`, Apache-2.0.
Mã đầy đủ cần thiết, SQL JOB và giấy phép nằm trong `source/`.
Không đóng gói PostgreSQL data directory, Conda env hay dữ liệu CSV IMDB.
Thư mục nộp bài chưa có Git; không init Git, commit hoặc push.

**Đã kiểm chứng:** fit thật → bundle → tiến trình train kết thúc → tiến trình
mới nạp bundle → Balsa chọn plan → PostgreSQL thực thi → CSV/SQL/EXPLAIN JSON.
Smoke LOO-template fold 02 mới có simulator thật, đủ 109 train/4 test;
simulator 6 optimizer step, agent 1 step, eval 4/4 hint khớp. Bằng chứng ở
`results/loo_template_smoke_02_retry/`, báo cáo và kế hoạch chờ duyệt ở
`results/HANDOFF_LOO.md`. Kết quả smoke cũ giữ nguyên.
Đây là smoke kiểm tra pipeline, **không phải benchmark hiệu năng hoặc kết quả
500 vòng**, không chứng minh chất lượng mô hình Balsa sau huấn luyện đầy đủ.

## Môi trường Linux đã dùng

- Python 3.7.12; torch 1.4.0; pytorch-lightning 0.9.0; Ray 1.9.2.
- Các phiên bản trong `source/requirements.txt`; `results/pip_check.log` ghi
  kết quả kiểm tra dependency. Không tự nâng stack legacy.
- CPU cho smoke/eval: `CUDA_VISIBLE_DEVICES=''`. RTX 5060 Ti/driver CUDA 13
  không tương thích đường chạy torch 1.4 đã thử; chưa nghiệm thu GPU.
- PostgreSQL server 12.5, role `wings`, DB `imdbload`, Unix socket `/tmp`,
  port 5432; cluster `/home/wings/.local/share/balsa-pgdata`.
- Preflight đếm thực tế cả **21 bảng**, không chỉ bảng `title`, và LOAD
  `pg_hint_plan`. Không sửa quyền, sudo hoặc khởi động lại PostgreSQL.

Sau khi dựng môi trường theo README upstream, cài mã và executor:

```bash
conda activate balsa
cd source
python -m pip install -e .
python -m pip install -e pg_executor
```

Từ thư mục gốc gói nộp, cấu hình dùng chung qua libpq:

```bash
export PGHOST=/tmp PGPORT=5432 PGUSER=wings BALSA_DB=imdbload
export CUDA_VISIBLE_DEVICES='' WANDB_MODE=offline
python source/scripts/preflight_balsa.py
python -m pip check
python -m unittest discover -s source/tests -v
```

Preflight, driver, evaluator và Ray worker đã được đối chiếu bằng DB identity,
data directory và PID, không chỉ suy luận từ biến môi trường. Bằng chứng:
`results/smoke_locked_v2/connections.jsonl`,
`results/eval_locked_connections.jsonl`, và preflight của run.
Không đặt mật khẩu/DSN bí mật vào mã hoặc log. Dùng `.pgpass` nếu cần.

## Split đã khóa

`source/manifests/` có manifest tường minh và SHA-256:

- `basequery1.json`: 94 train / 19 test, **split tùy chỉnh đã có**, không gọi
  là official Balsa; nguồn và hash `experiments.py` được ghi trong manifest.
- `random52.json`: 94/19; danh sách được đối chiếu trực tiếp với
  `RAND_52_TEST_QUERIES` tại commit upstream trên; split seed 52.
  Không bốc ngẫu nhiên lại. Split seed khác training seed.
- `smoke.json`: 4 train / 2 test, không simulator, 1 epoch, 1 vòng,
  beam 1; `skip_training_on_expert=False` để fit thật ngay vòng 0.
- `loo_template/fold_01.json` … `fold_33.json`: người dùng đã chọn
  **leave-one-query-template-out**, giao thức mở rộng của nhóm, không phải
  official split của bài báo Balsa. Mapping nguồn/hash ở `job_templates.json`.
  §2.2 bài JOB xác nhận 33 cấu trúc / 113 SQL; đã đối chiếu **bytes của cả
  113 SQL** với upstream cố định và kiểm tra FROM relation roles + các cạnh
  equijoin theo cột, chuẩn hóa alias (đặc biệt nhóm 8, 15). Không chỉ cắt tên tệp.

```bash
python source/scripts/make_splits.py --validate source/manifests/basequery1.json
python source/scripts/make_splits.py --validate source/manifests/random52.json
```

`make_splits.py --help` mô tả cách tạo manifest **ở đường dẫn mới**;
không ghi đè manifest đã khóa. BaseQuery không được nâng thành official chỉ
vì kiểm thử membership thành công.

## Train và test

Lệnh đã chạy với expert workload tin cậy có sẵn trên máy:

```bash
python -u source/scripts/train_split.py \
  --manifest source/manifests/smoke.json --output-dir results/smoke_locked_v2 \
  --smoke --seed 0 \
  --expert-data /home/wings/balsa_official_with_logging/data/initial_policy_data_basequery1_smoke.pkl

python source/scripts/evaluate_checkpoint.py \
  --checkpoint results/smoke_locked_v2/ufitcr9s/checkpoints/evaluation.pt \
  --output-dir results/eval_locked --seed 0
```

Các đường dẫn output trên **đã tồn tại**, nên khi chạy lại phải chọn đường
dẫn mới. Bỏ `--expert-data` để script thu expert bằng PostgreSQL thật. Expert
có sẵn được ghi `CACHE` và hash trong `expert_provenance.json`, không giả là
baseline vừa thực thi. Evaluator luôn đo baseline mới cho mỗi query/repeat.

`train_split.py` cô lập expert, replay và checkpoint theo run; full cần
`--allow-full` rõ ràng. `run_all_splits.sh` chạy tuần tự manifest và lưu
`exit_codes.tsv`; không hỗ trợ resume chính xác optimizer. Chưa chạy các
đường full hoặc Random full; LOO chỉ chạy smoke một fold, chưa có benchmark.

## Checkpoint và bằng chứng

- `latest.pt`: raw Balsa state_dict tương thích đường lưu cũ.
- `evaluation.pt`: bundle tin cậy chứa model, cấu hình, featurizer/vocabulary,
  workload metadata, biến đổi nhãn **đã fit trên train**, test membership và SQL.
- `evaluation.pt.json`: checksum/provenance khi dùng bản script mới.
- Bundle ghi qua tệp tạm rồi rename; **evaluation checkpoint, không phải
  checkpoint resume chính xác**. Chỉ load pickle/PyTorch từ nguồn tin cậy.
- Eval không khởi tạo `BalsaAgent`, không gọi train/simulator, không dựng lại
  dataset từ nhãn; dùng `eval()` + `no_grad()`.
- Hash tensor được tính trên tên/dtype/shape/bytes, không dùng hash của
  `torch.save(BytesIO)` vì định dạng tuần tự hóa không phải hash nội dung ổn định.
- `fit_evidence_0.json`: hash cùng tiến trình trước/sau fit, batch thực tế,
  Lightning global step và step trong Adam state. Vòng 0 gốc có thể skip fit;
  trạng thái skip không được dùng như bằng chứng huấn luyện.

`results/eval_locked/evaluation_summary.json`: 0 lỗi, weights và checkpoint
hash không đổi. `eval_locked_repeat/predictions.json` trùng chính xác với lần
nạp trước về prediction và hint; thời gian không bắt buộc giống nhau.

## Quy ước đo và CSV

CSV độc lập ở `results/eval_locked/balsa_results.csv` theo contract mục 6:
run/split/fold/seed, hash, thời gian, nguồn, hint và đường dẫn plan.
`fold_id` không áp dụng để trống, không đặt một fold giả.

- Balsa planning: client bao quanh encoding + beam search + chọn plan.
- PG hinted planning/execution: `Planning Time` / `Execution Time` trong
  EXPLAIN ANALYZE thực tế, không dùng estimated cost như mili giây.
- End-to-end = Balsa planning + PG hinted planning + execution khi đủ số đo.
- Wall clock đo riêng từ trước planning tới nhận EXPLAIN hinted, có overhead
  client và ghi hinted SQL, **không gồm baseline chạy sau đó**.
- Speedup execution = baseline execution / Balsa execution; speedup E2E =
  baseline (planning + execution) / Balsa E2E.
- Timeout/error/hint mismatch không được tính speedup như kết quả hoàn tất.
  Timeout không thay execution bằng ngưỡng timeout.
- Eval `MEASURED` mỗi repeat, không tái dùng execution cache. Cache nội bộ
  dự đoán của planner không phải cache kết quả PostgreSQL.
- Eval và baseline cùng GEQO off, parallel workers per gather = 0,
  cùng statement timeout, 0 warm-up riêng; thứ tự hinted trước baseline.
  **Cold cache không được kiểm soát**: không drop OS/shared buffer cache,
  `sync` và `DISCARD ALL` không được coi là drop cache. Kết quả smoke không
  phải phép so sánh cold-cache công bằng tuyệt đối.
- CSV trong vòng train là monitoring riêng, baseline expert có thể cũ;
  không trộn với CSV eval độc lập để tuyên bố benchmark.
- Smoke seed 0 có train query `16b` timeout 10 giây; đây là lỗi/timeout thật
  được lưu N/A execution. Hai query eval độc lập hoàn tất nhưng mô hình smoke
  chậm hơn PostgreSQL; không diễn giải là tăng tốc.

W&B 0.18.7 offline `Table.to_json` gọi kiểm tra client/server qua GraphQL,
dẫn tới 401; không phải train thất bại. Offline giữ bảng JSON cục bộ và bỏ
đường Table gửi mạng; scalar ghi `local_metrics.jsonl` trước backend. Lỗi
backend có cảnh báo rõ, không `except: pass`.

## Gom kết quả

`collect_submission.py --help` kiểm tra CSV/plan/hash, từ chối run chưa đạt,
gom mã theo allowlist cùng checkpoint và tạo `artifact_manifest.json`.
Các script Python xử lý `--help` trước khi import ML khi khả thi.

Không chạy nhiều phép đo độ trễ đồng thời trên cùng PostgreSQL. Chưa chạy
500 vòng hoặc toàn bộ LOO. Xem `results/HANDOFF.md` để biết kết quả kiểm thử,
các lệnh, giới hạn còn lại và dự toán full.

## LOO-template và orchestration dùng chung

`run_experiment.py` dùng cho Random, BaseQuery và LOO: chờ **tiến trình train
kết thúc** rồi gọi evaluator trong tiến trình mới. `run_all_splits.sh` gọi entry
này tuần tự, lưu exit code và không coi run thất bại là đã hoàn tất.
Đầu ra mỗi run:

```text
<protocol>/<split_or_fold>/<seed>/<run_id>/
  raw_train.log, raw_test.log, run_summary.json
  training/
    split.json, config.txt, preflight.log, connections.jsonl
    simulator/{simulator.ckpt,provenance.json,sim-data-*,sim-featurized-*}
    <training_run_id>/checkpoints/{latest.pt,evaluation.pt,evaluation.pt.json}
    <training_run_id>/{replay/,iterations/,fit_evidence_*,experiment.log}
  evaluation/
    balsa_results.csv, experiment.log, split.json, config.txt,
    environment.json, evaluation_summary.json, predictions.json, plans/
```

Caller chọn output directory mới; manifest/CSV mang protocol/fold/seed/hash
dù tên đường dẫn có convention khác. Mỗi process tạo model, optimizer và
query execution cache mới. Simulator checkpoint, simulation data/feature cache,
expert và replay đều nằm dưới run; không lấy checkpoint simulator chia sẻ từ
Random hoặc fold khác. Query-featurizer Fit và label transforms chỉ dùng train.

Manifest LOO giữ **đủ 113 SQL trong mọi fold**, test 2–6 biến thể của đúng nhóm,
train toàn bộ nhóm còn lại. Unit test kiểm tra cả 33 fold, mỗi SQL test đúng
một lần; đổi nhóm dù tính lại hash cũng bị từ chối. Trước train kiểm tra hash
mapping và hash nội dung SQL tại chỗ. Snapshot mã nguồn BaseQuery cũ được giữ
ở `source/manifests/provenance/experiments_original.py` để nguồn/hash của
manifest cũ vẫn truy xuất được khi mã training đã bổ sung tùy chọn simulator.

Smoke simulator dùng đủ membership fold nhưng 1 epoch simulator, 1 epoch
agent, 1 iteration, beam 1, simulator chỉ enumerate query dưới 6 relation;
full mặc định ngưỡng 12. Giới hạn smoke này **không đại diện full Balsa**.
Baseline timeout được ghi TIMEOUT/N/A; nhãn phạt nội bộ không phải thời gian
thực thi, và những expert timeout bị loại khỏi tập fit. Tập test không đi vào
replay hoặc fit chuẩn hóa. Xem `results/HANDOFF_LOO.md` cho bằng chứng và
kế hoạch ngân sách; chưa tự chạy lệnh full khi người dùng chưa duyệt.
