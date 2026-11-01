# Logging LOO-template Fold 02 và Random — nghiệm thu

**Logging LOO và Random đã được nghiệm thu**, chỉ trong phạm vi pilot/integration được duyệt.

| Protocol | Simulator | Agent vòng / optimizer step mới | Fresh eval thành công / timeout / lỗi / mismatch | Cặp baseline hợp lệ | Wall s |
|---|---|---|---|---|---|
| Fold 02 | Cùng fold, checkpoint epoch 11 / 6050 step; chưa chứng minh hội tụ | 3 / 65 (45 + 20) | 4 / 0 / 0 / 0 | 4/4 | 1282.631 |
| Random-52 | Tạo mới train-only, integration 1 epoch / 455 step, 516759 điểm | 3 / 200 (100 + 100) | 19 / 0 / 0 / 0 | 19/19 | 1799.404 |

Agent cả hai: seed 0, full max_epochs=100 với early stopping gốc, beam=20, 10 complete plans. Seed split Random=52 (không phải seed agent). Vòng 0 SKIPPED_EXPERT_INIT, 0 step; hai vòng tiếp theo đổi 26 tensor mỗi lần. Optimizer tạo lại theo cấu hình gốc, số step trong bảng là tổng các lần fit, không phải counter resume liên tục.

A bao gồm chuẩn bị/checksum và hai lần lỗi trước fit đã sửa trong đồng hồ gốc (Ray socket quá dài; logging optimizers=None khi skip fit). Tổng điều phối train/eval: 3083.052 s (~51.4 phút), không chạy hai protocol song song.

## Các giai đoạn (giây; không cộng lồng phase cha/con)
| Phase | Fold 02 | Random |
|---|---:|---:|
| Init agent/simulator | 3.742 (nạp simulator đã kiểm chứng) | 726.372 (gồm collection/encoding/pretrain) |
| Fresh expert baseline | Không chạy lại; giữ CSV 110 success / 3 timeout | 79.588 |
| Simulator data collection | Tái sử dụng, không chạy lại | 225.347 |
| Simulator featurization | Cache đã đối chiếu toàn bộ 625035 vector query/nhãn; không tạo lại tree tensor | 84.179 |
| Simulator neural training | Không chạy lại | 387.252 |
| Agent fit/encoding dataset | 58.258 | 153.472 |
| Agent planning train | 180.026 | 86.966 |
| Client wait train execution | 824.467 | 668.687 |
| Eval Balsa planning (tổng test) | 0.496 | 5.282 |
| Eval PG hinted planning (tổng test) | 0.005 | 0.087 |
| Eval PG execution (tổng test) | 7.571 | 48.755 |
| Eval component E2E (tổng test) | 8.072 | 54.124 |
| Eval client wall (tổng test, chưa gồm baseline sau hinted) | 8.086 | 54.196 |

Thời gian init/collection/pretrain là phase lồng nhau; wait phía client không phải tổng Execution Time PostgreSQL. Baseline mới của eval được đo theo từng query/repeat, cùng GEQO off, parallel/query=0, timeout 30s; chỉ các cặp hợp lệ có speedup. Không kết luận Balsa tăng tốc từ các phép đo nghiệm thu này.

## Tài nguyên và ngân sách
```json
{
  "peak_cgroup_kernel_gib": 21.976459503173828,
  "min_machine_available_gib": 7.608707427978516,
  "pg_growth_peak_gib": 5.9569549560546875,
  "live_temp_peak_gib": 5.957763671875,
  "per_phase_peak_gib": {
    "A_": 19.261619567871094,
    "B_": 21.976459503173828
  },
  "memory_events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\nsock_throttled 0\n",
  "elapsed_through_packaging_seconds": 3308.8281075954437,
  "note": "Overall machine minimum recomputed across all attempts; supervisor continuation summary had lost the first-attempt minimum. Values here come from append-only resource samples."
}
```

Đỉnh mẫu RAM theo phase bao gồm helper/filesystem cache; kernel peak là counter toàn cgroup. A phase gồm các lần sửa trước fit. Cả phase có mốc riêng 2 giờ, toàn phiên 4 giờ. Artifact gồm image preallocated 17 GiB thực cấp, export/source và gói nộp, tính tổng bằng st_blocks; số liệu cuối nằm trong FINAL_RESOURCE_AUDIT.json.

## Bằng chứng và đường dẫn
- Fold 02: `A_fold02_retry2/training/7qkbbqub/checkpoints/evaluation.pt`; CSV `A_fold02_retry2/evaluation/balsa_results.csv`.
- Random: `B_random52_retry2/training/ax7jy4l3/checkpoints/evaluation.pt`; CSV `B_random52_retry2/evaluation/balsa_results.csv`.
- Mỗi protocol: `experiment.log` ghép đủ mục [1]–[4], log train/eval thô, `evaluation/plans/` chứa hinted SQL, EXPLAIN ANALYZE JSON và baseline JSON; `evaluation/postgres_baseline.csv` tương ứng.
- `AUDIT.json`: checkpoint/split hash và coverage. `reload_and_plan_audit.json`: đã nạp lại hai checkpoint và phân tích lại tất cả JSON, hint khớp SQL.
- `correctness_crosscheck.json`: toàn bộ train feedback mỗi vòng khớp 109/94 query train; không có query test trong feedback. PID evaluator mới khác trainer/worker; train_called=false, hash trọng số/checkpoint bất biến. Kết nối cùng PG12.5/imdbload/wings/data_directory.
- Fold 02 simulator metadata/bundle và expert pickle trong `A_fold02_retry2/simulator_initialization/`; all-query feature/label digest, vocabulary và chuẩn hóa được phục hồi/xác minh. Random simulator checkpoint/provenance riêng trong `B_random52_retry2/training/simulator/`.
- `final_unit_tests_verified.log` và `package_unit_tests.log`: 25/25 PASS, gồm tương đương tree feature qua chunk4096, bảo toàn metadata, từ chối cross-split, parser/CSV và eval không gọi train. `filesystem_readonly_check.log`: e2fsck -fn PASS 5 pass.
- Source và phiên bản sửa trước/sau thử lỗi được giữ trong `source/`, `code_corrections/`, nguồn snapshot ban đầu trong phiên gốc. Không có Git commit vì thư mục không phải repo.

## Giới hạn và đính chính
- Random mang nhãn integration do simulator chỉ 1 epoch; không gọi là pilot full-simulator hay benchmark bài báo. Agent hyperparameters vẫn full.
- Fold 02 dùng simulator đã huấn luyện một phần, không exact resume; không chứng minh simulator hội tụ, không biết thêm 3 epoch có early-stop hay không.
- 1 repeat/query cho nghiệm thu; cache máy warm không kiểm soát. Không xác nhận benchmark 33 fold, 500 vòng, hiệu năng dài hạn hoặc dự toán 35 run.
- Các gói smoke cũ 335/387 checksum vẫn đúng. Không xóa pilot cũ, không restart/sửa PG config, không sudo/drop cache, không push Git. Tất cả service/timer thí nghiệm đã inactive, không còn query mang tag phiên. Không chạy tiếp.
