## Experimental Architecture

Nghiên cứu sử dụng ba cách chia dữ liệu chính thức:

- LOO
- Random
- Base Query

Mỗi cách chia tạo ra một cặp Train/Test độc lập:

`Train_LOO → Test_LOO`

`Train_RANDOM → Test_RANDOM`

`Train_BASE → Test_BASE`

Bao và Balsa phải được huấn luyện hoặc nạp checkpoint tương ứng với từng tập Train. Vì vậy, về mặt thực nghiệm cần có sáu trạng thái mô hình:

| Split | Bao | Balsa |
|---|---|---|
| LOO | Bao_LOO | Balsa_LOO |
| Random | Bao_RANDOM | Balsa_RANDOM |
| Base Query | Bao_BASE | Balsa_BASE |

### Official benchmark evaluation

Trước tiên, mỗi mô hình được đánh giá trên tập Test chính thức tương ứng:

`Bao_LOO / Balsa_LOO → Test_LOO`

`Bao_RANDOM / Balsa_RANDOM → Test_RANDOM`

`Bao_BASE / Balsa_BASE → Test_BASE`

Kết quả này được sử dụng để tái lập benchmark và kiểm tra hiệu năng của từng LQO theo từng cách chia dữ liệu.

### Controlled Novelty Analysis

Để phân tích ảnh hưởng của tập Train lên cùng một truy vấn kiểm thử, nhóm xác định:

`Q_common = Test_LOO ∩ Test_RANDOM ∩ Test_BASE`

Với mỗi query `q ∈ Q_common`, giữ nguyên query và chỉ thay đổi tập Train.

Đối với mỗi split `s`:

`Novelty_s(q) = min D(q,h), h ∈ Train_s`

Song song, query `q` được đánh giá bằng Bao và Balsa đã được huấn luyện trên chính `Train_s`.

Do đó một query sẽ tạo ra ba nhóm quan sát:

`q + Train_LOO → Novelty_LOO + Performance_LOO`

`q + Train_RANDOM → Novelty_RANDOM + Performance_RANDOM`

`q + Train_BASE → Novelty_BASE + Performance_BASE`

Novelty Score không được đưa vào mô hình LQO. Novelty và hiệu năng được tính độc lập, sau đó mới được ghép lại trong bước Data Mining.

### Final Analysis

Bảng kết quả cuối cùng có dạng:

| query_id | split | novelty | model | planning_ms | execution_ms | end_to_end_ms | pg_e2e_ms | speedup |
|---|---|---:|---|---:|---:|---:|---:|---:|
| qX | LOO | ... | Bao | ... | ... | ... | ... | ... |
| qX | LOO | ... | Balsa | ... | ... | ... | ... | ... |
| qX | Random | ... | Bao | ... | ... | ... | ... | ... |
| qX | Random | ... | Balsa | ... | ... | ... | ... | ... |
| qX | BaseQuery | ... | Bao | ... | ... | ... | ... | ... |
| qX | BaseQuery | ... | Balsa | ... | ... | ... | ... | ... |

Phân tích cuối tập trung vào mối quan hệ:

`Training Split → Query Novelty → LQO Performance`

và trả lời hai câu hỏi:

**RQ1.** Cách tổ chức tập Train theo LOO, Random và Base Query ảnh hưởng như thế nào đến Novelty Score của cùng một test query?

**RQ2.** Khi Novelty Score thay đổi theo tập Train, hiệu năng của Bao và Balsa so với PostgreSQL thay đổi như thế nào?