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

Kết quả này được sử dụng cho team evaluation trên ba official Split-1
(`LOO-1`, `Random-1`, `BaseQuery-1`). Đây là một subset của benchmark trong
paper, không phải reproduction đầy đủ Figure 4 với toàn bộ các split.

### Structural Novelty Analysis

Ba official test set không có một tập giao chung đủ để giữ nguyên cùng một
query qua cả ba split. Vì vậy Novelty được phân tích **trong từng split**.

Với mỗi Test Query `q` của split `s`:

`Novelty_s(q) = min D(q,h), h ∈ Train_s`

Song song, chính Test Query đó được đánh giá bằng Bao/Balsa tương ứng với
split và so với PostgreSQL baseline của pipeline đó.

Novelty Score không được đưa vào mô hình LQO. Novelty và hiệu năng được tính
độc lập, sau đó mới được ghép lại trong bước Data Mining.

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

**RQ1.** Ba chiến lược LOO, Random và Base Query tạo ra mức Structural Novelty khác nhau như thế nào?

**RQ2.** Trong từng split, Structural Novelty có liên hệ như thế nào với hiệu năng của Bao/Balsa so với PostgreSQL?