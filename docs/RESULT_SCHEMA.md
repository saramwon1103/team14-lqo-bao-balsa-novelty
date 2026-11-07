# Result Schema

Bao và Balsa cần xuất kết quả về cùng schema tối thiểu.

| Field | Ý nghĩa |
|---|---|
| split_type | LOO / Random / BaseQuery |
| split_id | Mã split |
| query_id | Mã truy vấn |
| family | Họ truy vấn |
| model | Bao / Balsa / PostgreSQL |
| planning_ms | Thời gian lập kế hoạch |
| inference_ms | Thời gian suy luận nếu đo riêng |
| execution_ms | Thời gian thực thi |
| end_to_end_ms | Thời gian đầu cuối |
| status | OK / TIMEOUT / ERROR / NOT_RUN (`NOT_RUN` dùng khi phép đo đó chưa thực sự được thực thi) |
| plan_path | Đường dẫn plan hoặc log |
| model_source | team_trained / author_checkpoint |

Đơn vị thời gian thống nhất: millisecond (ms).
