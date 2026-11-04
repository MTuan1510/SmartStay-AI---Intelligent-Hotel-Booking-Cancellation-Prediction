# Kết quả người 3 — LightGBM, E2, E7, SHAP

Chạy ngày 01/11/2026 trên dữ liệu hiện có. Chỉ sử dụng `train.csv` (40.688 dòng, tỷ lệ hủy 24,03%) và `val.csv` (14.400 dòng, tỷ lệ hủy 28,40%); **chưa mở `test.csv`**. Toàn bộ biến đổi đặc trưng nằm trong Pipeline fit trên train. Cấu hình LightGBM: 250 cây, learning rate 0,05, 31 lá, seed 42. Số liệu đầy đủ ở `reports/results/nguoi3.csv`.

| Biến thể / thử nghiệm | ROC-AUC ↑ | PR-AUC ↑ | F1 ở ngưỡng 0,5 ↑ | Brier ↓ |
|---|---:|---:|---:|---:|
| E1 LightGBM gốc | 0,7192 | 0,5458 | 0,2514 | 0,2081 |
| E2 `class_weight=balanced` | 0,7141 | 0,5383 | **0,4299** | 0,1965 |
| E2 SMOTENC, tỷ lệ thiểu số/đa số 0,75 | **0,7723** | **0,5761** | 0,2848 | 0,1907 |

**E2:** Cân bằng lớp không cải thiện đồng thời mọi chỉ số. `class_weight` nâng recall từ 0,150 lên 0,327 và F1 tại ngưỡng cố định 0,5, nhưng ROC-AUC giảm nhẹ. SMOTENC nâng ROC-AUC và PR-AUC, cũng cho Brier thấp nhất trong ba biến thể. Nếu mục tiêu là chọn ngưỡng theo chi phí overbooking, nên dùng xác suất đã kiểm tra hiệu chỉnh và để người 4 thử nhiều ngưỡng; không chọn chỉ theo F1 tại 0,5. SMOTENC nằm trong `imblearn.Pipeline`, chỉ tổng hợp trên train và bảo toàn các cột phân loại đã mã hóa.

**E7:** Để có dữ liệu hiệu chỉnh độc lập, fit một LightGBM mới trên 32.475 đơn có ngày đến trước 01/07/2016, rồi fit sigmoid trên 8.213 đơn còn lại của *train*. So sánh hai bản trên cùng 14.400 đơn val:

| Cùng một mô hình nền | ROC-AUC ↑ | Brier ↓ | Log loss ↓ |
|---|---:|---:|---:|
| Trước hiệu chỉnh | 0,7051 | 0,2099 | 0,8107 |
| Sau hiệu chỉnh sigmoid | 0,7051 | **0,1849** | **0,5668** |

ROC-AUC giữ nguyên vì hiệu chỉnh sigmoid chỉ biến đổi xác suất theo thứ tự đơn điệu. Mô hình E7 dùng ít dữ liệu để fit cây hơn E1 nên không so sánh hai bản như thể cùng mô hình nền. Biểu đồ `figures/nguoi3_e7_calibration.png` cho thấy cải thiện tổng thể nhưng dự đoán ở một số khoảng vẫn lệch đường lý tưởng. Xác suất chưa thể được coi là chính xác tuyệt đối, nhất là khi tỷ lệ hủy thay đổi theo thời gian.

**SHAP:** Trên 500 đơn val được lấy mẫu với seed 42, 5 cột có độ lớn SHAP trung bình cao nhất là `country`, `arrival_date_month`, `agent`, `lead_time`, `market_segment`. Hình và bảng: `figures/nguoi3_shap_importance.png`, `nguoi3_shap_importance.csv`. Các cột chữ được mã hóa thứ tự để phục vụ mô hình cây; SHAP chỉ mô tả hành vi mô hình, không kết luận rằng một cột gây ra hủy đặt phòng. Đặc biệt, ảnh hưởng của tháng và quốc gia cần kiểm tra thêm vì val diễn ra sau train.

## Bàn giao

- Người 4 có thể đọc `models/proba_val_lightgbm_calibrated.npy` theo đúng thứ tự của `val.csv` để chạy E8; so sánh chi phí với các mô hình khác theo cùng hàng dữ liệu. `models/proba_val_lightgbm.npy` là mô hình E1 fit toàn train. Không cộng/trộn xác suất hai bản khi chưa xác định mô hình dùng.
- Mô hình hiệu chỉnh có thể nạp bằng `joblib.load('models/lightgbm_calibrated.joblib')`; cần giữ `src/calibration.py` trong dự án. `predict_proba(X)[:, 1]` trả về xác suất hủy.
- Người 1/2/4 đưa kết quả E1 vào cùng bảng so sánh. Nếu sau này một mô hình khác tốt hơn và muốn hiệu chỉnh E7 cho **hai mô hình đứng đầu**, phải fit lại mô hình đó trên phần train trước 01/07/2016 và hiệu chỉnh trên phần train còn lại, rồi đánh giá trên val. Không dùng nhãn val để fit hiệu chỉnh rồi chấm điểm trên chính toàn bộ val.
- Tập `test` vẫn dành cho đánh giá cuối cùng của cả nhóm. Việc chia theo **ngày đến** chưa bảo đảm đúng thứ tự thời điểm đặt phòng: ngày đến muộn có thể được đặt trước một đơn đến sớm.

Chạy lại notebook `notebooks/04_models_nguoi3.ipynb` bằng **Restart & Run All** sau khi cài `requirements.txt`. Tên người ghi log hiện là `nguoi3`; có thể đổi biến `AUTHOR` trong cell đầu.
