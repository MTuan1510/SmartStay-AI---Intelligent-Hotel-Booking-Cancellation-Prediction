# Protocol đánh giá cuối trên tập test

**Trạng thái:** ĐÃ CHỐT, phiên bản v1.

Các số dưới đây đều tính trên `train` và `val`; chưa ai mở `test`.

## 1. Phạm vi

- Câu hỏi chính: mô hình dự đoán hủy đặt phòng có giúp quyết định overbooking với chi phí thấp hơn việc không overbook không.
- Test được mở đúng **một lần**, cho đúng danh sách mô hình ở mục 3, bằng đúng các file đã khóa ở mục 2.

## 2. Đối tượng được khóa

| Đối tượng | Giá trị khóa |
|---|---|
| `data/processed/train.csv` | sha256 `4e75d796b7e993bb350d202d428f9e3ad9d746daa3e42704deb0cbb21482329e` |
| `data/processed/val.csv` | sha256 `1fcbee5f639ac363bbbf155235d2cad45436d5e3dd7a73b1c1e422289c3485c1` |
| `data/processed/test.csv` | sha256 `2cac5114304eb69a10b47f34881ded19dfbd5bbcb3d0a9ed107b438c100ae200` (chỉ băm file, chưa đọc nhãn) |
| `models/Tung_mlp_v2.joblib` | sha256 `5090cbc6273c034ad4446a69b9f8e728217a6a956084de504ca248553759b970` |
- Mã băm các mô hình khác (ghi trước khi tag):
  - `models/logreg.joblib` sha256 `abd3ee39fbf5801dfcfb070131525bb46e245881b71346aa8196b24742048f47`
  - `models/tree.joblib` sha256 `cd568c2a9815cd540c4989b347c9d4dc452927a14f8213c12044c91b06b3c29f`
  - `models/lightgbm_smotenc.joblib` sha256 `5ac7db00fd8facd7cca3d11ffb94c886eda0523dfd2078818dce44a500f919c5`

- Phiên bản thư viện tham chiếu (theo metadata E8 của N4): Python 3.11.2, scikit-learn 1.9.1, numpy 2.4.6, pandas 3.1.1, joblib 1.6.0. Phiên bản thực tế khi chạy được notebook tự ghi vào `reports/results/test_run_log.md`.
- Notebook 07 tự đối chiếu mã băm với bảng trên; lệch thì dừng trước khi đọc test. Mã băm của file mô hình khác (logreg, tree, LightGBM) được ghi vào `test_run_log.md` khi chạy.

## 3. Mô hình đưa lên test

Mỗi mô hình dùng **đúng file đã huấn luyện trên train, không huấn luyện lại**. Ngưỡng của mỗi mô hình chọn trên val với r = 2 (hàm `best_threshold` trong `src/final_eval.py`: lưới 0,01 đến 0,99, hòa thì chọn ngưỡng thấp nhất) và khóa trong `reports/results/frozen_thresholds.csv`. Đó là nguồn duy nhất của mọi ngưỡng.

| Vai trò | Mô hình | Phụ trách | File | PR-AUC val | ROC-AUC val | Ngưỡng r = 2 |
|---|---|---|---|---:|---:|---|
| Mốc 1 | Dummy (hằng số = tỷ lệ hủy của train) | N1 | không có file | 0,284 | 0,500 | không dự đoán hủy |
| Baseline | Logistic Regression, C = 0,1 | N1 | `models/logreg.joblib` | 0,573 | 0,784 | 0,67 |
| Tham chiếu | Decision Tree, max_depth = 5, min_samples_leaf = 100 | N1 | `models/tree.joblib` | 0,508 | 0,767 | 0,60 |
| So sánh | LightGBM + SMOTENC 0,75 | N3 | `models/lightgbm_smotenc.joblib` | 0,575 | 0,771 | theo `frozen_thresholds.csv` |
| **Mô hình chính** | **MLP v2** (32,16), alpha 1,0, lr 0,003, batch 256, seed 42 | N4 | `models/Tung_mlp_v2.joblib` | **0,631** | **0,813** | **0,43** (chi phí val 3.757) |

Ghi chú:

- Mô hình chính được chọn **chỉ theo val** (PR-AUC, rồi ROC-AUC, rồi Brier, rồi chi phí, rồi độ đơn giản). MLP hơn mô hình đứng sau khoảng 0,05 PR-AUC, lớn hơn cả độ lệch giữa các seed (khoảng 0,006) lẫn nhiễu bootstrap.
- Điểm val của MLP hơi lạc quan vì cấu hình được chọn trong 33 lần thử trên val, trong khi các mô hình khác được tinh chỉnh ít hơn. Seed 42 cũng là seed tốt nhất trong 5 seed (PR-AUC 0,616 đến 0,631, trung bình 0,622).
- Bản LightGBM hiệu chỉnh E7 (fit lại trên phần train cũ) là một mô hình khác, **không** đưa lên test.
- Không có mô hình nào ngoài bảng này được đưa lên test.
- Bản `lightgbm_smotenc` được tái sinh trên máy nhóm trưởng (Windows, Python 3.14, lightgbm 4.7.0, imbalanced-learn 0.14.2, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6, joblib 1.6.0), lệch PR-AUC 0,0015 và ROC-AUC 0,0018 so với `nguoi3.csv` do khác nền tảng; chạy lặp lại cho cùng kết quả. Ngưỡng r = 2 là 0,58 (chi phí val 3.846), lấy từ `frozen_thresholds.csv`.

## 4. Phương án A (train-only)

- Test được chấm bằng các file đã huấn luyện trên **train**. Không huấn luyện lại trên train + val.
- Lý do: ngưỡng được chọn từ xác suất của mô hình train-only. Huấn luyện lại làm thang xác suất đổi, ngưỡng không còn khớp. Thử 5 seed trên val cho thấy ngưỡng tối ưu dao động từ 0,43 đến 0,70 giữa các seed, và dùng ngưỡng 0,43 cho các seed khác cho chi phí 3.859 đến 4.033 so với 3.757.
- Hệ quả: nếu ai đó huấn luyện lại mô hình, **ngưỡng và protocol này mất hiệu lực** và phải lập bản mới trước khi mở test.
- Giới hạn cần ghi trong báo cáo: bốn tháng val (09 đến 12/2016) không được dùng để huấn luyện dù gần test hơn.

## 5. Quyết định về hiệu chỉnh xác suất (E7) cho MLP

Đã chạy thử trên val với đúng cấu hình MLP v2, 5 seed. Cách làm theo `src/calibration.py`: fit mô hình trên phần train có ngày đến trước 01/07/2016 (32.475 dòng), fit sigmoid trên phần train còn lại (8.213 dòng, tỷ lệ hủy 30,8%), chấm trên val.

| Bản (trung bình 5 seed) | PR-AUC | ROC-AUC | Brier | Xác suất trung bình | Chi phí tốt nhất (r = 2) |
|---|---:|---:|---:|---:|---:|
| Thô, fit toàn train (bản chính) | 0,622 | 0,810 | 0,168 | 17,9% | 3.801 |
| Thô, fit phần train cũ | 0,577 | 0,780 | 0,194 | 12,8% | 3.865 |
| Hiệu chỉnh, fit phần train cũ | 0,577 | 0,780 | 0,165 | 27,5% | 3.867 |

Tỷ lệ hủy thật của val là 28,4%.

**Quyết định:** bản chính **không dùng** bộ hiệu chỉnh. Bản hiệu chỉnh được báo cáo như một phân tích phụ (đã làm xong trên val, không chạy lại trên test).

Lý do:

1. Hiệu chỉnh làm xác suất khớp thực tế hơn (Brier giảm, xác suất trung bình gần 28,4%, ngưỡng tối ưu về khoảng 0,62, gần ngưỡng lý thuyết 2/3), nhưng buộc phải fit lại mô hình trên 80% dữ liệu cũ nhất. PR-AUC giảm khoảng 0,045 và chi phí tốt nhất cao hơn.
2. Hiệu chỉnh đơn điệu không đổi thứ hạng, nên không giảm chi phí tối thiểu trên val. Không viết rằng hiệu chỉnh giúp giảm chi phí.
3. Bản MLP thô đã có ngưỡng chọn xong.

Hệ quả cho demo: **không hiển thị xác suất thô** (trung bình 15,7% so với 28,4% thực tế). Demo hiển thị mức nguy cơ so với ngưỡng 0,43 và ghi chú rằng đây không phải xác suất đã hiệu chỉnh.

## 6. Giả định chi phí (E8)

- Giả định của nhóm, **không phải số liệu thực của khách sạn**: đuổi một khách đã đến (walk) tốn `c_walk`, một phòng trống do khách hủy tốn `c_empty`. r = `c_walk` / `c_empty`.
- Mức chính: **r = 2** (`c_walk` = 2, `c_empty` = 1). Mô hình đơn giản: mỗi đơn bị dự đoán hủy thì overbook đúng một phòng.
- Chi phí so với **không overbook** (chi phí bằng số đơn hủy thật).
- Phân tích độ nhạy: r thuộc {0,5; 1; 2; 3; 5; 10}. Ngưỡng của từng r đã chọn trên val và khóa như sau (MLP v2):

| r | 0,5 | 1 | 2 | 3 | 5 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| Ngưỡng khóa | 0,18 | 0,26 | 0,43 | 0,57 | 0,79 | 0,79 |
| Tiết kiệm trên val so với không overbook | 40,4% | 19,9% | 8,1% | 5,3% | 4,3% | 3,6% |

- Ngưỡng từng r của mọi mô hình nằm trong `reports/results/frozen_thresholds.csv` (cột: model, r, threshold, val_cost), tạo bằng `python -m src.final_eval <tên mô hình>` từ `models/proba_val_<tên>.npy` (chỉ dùng val).
- **Tuyệt đối không chọn lại ngưỡng trên test.** Báo cáo thêm chi phí tại t = 0,5 và t = 2/3 như độ nhạy, không dùng để chọn lại.

## 7. Chỉ số và bảng sẽ báo cáo

Danh sách này đã khóa; bảng thêm sau khi xem test phải ghi là phân tích thăm dò (mục 10).

1. Bảng so sánh mọi mô hình ở mục 3 trên test: PR-AUC (chính), ROC-AUC, precision, recall, F1 tại ngưỡng đã khóa, ma trận nhầm lẫn.
2. Chi phí E8 ở r = 2 của từng mô hình so với không overbook; chi phí của MLP tại t = 0,43, 0,5 và 2/3.
3. Bảng độ nhạy r của MLP với các ngưỡng đã khóa ở mục 6.
4. Chỉ số của mô hình chính **tách theo khách sạn** (City, Resort) và tách theo tháng đến.
5. Khoảng tin cậy bootstrap ghép cặp (`paired_bootstrap`) cho chênh PR-AUC và ROC-AUC giữa MLP và LightGBM (mô hình đứng thứ hai trên val).
6. Xác suất trung bình dự đoán so với tỷ lệ hủy thật trên test (chẩn đoán hiệu chỉnh và drift).

## 8. Quy trình chạy test

1. **Ai chạy:** N4 (Tùng) chạy. N1 (Tuấn) chứng kiến (cùng xem màn hình hoặc xem lại log); nếu N1 vắng thì N3.
2. **Chạy ở đâu:** một notebook duy nhất `notebooks/07_final_test.ipynb`, đổi `FINAL = True`, Restart & Run All. Chỉ notebook này được gọi `load_split("test", allow_test=True)`.
3. **Cổng kiểm tra tự động:** notebook dừng **trước khi đọc test** nếu thiếu tag `protocol-v1`, protocol, `frozen_thresholds.csv` hoặc `src` đã đổi kể từ tag, mã băm ở mục 2 lệch, thiếu file mô hình hoặc ngưỡng khóa, hoặc đã có `test_final.csv`.
4. **Kết quả lưu ở đâu:** notebook tự ghi `reports/results/test_final.csv`, `test_final_by_hotel.csv`, `test_final_by_month.csv`, `test_final_sensitivity.csv`, `test_final_bootstrap.csv`, các file `models/proba_test_<tên mô hình>.npy`, và một mục trong `reports/results/test_run_log.md` (hash commit, hash file, phiên bản thư viện, người chạy, người chứng kiến).
5. **Chạy đúng một lần.** Được chạy lại **chỉ khi** lỗi mang tính kỹ thuật (crash, thiếu file, lỗi đường dẫn) và **chưa ai xem bất kỳ con số nào**. Khi đó xóa file kết quả hỏng, sửa lỗi kỹ thuật, ghi lý do vào `test_run_log.md` rồi chạy lại. Đã xem số thì không chạy lại.
6. **Sau khi chạy:** không đổi mô hình, ngưỡng, chi phí hay danh sách bảng. Báo cáo kết quả như đã chạy, dù xấu.

## 9. Quy ước đặt tên và repo

- Mẫu: `<Người phụ trách>_<thuật toán>_<phiên bản>`, ví dụ `Tung_mlp_v2`. Áp dụng cho mọi file mới; các file cũ giữ nguyên tên.
- Trong log, cột `model` ghi đúng tên file (hiện `mlp_v2` và `Tung_mlp_v2` đang lẫn nhau; từ nay dùng `Tung_mlp_v2`).
- Không đưa `.joblib` lên Git trừ ngoại lệ ghi rõ; gửi riêng hoặc ghi hash. File `.npy` xác suất nhẹ, được phép lên Git.
- `src/cost.py` phải có trong repo. Notebook 07 dùng `src/final_eval.py`; hai bên phải cho cùng kết quả (kiểm tra: `Tung_mlp_v2`, r = 2 cho ngưỡng 0,43 và chi phí 3.757 trên val).

## 10. Sau khi xem test

- Mọi phân tích thêm sau khi xem test được gắn nhãn "thăm dò (post-hoc)" và không được dùng để đổi kết luận chính.
- Tỷ lệ hủy test có thể khác val (drift theo thời gian). Nếu lệch, báo cáo đó là phát hiện, không chỉnh lại ngưỡng.
- Demo Streamlit dùng đúng file và ngưỡng đã khóa (mục 5 quy định cách hiển thị); chỉ dựng sau khi có tag.

## 11. Giới hạn đã biết (ghi sẵn vào báo cáo)

- Chi phí là giả định của nhóm (r = 2) và mô hình overbook một phòng cho mỗi đơn dự đoán hủy; lợi ích ở r = 2 chỉ khoảng 8% trên val. Ở ngưỡng 0,43, MLP có precision khoảng 0,77 nhưng recall chỉ khoảng 0,20.
- Điểm val của MLP lạc quan do chọn cấu hình trên val và do seed 42 là seed tốt nhất; ngưỡng 0,43 gắn với đúng một file huấn luyện.
- Chênh chi phí giữa t = 0,43 và t = 0,5 nằm trong khoảng nhiễu bootstrap (khoảng tin cậy 95% là [1; 110] đơn vị trên val).
- Xác suất MLP thô thấp hơn thực tế ở cả hai khách sạn (City 18,9% so với 31,9%; Resort 9,8% so với 22,1% trên val); không dùng như xác suất đã hiệu chỉnh.
- Dòng trùng bị xóa nhưng dữ liệu không có mã đặt phòng nên không biết đó là bản ghi lặp hay đơn thật; cột `total_of_special_requests` bị loại vì rủi ro rò rỉ (E3, E4).
- Chia theo ngày đến chưa bảo đảm đúng thứ tự thời điểm đặt phòng; val chỉ gồm bốn tháng.

## 12. Phân công và xác nhận


| Thành viên | Phần phụ trách | Việc theo protocol |
|---|---|---|
| N1 (Tuấn) | Dummy, LogReg, Tree | Commit `reports/results/tuan.csv`, `proba_val_logreg.npy`, `proba_val_tree.npy`; chứng kiến lần chạy test |
| N2 | E5, E6, E9 | Không có việc riêng trong protocol; chuẩn bị bảng và hình theo mục 7 |
| N3 | LightGBM, E7, SHAP | Đặt `models/lightgbm_smotenc.joblib` và `proba_val_lightgbm_smotenc.npy`, chạy `python -m src.final_eval lightgbm_smotenc`, commit `frozen_thresholds.csv`;|
| N4 (Tùng) | MLP, E8 | Commit `src/cost.py`, `src/final_eval.py`, `notebooks/07_final_test.ipynb`; chạy test |

## 13. Thứ tự thực hiện

1. N1, N3, N4 commit các file ở mục 12.
2. Nhóm trưởng chạy `sha256sum` trên 4 file ở mục 2 và so với mục 2; lệch thì dừng.
3. Nhóm trưởng commit `reports/final_protocol.md`, rồi `git tag protocol-v1` và `git push --tags`.
4. N4 chạy test (mục 8), N1 chứng kiến.
5. Dựng demo Streamlit và viết phần kết quả sau khi có kết quả test.