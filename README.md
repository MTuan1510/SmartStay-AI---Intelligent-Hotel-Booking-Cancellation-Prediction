# Dự đoán hủy đặt phòng khách sạn

## Đề tài

Dự đoán một đơn đặt phòng có bị hủy hay không **tại thời điểm khách đặt**, dựa trên bộ dữ liệu Hotel Booking Demand (86.617 dòng sau làm sạch).

- **Câu hỏi chính:** nếu khách sạn dùng xác suất hủy để quyết định overbooking, thì ngưỡng nào làm tổng chi phí nhỏ nhất? (chi phí là giả định của nhóm, thử nhiều mức)
- **Câu hỏi phụ:** mô hình còn đúng khi dữ liệu thay đổi theo thời gian không (tỷ lệ hủy tăng từ 24% lên 32%)? Các lựa chọn xử lý dữ liệu (dòng trùng, `adr`) ảnh hưởng bao nhiêu?

## Dữ liệu

| Bản | Đường dẫn | Ghi chú |
|---|---|---|
| Gốc | `data/raw/hotel_bookings.csv` | 119.390 dòng x 32 cột, **không bao giờ sửa** |
| Đã làm sạch | `data/interim/hotel_bookings_clean.csv` | 86.617 dòng x 25 cột, tỷ lệ hủy 27,7% |
| Ba tập chia theo thời gian | `data/processed/` | `train.csv`, `val.csv`, `test.csv` (chưa tạo) |
| Cleaning log | `reports/cleaning_log.csv` | 34 dòng: phát hiện, hành động, số dòng, lý do |

Thư mục `data/` **không được đẩy lên Git** (xem `.gitignore`). Sau khi clone, tải bộ *Hotel booking demand* từ Kaggle, giải nén `hotel_bookings.csv` vào `data/raw/`, rồi chạy lại bước làm sạch bên dưới.

## Quy ước làm việc

- Thời điểm dự đoán là lúc đặt phòng, nên các cột chỉ biết sau đó (hoặc làm lộ đáp án) đã bị loại; lý do từng cột nằm trong cleaning log.
- Chia theo thời gian: huấn luyện 2015-07 đến 2016-08, kiểm định 2016-09 đến 2016-12, kiểm tra 2017-01 đến 2017-08. **Tập kiểm tra chỉ chạy một lần ở cuối.**
- Mọi bước học số liệu (chuẩn hóa, mã hóa, điền thiếu, lấy mẫu lại) nằm trong `Pipeline` và chỉ học trên tập huấn luyện.
- Mỗi người làm trên nhánh riêng (`feature/...`), gửi Pull Request vào `main`; ai viết notebook nào thì tự sửa notebook đó.

## Cấu trúc thư mục

```
hotel-cancellation/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # file gốc
│   ├── interim/      # file đã làm sạch và các bản so sánh
│   └── processed/    # train / val / test
├── src/              # code dùng chung (Pipeline, mô hình, đánh giá, chi phí)
├── notebooks/        # EDA và các thực nghiệm E1-E9
└── reports/
    ├── figures/      # biểu đồ cho báo cáo
    └── cleaning_log.csv
```
