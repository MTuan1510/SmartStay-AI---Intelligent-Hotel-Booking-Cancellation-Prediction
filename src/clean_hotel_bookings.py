"""
Làm sạch bộ dữ liệu Hotel Booking Demand (hotel_bookings.csv)
Mục tiêu: dự đoán khách hàng hủy đặt phòng (cột đích: is_canceled)

Cách chạy (cấu hình mặc định):
    python clean_hotel_bookings.py hotel_bookings.csv hotel_bookings_clean.csv

Tạo các bản để so sánh mô hình (không cần sửa code):
    python clean_hotel_bookings.py in.csv out_keepdup.csv --keep-duplicates
    python clean_hotel_bookings.py in.csv out_noadr.csv   --drop-adr
    python clean_hotel_bookings.py in.csv out_withreq.csv --keep-special-requests

Ngoài file dữ liệu sạch, script ghi cleaning log (<dst>_log.csv, mã hóa utf-8-sig để
Excel đọc đúng tiếng Việt) theo mẫu: Step | Column | Finding | Action | Rows | Why.
Các con số trong cột Finding/Why được tính trực tiếp từ dữ liệu ở thời điểm chạy.
"""
import argparse
import pandas as pd

# ============================== CÔNG TẮC CẤU HÌNH ==============================
DROP_DUPLICATES = True          # True: xóa dòng trùng hoàn toàn
DROP_SPECIAL_REQUESTS = True    # True: loại total_of_special_requests (vùng xám, chọn thận trọng)
DROP_ADR = False                # True: loại adr (vùng xám, mặc định GIỮ để làm thực nghiệm có/không có)
# ==============================================================================

ap = argparse.ArgumentParser()
ap.add_argument("src", nargs="?", default="hotel_bookings.csv")
ap.add_argument("dst", nargs="?", default="hotel_bookings_clean.csv")
ap.add_argument("--keep-duplicates", action="store_true", help="không xóa dòng trùng")
ap.add_argument("--keep-special-requests", action="store_true", help="giữ total_of_special_requests")
ap.add_argument("--drop-adr", action="store_true", help="loại cột adr")
args = ap.parse_args()
if args.keep_duplicates:
    DROP_DUPLICATES = False
if args.keep_special_requests:
    DROP_SPECIAL_REQUESTS = False
if args.drop_adr:
    DROP_ADR = True

src, dst = args.src, args.dst
df = pd.read_csv(src)
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], 1)}

# ------------------------------------------------------------------ cleaning log
log = []


def add(column, finding, action, rows, why):
    """Thêm một dòng vào cleaning log."""
    log.append({"Step": len(log) + 1, "Column": column, "Finding": finding,
                "Action": action, "Rows": rows, "Why": why})


def P(x):
    return f"{x:.1%}"


def stat(sub):
    """Mô tả ngắn một nhóm dòng: số dòng và tỉ lệ hủy."""
    return f"{len(sub)} dòng, {P(sub.is_canceled.mean())} bị hủy" if len(sub) else "0 dòng"


def remove(column, bad, finding, why):
    """Xóa các dòng có bad=True và ghi log."""
    global df
    add(column, finding, "Xóa dòng", int(bad.sum()), why)
    df = df[~bad]


rate_raw = df["is_canceled"].mean()
children_na_raw = int(df["children"].isna().sum())
add("(toàn bộ)", f"{len(df)} dòng x {df.shape[1]} cột; tỉ lệ hủy {P(rate_raw)}",
    "Đọc file gốc, không sửa file raw", len(df),
    "File raw giữ nguyên; mọi thay đổi đi qua script này nên chạy lại được từ đầu")

# =================== A. Chuẩn hóa nhãn (làm TRƯỚC khi xóa trùng) ===================
cn = df["country"].eq("CN")
add("country", f"'CN' ({int(cn.sum())} dòng) và 'CHN' ({int(df['country'].eq('CHN').sum())} dòng) cùng chỉ Trung Quốc",
    "Đổi 'CN' -> 'CHN'", int(cn.sum()),
    "Một quốc gia nhưng có hai mã (ISO-2 lẫn ISO-3); để nguyên thì mô hình coi là hai nước khác nhau")
tmp = df["country"].eq("TMP")
add("country", f"'TMP' ({int(tmp.sum())} dòng) là mã cũ của Đông Timor, mã hiện hành là 'TLS'",
    "Đổi 'TMP' -> 'TLS'", int(tmp.sum()),
    "Dùng mã hiện hành để nhất quán với phần còn lại của cột")
df["country"] = df["country"].replace({"CN": "CHN", "TMP": "TLS"})

und = df["meal"].eq("Undefined")
add("meal", f"'Undefined' ({int(und.sum())} dòng) và 'SC' cùng nghĩa: không kèm gói ăn",
    "Gộp 'Undefined' vào 'SC'", int(und.sum()),
    "Tài liệu mô tả bộ dữ liệu coi hai nhãn này là một; tách ra chỉ tạo thêm một nhóm trùng nghĩa")
df["meal"] = df["meal"].replace("Undefined", "SC")

# ================================ B. Dòng trùng lặp ================================
if DROP_DUPLICATES:
    dup = df.duplicated()
    remove("(tất cả cột)",
           dup,
           f"{int(dup.sum())} dòng trùng hoàn toàn; nhóm trùng hủy {P(df[dup].is_canceled.mean())} "
           f"so với {P(df.is_canceled.mean())} toàn bộ",
           "Không có mã đặt phòng nên khóa kinh doanh là toàn bộ cột; dòng giống hệt có thể là bản ghi lặp "
           "(giữ bản đầu). Rủi ro: có thể là các đặt phòng thật khác nhau, nên có công tắc DROP_DUPLICATES "
           "để so sánh mô hình. Hệ quả: tỉ lệ hủy chung giảm")
else:
    add("(tất cả cột)", f"{int(df.duplicated().sum())} dòng trùng hoàn toàn", "Giữ nguyên (DROP_DUPLICATES=False)", 0,
        "Giữ để so sánh mô hình có/không xóa trùng")

# ====================== C. Bản ghi bất khả (phi logic trong một dòng) ======================
guests = df["adults"] + df["children"].fillna(0) + df["babies"]
bad = guests == 0
remove("adults, children, babies", bad,
       f"Đặt phòng không có khách nào: {stat(df[bad])}",
       "Một đặt phòng không có người ở là phi logic; nhiều khả năng là bản ghi nhập sai")

nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
bad = nights == 0
remove("stays_in_weekend_nights, stays_in_week_nights", bad,
       f"Đặt phòng có tổng số đêm = 0: {stat(df[bad])}",
       "Lưu trú 0 đêm là phi logic; hầu hết không hủy nên nhiều khả năng là lỗi nhập hoặc ghé trong ngày, "
       "không phản ánh hành vi đặt/hủy cần dự đoán")

bad = df["market_segment"].eq("Undefined")
remove("market_segment", bad, f"Giá trị 'Undefined': {stat(df[bad])}",
       "Không xác định được nhóm; chỉ vài dòng (<0.01%) nên xóa không ảnh hưởng phân bố")
bad = df["distribution_channel"].eq("Undefined")
remove("distribution_channel", bad, f"Giá trị 'Undefined': {stat(df[bad])}",
       "Không xác định được kênh; chỉ vài dòng (<0.01%) nên xóa không ảnh hưởng phân bố")

# ====================================== D. Ngoại lệ ======================================
bad = df["adr"] < 0
remove("adr", bad, f"adr âm (min {df['adr'].min():.2f}): {stat(df[bad])}",
       "Giá phòng/đêm âm là giá trị bất khả (lỗi nhập hoặc điều chỉnh kế toán), không phải giá thật")
next_max = df.loc[df["adr"] <= 1000, "adr"].max()
bad = df["adr"] > 1000
remove("adr", bad, f"adr > 1000 (max {df['adr'].max():.0f}): {stat(df[bad])}",
       f"Gấp hơn 10 lần mức cao nhất còn lại ({next_max:.0f}); nhiều khả năng nhập thừa chữ số")

bad = df["adults"] > 10
sub = df[bad]
remove("adults", bad,
       f"adults > 10 (max {int(df['adults'].max())}): {stat(sub)}; "
       f"{int((sub.customer_type == 'Group').sum())} dòng customer_type=Group, adr=0 ở {int((sub.adr == 0).sum())} dòng",
       "Có vẻ là giữ chỗ theo khối cho đoàn (không phải khách đơn lẻ), giá 0 và đều bị hủy; "
       "không đại diện cho hành vi đặt phòng thông thường")
bad = df["children"].fillna(0) > 5
remove("children", bad, f"children > 5: {stat(df[bad])}",
       "Số lượng không thực tế cho một đặt phòng, nhiều khả năng nhập sai; chỉ vài dòng")
bad = df["babies"] > 5
remove("babies", bad, f"babies > 5 (max {int(df['babies'].max())}): {stat(df[bad])}",
       "Số em bé không thực tế cho một đặt phòng, nhiều khả năng nhập sai; chỉ vài dòng")

df = df.copy()

# ============== E. Ngoại lệ/bất thường GIỮ LẠI (ghi lý do để không bị hỏi lại) ==============
n_adr0 = df[(df["adr"] == 0) & (df["market_segment"] != "Complementary")]
comp0 = df[(df["adr"] == 0) & (df["market_segment"] == "Complementary")]
add("adr", f"adr = 0 ngoài nhóm Complementary: {stat(n_adr0)}; nhóm Complementary có adr=0: {stat(comp0)}; toàn bộ: "
           f"{P(df.is_canceled.mean())} hủy",
    "Giữ nguyên", len(n_adr0),
    "Tỉ lệ hủy gần với nhóm phòng tặng và thấp hơn nhiều mức chung, giống phòng miễn phí thật hơn là thiếu dữ liệu "
    "ngẫu nhiên; dữ liệu không đủ để khẳng định là thiếu nên không điền")

hist = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
a = df[(df["is_repeated_guest"] == 0) & (hist > 0)]
b = df[(df["is_repeated_guest"] == 1) & (hist == 0)]
add("is_repeated_guest, previous_cancellations, previous_bookings_not_canceled",
    f"is_repeated_guest=0 nhưng có lịch sử đặt phòng: {stat(a)}; is_repeated_guest=1 nhưng không có lịch sử: {stat(b)}",
    "Giữ nguyên", len(a) + len(b),
    "Không xác định được cột nào sai; xóa sẽ mất nhiều dòng và làm lệch tỉ lệ hủy (nhóm đầu hủy nhiều hơn mức chung). "
    "Các cột lịch sử vẫn giữ để mô hình tự dùng")

a0 = df[(df["adults"] == 0) & ((df["children"].fillna(0) + df["babies"]) > 0)]
add("adults, children, babies", f"adults = 0 nhưng có trẻ em/em bé: {stat(a0)}", "Giữ nguyên", len(a0),
    "Có thể là nhập thiếu người lớn hoặc trẻ ở cùng người lớn khác; chưa đủ cơ sở để coi là sai")
bb = df[df["babies"] > df["adults"]]
add("babies, adults", f"babies > adults: {stat(bb)}", "Giữ nguyên", len(bb),
    "Số dòng rất nhỏ; có thể là cặp song sinh/gia đình đông, không chắc là lỗi")

lt = df["lead_time"]
add("lead_time", f"max {int(lt.max())} ngày; {int((lt > 365).sum())} dòng > 365 ngày (lệch phải)", "Giữ nguyên",
    int((lt > 365).sum()),
    "Đặt trước hơn một năm là có thể xảy ra và là tín hiệu thật; khi huấn luyện dùng RobustScaler hoặc mô hình cây, "
    "nếu cắt ngưỡng (winsorize) phải làm trong Pipeline, chỉ học trên tập train")
tn = nights
add("stays_in_weekend_nights, stays_in_week_nights", f"tổng đêm max {int(tn.max())}; {int((tn > 14).sum())} dòng > 14 đêm",
    "Giữ nguyên", int((tn > 14).sum()),
    "Lưu trú dài là khả dĩ và hiếm; chưa có bằng chứng là lỗi nhập")

# ==================================== F. Giá trị thiếu ====================================
m = df["children"].isna()
if m.sum() > 0:
    add("children", f"{int(m.sum())} giá trị thiếu ({P(m.mean())}); {P((df['children'] == 0).mean())} đặt phòng còn lại có children=0",
        "Điền 0", int(m.sum()),
        "Rất ít dòng; trường hợp phổ biến nhất là không có trẻ em. Điền hằng số (không học từ dữ liệu) nên không gây rò rỉ")
else:
    add("children", f"Dữ liệu gốc có {children_na_raw} giá trị thiếu; sau các bước xóa dòng ở trên không còn dòng nào thiếu",
        "Không cần điền", 0, "Các dòng thiếu đã bị loại ở các bước xóa phía trên; vẫn ép kiểu int ở bước cuối")
df["children"] = df["children"].fillna(0).astype(int)

m = df["country"].isna()
add("country", f"{int(m.sum())} giá trị thiếu ({P(m.mean())}); nhóm này hủy {P(df[m].is_canceled.mean())} so với {P(df.is_canceled.mean())} toàn bộ",
    "Điền 'Unknown'", int(m.sum()),
    "Không có cơ sở điền một nước cụ thể; nhóm 'Unknown' giữ lại thông tin 'quốc gia không được ghi'. "
    "Điền hằng số nên không gây rò rỉ")
df["country"] = df["country"].fillna("Unknown")

m = df["agent"].isna()
add("agent", f"{int(m.sum())} giá trị thiếu ({P(m.mean())}); kênh Direct chiếm {P((df[m].distribution_channel == 'Direct').mean())} "
             f"nhóm thiếu so với {P((df[~m].distribution_channel == 'Direct').mean())} nhóm có agent; "
             f"thiếu: {P(df[m].is_canceled.mean())} hủy, có agent: {P(df[~m].is_canceled.mean())} hủy",
    "Điền 0 (quy ước: không qua đại lý)", int(m.sum()),
    "Thiếu liên quan rõ đến kênh đặt (nhiều khả năng không phải ngẫu nhiên) và hành vi hủy khác hẳn, nên giữ thành một "
    "nhóm riêng thay vì xóa hoặc điền trung vị. agent là mã định danh, khi huấn luyện phải mã hóa như biến phân loại")
df["agent"] = df["agent"].fillna(0).astype(int)

m = df["company"].isna()
add("company", f"{int(m.sum())}/{len(df)} thiếu ({P(m.mean())})", "Thêm has_company (0/1) rồi xóa cột company",
    len(df), "Thiếu hơn 60% nên bỏ cột; giữ cột chỉ báo để còn dấu vết dòng nào có thông tin công ty")
df["has_company"] = df["company"].notna().astype(int)
df = df.drop(columns=["company"])

# ================== G. Cột rò rỉ nhãn (không biết được lúc đặt phòng) ==================
chg = df["reserved_room_type"] != df["assigned_room_type"]
add("reservation_status",
    "Chỉ nhận 'Canceled'/'No-Show' khi is_canceled=1 (khớp "
    f"{P((df.reservation_status.isin(['Canceled', 'No-Show']) == (df.is_canceled == 1)).mean())})",
    "Xóa cột", len(df), "Gần như là đáp án của nhãn; chỉ biết sau khi khách hủy hoặc nhận phòng")
add("reservation_status_date", "Ngày của trạng thái cuối cùng", "Xóa cột", len(df),
    "Chỉ biết sau khi hủy/nhận phòng")
add("assigned_room_type",
    f"Đổi phòng (khác reserved_room_type): hủy {P(df[chg].is_canceled.mean())}; không đổi: {P(df[~chg].is_canceled.mean())}",
    "Xóa cột", len(df), "Phòng thực tế thường chỉ được gán khi khách đến; tỉ lệ hủy rất khác nhau giữa hai nhóm cho thấy "
    "cột mang thông tin của kết quả")
add("booking_changes",
    f"0 lần sửa: hủy {P(df[df.booking_changes == 0].is_canceled.mean())}; từ 1 lần: "
    f"hủy {P(df[df.booking_changes > 0].is_canceled.mean())}",
    "Xóa cột", len(df), "Là số lần sửa đặt phòng tính đến lúc nhận phòng hoặc lúc hủy, nên chỉ biết sau thời điểm đặt")
pk = df[df.required_car_parking_spaces > 0]
add("required_car_parking_spaces", f"Có yêu cầu chỗ đỗ xe: {stat(pk)}", "Xóa cột", len(df),
    "Tỉ lệ hủy đúng bằng 0 cho thấy cột này chỉ được ghi cho khách đã đến, là rò rỉ nhãn")
LEAKAGE_COLS = ["reservation_status", "reservation_status_date", "assigned_room_type",
                "booking_changes", "required_car_parking_spaces"]

# ======================== H. Cột vùng xám (có công tắc) và cột bỏ do thừa ========================
if DROP_SPECIAL_REQUESTS:
    r0 = df[df.total_of_special_requests == 0]
    r3 = df[df.total_of_special_requests >= 3]
    add("total_of_special_requests",
        f"0 yêu cầu: hủy {P(r0.is_canceled.mean())}; từ 3 yêu cầu: hủy {P(r3.is_canceled.mean())}",
        "Xóa cột (DROP_SPECIAL_REQUESTS=True)", len(df),
        "Không xác định được thời điểm ghi nhận; khách hủy sớm cũng ít có cơ hội gửi thêm yêu cầu. "
        "Chọn thận trọng; dùng --keep-special-requests để chạy so sánh có/không")
    LEAKAGE_COLS.append("total_of_special_requests")
if DROP_ADR:
    add("adr", "Giá/đêm có thể được tính lại trong thời gian khách ở", "Xóa cột (DROP_ADR=True)", len(df),
        "Không chắc biết lúc đặt; dùng để chạy so sánh có/không có adr")
    LEAKAGE_COLS.append("adr")
else:
    mean_adr = df.groupby("is_canceled")["adr"].mean()
    add("adr", f"Cột vùng xám; trung bình nhóm hủy {mean_adr[1]:.1f} so với {mean_adr[0]:.1f} nhóm không hủy",
        "Giữ nguyên", len(df),
        "Giá thường đã biết từ lúc đặt. Nhóm hủy có giá cao hơn, không phải hướng mong đợi nếu giá bị đưa về 0 khi hủy; "
        "tuy vậy giá trong dữ liệu là giá cuối cùng nên cần ghi chú. Chạy so sánh có/không có adr bằng --drop-adr")
add("days_in_waiting_list", f"{P((df.days_in_waiting_list == 0).mean())} dòng bằng 0", "Giữ nguyên", len(df),
    "Xác định trước ngày lưu trú nên không rò rỉ; gần như không mang thông tin nhưng giữ cũng không hại")

arr = pd.to_datetime(dict(year=df.arrival_date_year, month=df.arrival_date_month.map(MONTHS),
                          day=df.arrival_date_day_of_month), errors="coerce")
iso_ok = (arr.dt.isocalendar().week.astype("Int64") == df.arrival_date_week_number).fillna(False).mean()
add("arrival_date_week_number", f"Chỉ khớp số tuần ISO của ngày đến ở {P(iso_ok)} dòng; quy tắc đánh số không xác định",
    "Xóa cột", len(df),
    "Thừa vì đã có năm, tháng, ngày đến; không rõ cách đánh số nên không dùng làm đặc trưng")
LEAKAGE_COLS.append("arrival_date_week_number")

df = df.drop(columns=LEAKAGE_COLS)

# ======================================= I. Kiểu dữ liệu =======================================
add("children, agent", "children và agent đọc ra dạng số thực (float) vì có giá trị thiếu", "Ép kiểu int", len(df),
    "Là số đếm và mã định danh nguyên")

df.to_csv(dst, index=False)
pd.DataFrame(log).to_csv(dst.replace(".csv", "_log.csv"), index=False, encoding="utf-8-sig")

# ======================================= Báo cáo nhanh =======================================
print("Cấu hình: DROP_DUPLICATES=%s | DROP_SPECIAL_REQUESTS=%s | DROP_ADR=%s"
      % (DROP_DUPLICATES, DROP_SPECIAL_REQUESTS, DROP_ADR))
print(f"Dữ liệu gốc: {len(pd.read_csv(src))} dòng -> sau làm sạch: {df.shape[0]} dòng x {df.shape[1]} cột")
print(f"Giá trị thiếu còn lại: {int(df.isna().sum().sum())}")
print(f"Tỉ lệ hủy phòng: trước {rate_raw:.1%} -> sau {df['is_canceled'].mean():.1%}")
print(f"Đã ghi {len(log)} dòng cleaning log vào {dst.replace('.csv', '_log.csv')}")
