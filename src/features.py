"""Tiền xử lý dùng chung (Pipeline) — chỉ học số liệu trên tập huấn luyện.

build_preprocessor(kind) trả về một ColumnTransformer CHƯA fit. Gắn nó vào đầu Pipeline
rồi fit trên train; khi đó mọi bước học số liệu (điền thiếu, chuẩn hóa, mã hóa) chỉ nhìn
thấy tập huấn luyện, không bị rò rỉ sang val/test.

    pipe = Pipeline([("prep", build_preprocessor("linear")),
                     ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)

Hai kiểu:
  "linear": cho Logistic Regression và MLP   (chuẩn hóa số + one-hot cho chữ)
  "tree"  : cho Decision Tree, Random Forest, XGBoost, LightGBM (số giữ nguyên + mã hóa thứ tự cho chữ)
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

# ---- Phân loại các cột của file sạch (25 cột = 1 cột đích + 12 số + 10 chữ + 2 không dùng) ----
NUMERIC = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "days_in_waiting_list", "adr", "has_company",
]
# Lưu ý: agent là MÃ đại lý (số nhưng không có thứ tự), nên xử lý như cột chữ
CATEGORICAL = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type", "agent",
    "customer_type",
]
# Cột cố ý không đưa vào mô hình (sửa ở đây nếu cả nhóm đồng ý đổi)
EXCLUDED = {
    "arrival_date_year": "tập kiểm tra có năm 2017 mà mô hình chưa từng thấy; nếu dùng, mô hình "
                         "không khái quát được và còn che mất hiện tượng dữ liệu trôi theo thời gian",
    "arrival_date_day_of_month": "ngày trong tháng ít thông tin; tháng (arrival_date_month) đã giữ mùa vụ",
}
TARGET = "is_canceled"


def _to_str(X):
    """Ép mọi cột chữ/mã về chuỗi (để agent dạng số được coi là nhãn, không phải số đo)."""
    return np.asarray(X).astype(str)


def check_columns(X, extra_numeric=()):
    """Báo lỗi nếu X có cột lạ hoặc thiếu cột — tránh việc cột mới bị bỏ qua trong im lặng."""
    known = set(NUMERIC) | set(CATEGORICAL) | set(EXCLUDED) | set(extra_numeric)
    unknown = set(X.columns) - known
    missing = (set(NUMERIC) | set(CATEGORICAL)) - set(X.columns)
    if unknown:
        raise ValueError(f"Cột chưa được phân loại trong features.py: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")


def build_preprocessor(kind: str, drop=(), extra_numeric=(), min_frequency: int = 30):
    """Tạo bộ tiền xử lý chưa fit.

    kind          : "linear" hoặc "tree" (xem docstring đầu file).
    drop          : danh sách cột bỏ thêm. E4 bỏ adr: drop=["adr"]. E6 bỏ hotel: drop=["hotel"].
    extra_numeric : cột số thêm vào, ví dụ ["total_of_special_requests"] cho bản dữ liệu E4.
    min_frequency : (chỉ "linear") giá trị chữ xuất hiện ít hơn ngần này dòng trong tập huấn luyện
                    được gộp thành một nhóm "hiếm" (country, agent có hàng trăm giá trị).
    """
    if kind not in ("linear", "tree"):
        raise ValueError("kind phải là 'linear' hoặc 'tree'")
    drop = set(drop)
    bad = drop - set(NUMERIC) - set(CATEGORICAL) - set(extra_numeric)
    if bad:
        raise ValueError(f"drop chứa cột không có trong danh sách đặc trưng: {sorted(bad)}")

    numeric = [c for c in list(NUMERIC) + list(extra_numeric) if c not in drop]
    categorical = [c for c in CATEGORICAL if c not in drop]

    if kind == "linear":
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
        cat_encoder = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                    min_frequency=min_frequency, sparse_output=False)
    else:
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("to_str", FunctionTransformer(_to_str, feature_names_out="one-to-one")),
                         ("encode", cat_encoder)])

    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
        remainder="drop", verbose_feature_names_out=False)
