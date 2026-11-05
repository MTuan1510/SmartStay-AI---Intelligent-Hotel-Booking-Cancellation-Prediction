"""Hai mô hình tham chiếu dùng chung cho các thực nghiệm so sánh dữ liệu (E3, E4, E5, E6).

Cố ý để tham số mặc định, KHÔNG tinh chỉnh: mục đích là so sánh các lựa chọn về dữ liệu,
không phải tìm mô hình tốt nhất. Cả nhóm dùng đúng hai mô hình này để kết quả đối chiếu được.

    model = make_reference("logreg")                       # hoặc "forest"
    model = make_reference("forest", drop=["adr"])         # bỏ thêm cột
    model = make_reference("logreg", extra_numeric=["total_of_special_requests"])
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import SEED
from src.features import build_preprocessor

REFERENCE = ("logreg", "forest")


def make_reference(name: str, drop=(), extra_numeric=()):
    if name == "logreg":
        prep = build_preprocessor("linear", drop=drop, extra_numeric=extra_numeric)
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
    elif name == "forest":
        prep = build_preprocessor("tree", drop=drop, extra_numeric=extra_numeric)
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=SEED)
    else:
        raise ValueError("name phải là 'logreg' hoặc 'forest'")
    return Pipeline([("prep", prep), ("clf", clf)])
