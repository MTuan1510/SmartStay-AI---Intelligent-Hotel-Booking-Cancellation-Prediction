"""Đọc dữ liệu đã chia (train / val / test) — dùng chung cho cả nhóm.

Quy tắc của nhóm: tập kiểm tra (test) chỉ được mở MỘT LẦN ở bước đánh giá cuối.
Vì vậy load_xy("test") báo lỗi trừ khi cố ý truyền allow_test=True.

Cách dùng trong notebook (notebook nằm trong thư mục notebooks/):
    import sys; sys.path.insert(0, "..")
    from src.data import load_xy
    X_train, y_train = load_xy("train")
    X_val, y_val = load_xy("val")
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"

TARGET = "is_canceled"
SEED = 42  # mọi người dùng chung một seed để kết quả so sánh được với nhau


def load_split(name: str, allow_test: bool = False) -> pd.DataFrame:
    """Đọc nguyên một tập (gồm cả cột đích). name: 'train', 'val' hoặc 'test'."""
    if name not in ("train", "val", "test"):
        raise ValueError(f"name phải là 'train', 'val' hoặc 'test', nhận được {name!r}")
    if name == "test" and not allow_test:
        raise PermissionError(
            "Tập kiểm tra chỉ được mở một lần ở bước đánh giá cuối "
            "(notebook cuối, cả nhóm cùng chạy). Dùng 'val' để thử và chỉnh tham số."
        )
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Chạy notebook 01 (làm sạch) rồi 02 (chia tập) trước."
        )
    return pd.read_csv(path)


def load_xy(name: str, allow_test: bool = False):
    """Trả về (X, y): X là các cột đặc trưng, y là cột is_canceled."""
    df = load_split(name, allow_test=allow_test)
    return df.drop(columns=[TARGET]), df[TARGET]
