"""Chia dữ liệu theo thời gian trong bộ nhớ — dùng cho các thực nghiệm so sánh bản dữ liệu (E3, E4...).

Ranh giới PHẢI khớp với notebook 02_split_data (huấn luyện 2015-07..2016-08,
kiểm định 2016-09..2016-12, kiểm tra 2017-01..2017-08), để các bản dữ liệu khác nhau
được chia cùng một cách và so sánh được với nhau.

Mặc định chỉ trả về train và val. Tập kiểm tra chỉ mở ở bước đánh giá cuối.
"""
import pandas as pd

RANGES = {"train": (201507, 201608), "val": (201609, 201612), "test": (201701, 201708)}
_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], 1)}


def time_split(df: pd.DataFrame, include_test: bool = False) -> dict:
    """Trả về {"train": ..., "val": ...} (thêm "test" nếu include_test=True), index đã reset."""
    arrival = pd.to_datetime(dict(year=df["arrival_date_year"],
                                  month=df["arrival_date_month"].map(_MONTHS),
                                  day=df["arrival_date_day_of_month"]))
    ym = arrival.dt.year * 100 + arrival.dt.month
    masks = {name: ym.between(lo, hi) for name, (lo, hi) in RANGES.items()}
    covered = sum(m.sum() for m in masks.values())
    assert covered == len(df), f"{len(df) - covered} dòng không thuộc tập nào"
    names = ["train", "val", "test"] if include_test else ["train", "val"]
    return {n: df[masks[n]].reset_index(drop=True) for n in names}
