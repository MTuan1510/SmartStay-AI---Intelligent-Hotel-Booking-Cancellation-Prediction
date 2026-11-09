"""Chi phí overbooking theo ngưỡng — dùng chung cho E8 (và app Streamlit).

Mô hình chi phí (MỌI con số chi phí đều là GIẢ ĐỊNH của nhóm, không phải số liệu kinh doanh thật):
  - Dự đoán "sẽ hủy" (proba >= ngưỡng)  -> khách sạn cho overbooking phòng đó.
      + Nếu khách vẫn đến (FP): phải chuyển khách đi nơi khác, bồi thường -> chi phí c_walk
  - Dự đoán "không hủy" (proba < ngưỡng) -> giữ phòng cho khách.
      + Nếu khách hủy (FN): phòng bỏ trống, mất doanh thu            -> chi phí c_empty
  - Dự đoán đúng (TP, TN): chi phí 0.

    Cost(t) = c_walk * FP(t) + c_empty * FN(t)

Chỉ có TỶ LỆ r = c_walk / c_empty mới quyết định ngưỡng tối ưu, nên mặc định c_empty = 1 và thử
nhiều giá trị r. Nếu xác suất được hiệu chỉnh tốt, ngưỡng tối ưu về lý thuyết là
    t* = c_walk / (c_walk + c_empty)
(overbooking một đơn có lợi khi (1-p)*c_walk < p*c_empty).

Cách dùng trong notebook (đặt trong notebooks/):
    import sys; sys.path.insert(0, "..")
    from src.evaluate import load_proba
    from src.cost import cost_curve, best_threshold, cost_table
"""
import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def theoretical_threshold(c_walk: float, c_empty: float = 1.0) -> float:
    """Ngưỡng tối ưu lý thuyết khi xác suất đã được hiệu chỉnh."""
    return c_walk / (c_walk + c_empty)


def cost_at_threshold(y_true, proba, threshold: float, c_walk: float, c_empty: float = 1.0) -> dict:
    """Chi phí tại MỘT ngưỡng (dùng để áp ngưỡng chọn trên val vào test)."""
    y = np.asarray(y_true).astype(int)
    over = np.asarray(proba) >= threshold          # dự đoán hủy -> overbooking
    fp = int((over & (y == 0)).sum())              # khách đến nhưng đã overbooking
    fn = int((~over & (y == 1)).sum())             # khách hủy nhưng không overbooking
    cost = c_walk * fp + c_empty * fn
    return {"threshold": threshold, "fp": fp, "fn": fn,
            "cost": cost, "cost_per_booking": cost / len(y)}


def cost_curve(y_true, proba, c_walk: float, c_empty: float = 1.0, thresholds=None) -> pd.DataFrame:
    """Bảng chi phí theo từng ngưỡng: cột threshold, fp, fn, cost, cost_per_booking."""
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    return pd.DataFrame([cost_at_threshold(y_true, proba, t, c_walk, c_empty) for t in thresholds])


def best_threshold(curve: pd.DataFrame) -> pd.Series:
    """Dòng có chi phí nhỏ nhất (nếu hòa thì lấy ngưỡng thấp nhất)."""
    return curve.loc[curve["cost"].idxmin()]


def baseline_costs(y_true, c_walk: float, c_empty: float = 1.0) -> dict:
    """Hai mốc đối chứng không cần mô hình.
    never_overbook : không bao giờ overbooking -> mất c_empty cho mọi đơn hủy.
    always_overbook: overbooking mọi đơn      -> mất c_walk cho mọi đơn khách đến.
    """
    y = np.asarray(y_true).astype(int)
    return {"never_overbook": c_empty * int((y == 1).sum()),
            "always_overbook": c_walk * int((y == 0).sum())}


def cost_table(y_true, proba, ratios=(0.5, 1, 2, 3, 5, 10), c_empty: float = 1.0,
               thresholds=None) -> pd.DataFrame:
    """Mỗi dòng một tỷ lệ r = c_walk / c_empty: ngưỡng lý thuyết, ngưỡng tối ưu, chi phí, mức tiết kiệm.

    saving_vs_never: % chi phí giảm so với không overbooking.
    gap_vs_theory  : ngưỡng tối ưu thực nghiệm trừ ngưỡng lý thuyết (lệch xa => xác suất chưa hiệu chỉnh tốt).
    """
    rows = []
    for r in ratios:
        c_walk = r * c_empty
        curve = cost_curve(y_true, proba, c_walk, c_empty, thresholds)
        b = best_threshold(curve)
        base = baseline_costs(y_true, c_walk, c_empty)
        t_theory = theoretical_threshold(c_walk, c_empty)
        at_half = cost_at_threshold(y_true, proba, 0.5, c_walk, c_empty)["cost"]
        rows.append({
            "r": r, "t_theory": round(t_theory, 3), "t_best": float(b["threshold"]),
            "gap_vs_theory": round(float(b["threshold"]) - t_theory, 3),
            "cost_best": float(b["cost"]), "cost_t0.5": float(at_half),
            "cost_never": base["never_overbook"], "cost_always": base["always_overbook"],
            "saving_vs_never": 1 - float(b["cost"]) / base["never_overbook"],
        })
    return pd.DataFrame(rows)


def cost_table_by_group(y_true, proba, groups, ratio: float, c_empty: float = 1.0,
                        thresholds=None) -> pd.DataFrame:
    """Ngưỡng tối ưu và chi phí riêng từng nhóm (ví dụ groups = X_val['hotel']) tại một tỷ lệ r."""
    y = pd.Series(np.asarray(y_true)); p = pd.Series(np.asarray(proba)); g = pd.Series(np.asarray(groups))
    rows = {}
    for name in sorted(g.unique()):
        m = g == name
        t = cost_table(y[m], p[m], ratios=(ratio,), c_empty=c_empty, thresholds=thresholds).iloc[0]
        rows[name] = {"n": int(m.sum()), "positive_rate": float(y[m].mean()), **t.drop("r").to_dict()}
    return pd.DataFrame(rows).T
