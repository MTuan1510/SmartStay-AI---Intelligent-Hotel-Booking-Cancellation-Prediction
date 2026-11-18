"""So sánh có kiểm tra nhiễu: hai mô hình chênh nhau bao nhiêu, và chênh lệch đó có lớn hơn nhiễu không.

paired_bootstrap lấy mẫu lại CÙNG các dòng của tập đánh giá cho cả hai mô hình (ghép cặp),
rồi cho khoảng tin cậy 95% của chênh lệch. Nếu khoảng này chứa 0 thì chưa đủ bằng chứng
rằng hai cách khác nhau.

Giới hạn: chỉ đo nhiễu do chọn mẫu đánh giá, không đo nhiễu do huấn luyện (seed, chia dữ liệu).
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}


def paired_bootstrap(y_true, proba_a, proba_b, metric: str = "pr_auc",
                     n_boot: int = 500, seed: int = 42) -> dict:
    """Chênh lệch metric(b) - metric(a) trên cùng các dòng, kèm khoảng tin cậy 95%."""
    f = METRICS[metric]
    y, a, b = np.asarray(y_true), np.asarray(proba_a), np.asarray(proba_b)
    rng = np.random.default_rng(seed)
    diff = f(y, b) - f(y, a)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        boots.append(f(yy, b[idx]) - f(yy, a[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": diff, "ci_low": lo, "ci_high": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0)}
