"""Đánh giá và ghi kết quả — dùng chung cho cả nhóm.

Mỗi người chạy xong một mô hình thì làm hai việc:
  1. log_result(...)  -> thêm một dòng vào reports/results/<tên bạn>.csv
  2. save_proba(...)  -> lưu xác suất dự đoán trên tập val vào models/proba_val_<mô hình>.npy
Nhờ đó người làm E7 (hiệu chỉnh), E8 (chi phí), E9 (SHAP, phân tích lỗi) lấy được xác suất của
bất kỳ mô hình nào mà không phải chờ ai.
"""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.data import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "reports" / "results"
MODELS_DIR = REPO_ROOT / "models"


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Các chỉ số chính tại một ngưỡng. proba là xác suất hủy (cột 1 của predict_proba)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        roc, pr = roc_auc_score(y_true, proba), average_precision_score(y_true, proba)
    except ValueError:  # nhóm chỉ có một lớp
        roc = pr = float("nan")
    return {
        "n": len(y_true), "positive_rate": y_true.mean(), "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc, "pr_auc": pr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_by_group(y_true, proba, groups, threshold: float = 0.5, min_n: int = 30) -> pd.DataFrame:
    """Chỉ số tách theo nhóm (ví dụ groups = X_val['hotel']). Bỏ nhóm có ít hơn min_n dòng."""
    y_true = pd.Series(np.asarray(y_true))
    proba = pd.Series(np.asarray(proba))
    groups = pd.Series(np.asarray(groups))
    rows = {}
    for g in sorted(groups.unique()):
        m = groups == g
        if m.sum() >= min_n:
            rows[g] = evaluate(y_true[m], proba[m], threshold)
    return pd.DataFrame(rows).T


def log_result(author: str, experiment: str, model: str, split: str, metrics: dict,
               note: str = "") -> None:
    """Ghi một dòng kết quả vào reports/results/<author>.csv.

    Chạy lại cùng (experiment, model, split, note) thì dòng cũ được thay, không bị lặp.
    experiment: "E1", "E2"...; note: mô tả ngắn, ví dụ "class_weight=balanced".
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{author}.csv"
    row = {"experiment": experiment, "model": model, "split": split, "note": note,
           **metrics, "time": datetime.now().strftime("%Y-%m-%d %H:%M")}
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        key = ["experiment", "model", "split", "note"]
        same = (old[key].astype(str).values == new[key].astype(str).values[0]).all(axis=1)
        new = pd.concat([old[~same], new], ignore_index=True)
    new.to_csv(path, index=False)


def save_proba(model: str, proba, split: str = "val") -> None:
    """Lưu xác suất hủy dự đoán, theo đúng thứ tự dòng của file val.csv."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / f"proba_{split}_{model}.npy", np.asarray(proba, dtype="float32"))


def load_proba(model: str, split: str = "val") -> np.ndarray:
    return np.load(MODELS_DIR / f"proba_{split}_{model}.npy")
