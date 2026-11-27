"""Tạo reports/results/frozen_thresholds.csv.
Nguong chi chon tren VAL. TUYET DOI KHONG doc test.
Khop src/cost.py: DEFAULT_THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2).
"""
import argparse
import sys
import numpy as np
import pandas as pd

from src.cost import cost_curve  # dung lai, KHONG nhan ban
from src.data import load_xy

RATIOS = (0.5, 1, 2, 3, 5, 10)
RESULTS_PATH = "reports/results/frozen_thresholds.csv"


def best_threshold(y_true, proba, c_walk: float, c_empty: float = 1.0, thresholds=None) -> dict:
    """Duyet luoi tang dan (mac dinh DEFAULT_THRESHOLDS); cost = c_walk*FP + c_empty*FN.
    Hoa -> nguong THAP NHAT (dung idxmin tren luoi tang dan, giong src/cost.py).
    Tra ve dict: threshold, cost, fp, fn, cost_per_booking."""
    curve = cost_curve(y_true, proba, c_walk, c_empty, thresholds)  # curve da sort theo nguong tang dan
    best = curve.loc[curve["cost"].idxmin()]  # hoa -> dong dau tien = nguong thap nhat
    return {
        "threshold": float(best["threshold"]),
        "fp": int(best["fp"]),
        "fn": int(best["fn"]),
        "cost": float(best["cost"]),
        "cost_per_booking": float(best["cost_per_booking"]),
    }


def paired_bootstrap(y_true, proba_a, proba_b, n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap GHEP CAP: lay mau lai CHI SO (khong lay mau rieng A/B).
    Moi resample tinh PR-AUC (average_precision_score) va ROC-AUC cho ca A va B tren CUNG resample.
    Bo qua resample chi co 1 lop (dem n_boot_skipped).
    Tra ve: pr_auc_a, pr_auc_b, roc_auc_a, roc_auc_b (diem tren toan bo val),
    diff_pr_auc (a-b), ci_pr_auc [p2.5, p97.5], diff_roc_auc, ci_roc_auc [p2.5, p97.5],
    frac_boot_a_better_pr, frac_boot_a_better_roc, n_boot_used, n_boot_skipped."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y_true).astype(int)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    assert len(y) == len(proba_a) == len(proba_b), "Do dai y/proba_a/proba_b khong khop"

    pr_auc_a = float(average_precision_score(y, proba_a))
    pr_auc_b = float(average_precision_score(y, proba_b))
    roc_auc_a = float(roc_auc_score(y, proba_a))
    roc_auc_b = float(roc_auc_score(y, proba_b))

    rng = np.random.default_rng(seed)
    n = len(y)
    diffs_pr, diffs_roc = [], []
    a_better_pr = []
    a_better_roc = []
    n_boot_skipped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.min() == ys.max():  # chi co 1 lop -> bo qua
            n_boot_skipped += 1
            continue
        a_pr = average_precision_score(ys, proba_a[idx])
        b_pr = average_precision_score(ys, proba_b[idx])
        a_roc = roc_auc_score(ys, proba_a[idx])
        b_roc = roc_auc_score(ys, proba_b[idx])
        diffs_pr.append(a_pr - b_pr)
        diffs_roc.append(a_roc - b_roc)
        a_better_pr.append(a_pr > b_pr)
        a_better_roc.append(a_roc > b_roc)

    diffs_pr = np.asarray(diffs_pr)
    diffs_roc = np.asarray(diffs_roc)
    n_boot_used = int(len(diffs_pr))

    def _ci(arr):
        if len(arr) == 0:
            return [float("nan"), float("nan")]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "pr_auc_a": pr_auc_a,
        "pr_auc_b": pr_auc_b,
        "roc_auc_a": roc_auc_a,
        "roc_auc_b": roc_auc_b,
        "diff_pr_auc": float(pr_auc_a - pr_auc_b),
        "ci_pr_auc": _ci(diffs_pr),
        "diff_roc_auc": float(roc_auc_a - roc_auc_b),
        "ci_roc_auc": _ci(diffs_roc),
        "frac_boot_a_better_pr": float(np.mean(a_better_pr)) if a_better_pr else float("nan"),
        "frac_boot_a_better_roc": float(np.mean(a_better_roc)) if a_better_roc else float("nan"),
        "n_boot_used": n_boot_used,
        "n_boot_skipped": int(n_boot_skipped),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="vi du: Tung_mlp_v2, logreg, tree, lightgbm_smotenc")
    args = parser.parse_args()

    _, y_val = load_xy("val")  # CHI VAL
    proba = np.load(f"models/proba_val_{args.model}.npy")
    assert len(proba) == len(y_val), f"Do dai xac suat ({len(proba)}) != y_val ({len(y_val)})"

    rows = []
    for r in RATIOS:
        bt = best_threshold(y_val, proba, float(r) * 1.0, 1.0)
        rows.append({"model": args.model, "r": float(r), "threshold": bt["threshold"], "val_cost": bt["cost"]})

    import os
    os.makedirs("reports/results", exist_ok=True)
    df_new = pd.DataFrame(rows, columns=["model", "r", "threshold", "val_cost"])
    try:
        df_old = pd.read_csv(RESULTS_PATH)
        df_old = df_old[df_old["model"] != args.model]  # xoa dong cu cua model nay
        df = pd.concat([df_old, df_new], ignore_index=True)
    except FileNotFoundError:
        df = df_new
    df = df.sort_values(["model", "r"], kind="stable").reset_index(drop=True)
    df.to_csv(RESULTS_PATH, index=False)

    print(df_new.to_string(index=False))

    if args.model == "Tung_mlp_v2":
        row_r2 = df_new[df_new["r"] == 2.0].iloc[0]
        ok = (float(row_r2["threshold"]) == 0.43) and (float(row_r2["val_cost"]) == 3757.0)
        if not ok:
            print(f"THAT BAI: Tung_mlp_v2 r=2 phai la threshold=0.43 / val_cost=3757, "
                  f"thuc te: {row_r2['threshold']} / {row_r2['val_cost']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
