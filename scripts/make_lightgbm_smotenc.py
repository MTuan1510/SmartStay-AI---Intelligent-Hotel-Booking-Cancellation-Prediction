"""Tái tạo models/lightgbm_smotenc.joblib + models/proba_val_lightgbm_smotenc.npy.

Cấu hình COPY NGUYÊN từ notebooks/04_models_nguoi3.ipynb (E2, SMOTENC 0,75, chỉ train).
CHỈ đọc train và val, KHÔNG đọc test. Chạy từ thư mục gốc repo:

    python scripts/make_lightgbm_smotenc.py

Sau đó:  python -m src.final_eval lightgbm_smotenc
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import REPO_ROOT, SEED, load_xy
from src.evaluate import save_proba
from src.features import CATEGORICAL, NUMERIC, build_preprocessor, check_columns


def classifier(**kwargs):
    return LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31,
                          max_depth=-1, random_state=SEED, n_jobs=2,
                          verbosity=-1, **kwargs)


X_train, y_train = load_xy("train")
X_val, y_val = load_xy("val")
check_columns(X_train)
check_columns(X_val)

cat_indices = list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL)))
model = ImbPipeline([("prep", build_preprocessor("tree")),
                     ("sample", SMOTENC(categorical_features=cat_indices,
                                        sampling_strategy=0.75, random_state=SEED, k_neighbors=5)),
                     ("clf", classifier())])
model.fit(X_train, y_train)
p = model.predict_proba(X_val)[:, 1]

pr, roc = average_precision_score(y_val, p), roc_auc_score(y_val, p)
print(f"val PR-AUC = {pr:.4f} | ROC-AUC = {roc:.4f}")

# Đối chiếu với dòng E2 SMOTENC trong reports/results/nguoi3.csv (PR-AUC 0,5761 / ROC-AUC 0,7723)
ref = pd.read_csv(REPO_ROOT / "reports" / "results" / "nguoi3.csv")
ref = ref[ref.note.str.startswith("SMOTENC_ratio=0.75")].iloc[-1]
d_pr, d_roc = abs(pr - ref.pr_auc), abs(roc - ref.roc_auc)
print(f"lech so voi nguoi3.csv: PR-AUC {d_pr:.2e}, ROC-AUC {d_roc:.2e}")
if d_pr > 5e-3 or d_roc > 5e-3:
    sys.exit("DUNG: khong tai sinh duoc ket qua cua N3 — khong luu file")

joblib.dump(model, REPO_ROOT / "models" / "lightgbm_smotenc.joblib")
save_proba("lightgbm_smotenc", p)

# Kiem tra nap lai (val) — thu tu dong va gia tri phai khop
re = joblib.load(REPO_ROOT / "models" / "lightgbm_smotenc.joblib").predict_proba(X_val)[:, 1]
print("max_abs_diff sau khi nap lai:", float(np.abs(re - np.load(REPO_ROOT / "models" / "proba_val_lightgbm_smotenc.npy")).max()))
print("OK — da luu models/lightgbm_smotenc.joblib va models/proba_val_lightgbm_smotenc.npy")
