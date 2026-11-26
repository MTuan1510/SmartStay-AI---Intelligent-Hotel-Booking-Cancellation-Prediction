import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib, numpy as np
from src.data import load_xy
Xv, _ = load_xy("val")
for m in ["logreg", "tree", "Tung_mlp_v2", "lightgbm_smotenc"]:
    try:
        p = joblib.load(f"models/{m}.joblib").predict_proba(Xv)[:, 1]
        d = np.abs(p - np.load(f"models/proba_val_{m}.npy")).max()
        print(f"{m:18s} {d:.2e}", "OK" if d <= 1e-6 else "LECH")
    except Exception as e:
        print(m, "LOI", type(e).__name__, e)