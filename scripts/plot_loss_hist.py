import argparse
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def main():
    ap = argparse.ArgumentParser(description="Loss distributions for members vs non-members.")
    ap.add_argument("--splits", default="data/processed/processed_splits.npz")
    ap.add_argument("--target_model", required=True)
    ap.add_argument("--out", default="results/loss_hist.pdf")
    args = ap.parse_args()

    data = np.load(args.splits, allow_pickle=True)
    X_tr, y_tr = data["X_t_train"], data["y_t_train"]
    X_te, y_te = data["X_t_test"], data["y_t_test"]

    model = joblib.load(args.target_model)

    p_tr = model.predict(X_tr)
    p_te = model.predict(X_te)

    abs_err_tr = np.abs(y_tr - p_tr)
    abs_err_te = np.abs(y_te - p_te)

    print("MAE train:", mean_absolute_error(y_tr, p_tr))
    print("MAE test :", mean_absolute_error(y_te, p_te))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(abs_err_tr, bins=30, alpha=0.6, label="Train (members)", density=True)
    ax.hist(abs_err_te, bins=30, alpha=0.6, label="Test (non-members)", density=True)
    ax.set_xlabel("|y - ŷ| (absolute error)")
    ax.set_ylabel("Density")
    ax.set_title("Loss distributions: members vs non-members")
    max_err = max(abs_err_tr.max(), abs_err_te.max())
    ax.set_xlim(0, max_err)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize="small")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] saved loss histogram → {out_path}")


if __name__ == "__main__":
    main()
