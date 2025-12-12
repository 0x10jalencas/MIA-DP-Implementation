import argparse
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def rf_pred_variance(model, X):
    tree_preds = np.stack([t.predict(X) for t in model.estimators_], axis=1)
    return tree_preds.mean(axis=1), tree_preds.var(axis=1)


def build_attack_features(y_true, y_pred, pred_var):
    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2
    return np.column_stack([y_pred, abs_err, sq_err, pred_var])


def main():
    ap = argparse.ArgumentParser(description="ROC curve for regression MIA attack.")
    ap.add_argument("--splits", default="data/processed/processed_splits.npz")
    ap.add_argument("--target_model", required=True)
    ap.add_argument("--attack_model", required=True)
    ap.add_argument("--out", default="results/attack_roc.pdf")
    args = ap.parse_args()

    data = np.load(args.splits, allow_pickle=True)
    X_tr, y_tr = data["X_t_train"], data["y_t_train"]
    X_te, y_te = data["X_t_test"], data["y_t_test"]

    target = joblib.load(args.target_model)
    attack = joblib.load(args.attack_model)

    # Build attack features for members (train) and non-members (test)
    pm, vm = rf_pred_variance(target, X_tr)
    X_m = build_attack_features(y_tr, pm, vm)

    pn, vn = rf_pred_variance(target, X_te)
    X_n = build_attack_features(y_te, pn, vn)

    X_all = np.vstack([X_m, X_n])
    y_membership = np.concatenate(
        [np.ones(X_m.shape[0], dtype=int), np.zeros(X_n.shape[0], dtype=int)]
    )

    probs = attack.predict_proba(X_all)[:, 1]
    fpr, tpr, _ = roc_curve(y_membership, probs)
    auc = roc_auc_score(y_membership, probs)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"Attack ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="_nolegend_")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Receiver Operarting Characteristic (ROC) Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize="small")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] saved ROC plot → {out_path}")


if __name__ == "__main__":
    main()
