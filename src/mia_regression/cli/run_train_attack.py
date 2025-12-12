import argparse, json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


def main():
    ap = argparse.ArgumentParser(
        description="Train membership inference attack model on regression attack features."
    )
    ap.add_argument(
        "--attack_npz",
        default="data/processed/attack_dataset_regression.npz",
        help="NPZ file produced by run_build_attack_dataset_blackbox.py.",
    )
    ap.add_argument(
        "--outdir",
        default="results/runs",
        help="Directory to save attack_model.joblib and attack_metrics.json.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load attack dataset constructed from target model behavior.
    # X_attack: [y_pred, |y - y_pred|, (y - y_pred)^2, pred_var]
    # y_attack: membership labels (1 = member, 0 = non-member).
    data = np.load(args.attack_npz, allow_pickle=True)
    X_attack = data["X_attack"]
    y_attack = data["y_attack"]

    # Shuffle for randomness / i.i.d. assumption in train/test split.
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(y_attack))
    X_attack = X_attack[idx]
    y_attack = y_attack[idx]

    # Split attack dataset into training and test parts (80/20).
    # This corresponds to training the attack model and then evaluating
    # its generalization to new membership queries.
    split = int(0.8 * len(y_attack))
    X_tr, X_te = X_attack[:split], X_attack[split:]
    y_tr, y_te = y_attack[:split], y_attack[split:]

    # Logistic regression is a standard, lightweight choice for learned MIAs.
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)

    # Evaluate attack performance: AUC and accuracy on the held-out portion.
    probs = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)
    metrics = {
        "auc": float(roc_auc_score(y_te, probs)),
        "acc": float(accuracy_score(y_te, preds)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, outdir / "attack_model.joblib")
    with open(outdir / "attack_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()