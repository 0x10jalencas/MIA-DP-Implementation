import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

from art.estimators.regression.scikitlearn import ScikitlearnRegressor
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox


def main():
    ap = argparse.ArgumentParser(
        description="Train and evaluate an ART membership inference attack on the RF regressor."
    )
    ap.add_argument(
        "--splits",
        default="data/processed/processed_splits.npz",
        help="NPZ produced by run_prepare_data.py.",
    )
    ap.add_argument(
        "--target_model",
        required=True,
        help="Path to trained RandomForestRegressor (target_model.joblib).",
    )
    ap.add_argument(
        "--outdir",
        default="results/runs",
        help="Directory to save ART attack model and metrics.",
    )
    ap.add_argument(
        "--attack_model_type",
        default="rf",
        help=(
            "ART attack_model_type: nn, rf, gb, lr, dt, knn, svm. "
            "We use 'rf' by default for tabular data."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--save_roc",
        action="store_true",
        help="If set, also save an ROC curve figure for the ART attack.",
    )
    ap.add_argument(
        "--roc_path",
        default=None,
        help="Optional explicit path for ROC figure (e.g., graphs/attack_roc_art.pdf).",
    )
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    # 1) Load splits and target regressor
    data = np.load(args.splits, allow_pickle=True)
    X_tr, y_tr = data["X_t_train"], data["y_t_train"]       # members for training
    X_te, y_te = data["X_t_test"], data["y_t_test"]         # non-members for evaluation
    X_shadow, y_shadow = data["X_shadow"], data["y_shadow"] # non-members for training ART attack

    target = joblib.load(args.target_model)

    # Wrap the scikit-learn regressor in an ART estimator.
    # ART will use its loss interface (MSE) for regression MIAs.
    art_reg = ScikitlearnRegressor(model=target)

    # 2) Configure and train the ART membership inference attack
    # For regressors, ART requires input_type="loss" (documentation explicitly says so).
    attack = MembershipInferenceBlackBox(
        estimator=art_reg,
        input_type="loss",
        attack_model_type=args.attack_model_type,
    )

    # Train on members (X_tr, y_tr) vs non-members (X_shadow, y_shadow).
    # ART will internally compute losses and learn a binary classifier.
    attack.fit(
        x=X_tr,
        y=y_tr,
        test_x=X_shadow,
        test_y=y_shadow,
    )

    # 3) Evaluate ART attack on target train vs test
    # Members: training points; Non-members: test points.
    # Ask ART for membership probabilities; if outputs are 2D (n,2),
    # we take the probability of the "member" class (index 1).
    probs_tr = attack.infer(X_tr, y_tr, probabilities=True)
    probs_te = attack.infer(X_te, y_te, probabilities=True)

    probs_tr = np.asarray(probs_tr)
    probs_te = np.asarray(probs_te)

    if probs_tr.ndim == 2 and probs_tr.shape[1] == 2:
        scores_tr = probs_tr[:, 1]
        scores_te = probs_te[:, 1]
    else:
        # Fallback: already a 1D score per sample.
        scores_tr = probs_tr.ravel()
        scores_te = probs_te.ravel()

    y_membership = np.concatenate(
        [np.ones_like(y_tr, dtype=int), np.zeros_like(y_te, dtype=int)]
    )
    scores = np.concatenate([scores_tr, scores_te])

    # Threshold at 0.5 for hard predictions.
    preds = (scores >= 0.5).astype(int)

    attack_auc = float(roc_auc_score(y_membership, scores))
    attack_acc = float(accuracy_score(y_membership, preds))
    cm = confusion_matrix(y_membership, preds).tolist()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "attack_type": "ART_MembershipInferenceBlackBox",
        "attack_model_type": args.attack_model_type,
        "auc": attack_auc,
        "acc": attack_acc,
        "confusion": cm,
        "n_members_eval": int(y_tr.shape[0]),
        "n_nonmembers_eval": int(y_te.shape[0]),
        "seed": int(args.seed),
        "splits": str(args.splits),
        "target_model": str(args.target_model),
    }

    # Save ART attack object and JSON metrics.
    joblib.dump(attack, outdir / "attack_art_model.joblib")
    (outdir / "attack_art_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))

    if args.save_roc:
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_membership, scores)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"ART MIA (AUC = {attack_auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC = 0.5)")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        roc_path = Path(args.roc_path) if args.roc_path is not None else (outdir / "attack_roc_art.pdf")
        roc_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(roc_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved ART ROC plot => {roc_path}")


if __name__ == "__main__":
    main()
