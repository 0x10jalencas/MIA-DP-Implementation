import argparse, json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def rf_pred_variance(model: RandomForestRegressor, X):
    """
    Compute per-sample prediction mean and variance across trees.

    In our RF-based target, this variance is a model
    confidence/uncertainty and becomes an additional attack feature
    (beyond prediction and loss).
    """
    tree_preds = np.stack([t.predict(X) for t in model.estimators_], axis=1)
    pred_mean = tree_preds.mean(axis=1)
    pred_var = tree_preds.var(axis=1)
    return pred_mean, pred_var


def build_attack_features(y_true, y_pred, pred_var):
    """
    Construct attack features from the black-box behavior of the target model.

    Following the loss-based MIA literature, we use:
      - y_pred: predicted quality score,
      - abs_err: |y - y_pred|,
      - sq_err:  (y - y_pred)^2,
      - pred_var: RF prediction variance as an uncertainty signal.

    These are “observable” quantities for an adversary with
    black-box access and knowledge of true labels in an offline evaluation.
    """
    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2
    X_attack = np.column_stack([y_pred, abs_err, sq_err, pred_var])
    return X_attack


def main():
    ap = argparse.ArgumentParser(
        description="Build attack dataset by querying the target regressor (black-box MIA)."
    )
    ap.add_argument(
        "--splits",
        type=str,
        required=True,
        help="Path to processed_splits.npz (contains member and non-member splits).",
    )
    ap.add_argument(
        "--target_model",
        type=str,
        required=True,
        help="Path to trained target_model.joblib.",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output npz with attack dataset (X_attack, y_attack).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling / balancing.",
    )
    ap.add_argument(
        "--balance",
        type=lambda x: str(x).lower() == 'true',
        default=True,
        help=(
            "If True, subsample to equal numbers of members and non-members, "
            "matching the standard 50/50 evaluation prior (e.g., Jayaraman et al.)."
        ),
    )
    args = ap.parse_args()

    splits_path = Path(args.splits)
    assert splits_path.exists(), f"Missing splits npz at {splits_path}"
    data = np.load(splits_path, allow_pickle=True)

    # Members (training points for the target) and non-members (hold-out pool)
    # This mirrors the walk-through: X_t_train = members, X_shadow = non-members.
    X_m, y_m = data["X_t_train"], data["y_t_train"]
    X_n, y_n = data["X_shadow"], data["y_shadow"]

    # Load trained target regressor (black-box model under attack).
    model = joblib.load(args.target_model)
    assert isinstance(model, RandomForestRegressor), "Expected RF target model."

    # Query the target model on members and non-members.
    # In our paper, this corresponds to the adversary querying the black-box.
    pm, vm = rf_pred_variance(model, X_m)  # members
    pn, vn = rf_pred_variance(model, X_n)  # non-members

    # Build feature vectors summarizing the model's behavior on each record.
    Xm_attack = build_attack_features(y_m, pm, vm)
    Xn_attack = build_attack_features(y_n, pn, vn)

    # Membership labels: 1 = member (training point), 0 = non-member (hold-out).
    ym_attack = np.ones(Xm_attack.shape[0], dtype=np.int32)
    yn_attack = np.zeros(Xn_attack.shape[0], dtype=np.int32)

    rng = np.random.RandomState(args.seed)

    # Balancing where we construct a 50/50 mixture of members and non-members.
    # This matches the standard MIA evaluation game with equal priors on membership.
    if args.balance:
        n_m = Xm_attack.shape[0]
        n_n = Xn_attack.shape[0]
        n_bal = min(n_m, n_n)
        m_idx = rng.choice(n_m, size=n_bal, replace=False)
        n_idx = rng.choice(n_n, size=n_bal, replace=False)
        Xm_attack, ym_attack = Xm_attack[m_idx], ym_attack[m_idx]
        Xn_attack, yn_attack = Xn_attack[n_idx], yn_attack[n_idx]

    # Concatenate member and non-member attack samples.
    X_attack = np.vstack([Xm_attack, Xn_attack])
    y_attack = np.concatenate([ym_attack, yn_attack])

    # Shuffle the combined attack dataset so membership labels are not ordered.
    idx = rng.permutation(X_attack.shape[0])
    X_attack = X_attack[idx]
    y_attack = y_attack[idx]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_attack=X_attack,
        y_attack=y_attack,
        feature_names=np.array(
            ["y_pred", "abs_error", "sq_error", "pred_var"], dtype=object
        ),
    )

    meta = {
        "rows_attack": int(X_attack.shape[0]),
        "dims_attack": int(X_attack.shape[1]),
        "balance": bool(args.balance),
        "seed": args.seed,
        "source": {
            "members": int(Xm_attack.shape[0]),
            "non_members": int(Xn_attack.shape[0]),
        },
        "splits_path": str(splits_path),
        "target_model": str(args.target_model),
    }
    print(json.dumps(meta, indent=2))
    print(f"[OK] saved attack dataset => {out_path}")


if __name__ == "__main__":
    main()
