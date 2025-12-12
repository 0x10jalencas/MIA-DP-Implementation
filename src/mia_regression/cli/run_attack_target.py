import argparse, json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


def rf_pred_variance(model, X):
    """
    Compute per-sample prediction mean and variance across RF trees.

    This mirrors the feature construction in run_build_attack_dataset_blackbox:
    we treat variance as an uncertainty signal in addition to the prediction
    and loss-based features.
    """
    tree_preds = np.stack([t.predict(X) for t in model.estimators_], axis=1)
    return tree_preds.mean(axis=1), tree_preds.var(axis=1)


def build_attack_features(y_true, y_pred, pred_var):
    """
    Build attack feature vectors from target model behavior.

    Features:
      - y_pred: predicted quality,
      - abs_err: |y - y_pred|,
      - sq_err:  (y - y_pred)^2,
      - pred_var: RF prediction variance.
    """
    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2
    return np.column_stack([y_pred, abs_err, sq_err, pred_var])


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate trained MIA attack model on target train vs. test points."
    )
    ap.add_argument(
        "--splits",
        default="data/processed/processed_splits.npz",
        help="Processed splits (contains X_t_train, X_t_test, etc.).",
    )
    ap.add_argument(
        "--target_model",
        default="results/runs/target_model.joblib",
        help="Path to trained target model (RandomForestRegressor).",
    )
    ap.add_argument(
        "--attack_model",
        default="results/runs/attack_model.joblib",
        help="Path to trained attack model (LogisticRegression).",
    )
    args = ap.parse_args()

    # Load member (train) and non-member (test) splits for evaluation.
    data = np.load(args.splits, allow_pickle=True)
    X_t_train = data["X_t_train"]
    y_t_train = data["y_t_train"]
    X_t_test = data["X_t_test"]
    y_t_test = data["y_t_test"]

    target = joblib.load(args.target_model)
    attack = joblib.load(args.attack_model)

    # Query target model on members (train) and non-members (test),
    # then build attack features exactly as in the training phase.
    pred_m, var_m = rf_pred_variance(target, X_t_train)
    X_m = build_attack_features(y_t_train, pred_m, var_m)

    pred_n, var_n = rf_pred_variance(target, X_t_test)
    X_n = build_attack_features(y_t_test, pred_n, var_n)

    # Construct membership labels for evaluation: 1 = member, 0 = non-member.
    X_all = np.vstack([X_m, X_n])
    y_membership = np.concatenate(
        [np.ones(X_m.shape[0], dtype=int), np.zeros(X_n.shape[0], dtype=int)]
    )

    # Use the trained attack model to infer membership and compute metrics.
    probs = attack.predict_proba(X_all)[:, 1]
    preds = attack.predict(X_all)

    metrics = {
        "attack_auc": float(roc_auc_score(y_membership, probs)),
        "attack_acc": float(accuracy_score(y_membership, preds)),
        "confusion": confusion_matrix(y_membership, preds).tolist(),
    }

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()