import argparse
import json
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


def rf_pred_variance(model, X):
    """
    Find prediction mean and variance per sample from a RandomForestRegressor.
    """
    import numpy as np

    tree_preds = np.stack([t.predict(X) for t in model.estimators_], axis=1)
    return tree_preds.mean(axis=1), tree_preds.var(axis=1)


def build_attack_features(y_true, y_pred, pred_var):
    """
    Create the same attack features as in the training phase.
    """
    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2
    return np.column_stack([y_pred, abs_err, sq_err, pred_var])


def main():
    # Create argument parser and set arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/processed/processed_splits.npz")
    ap.add_argument("--target_model", default="results/runs/target_model.joblib")
    ap.add_argument("--attack_model", default="results/runs/attack_model.joblib")
    args = ap.parse_args()

    # Load splits and models
    data = np.load(args.splits, allow_pickle=True)
    X_t_train = data["X_t_train"]
    y_t_train = data["y_t_train"]
    X_t_test = data["X_t_test"]
    y_t_test = data["y_t_test"]

    # Set target and attack models
    target = joblib.load(args.target_model)
    attack = joblib.load(args.attack_model)

    # Build member attack features
    pred_m, var_m = rf_pred_variance(target, X_t_train)
    X_m = build_attack_features(y_t_train, pred_m, var_m)

    # Build non-member attack features
    pred_n, var_n = rf_pred_variance(target, X_t_test)
    X_n = build_attack_features(y_t_test, pred_n, var_n)

    # Combine and evaluate attack
    X_all = np.vstack([X_m, X_n])
    y_membership = np.concatenate(
        [np.ones(X_m.shape[0], dtype=int), np.zeros(X_n.shape[0], dtype=int)]
    )

    # Get attack predictions and probabilities
    probs = attack.predict_proba(X_all)[:, 1]
    preds = attack.predict(X_all)

    # Compute metrics
    metrics = {
        "attack_auc": float(roc_auc_score(y_membership, probs)),
        "attack_acc": float(accuracy_score(y_membership, preds)),
        "confusion": confusion_matrix(y_membership, preds).tolist(),
    }

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
