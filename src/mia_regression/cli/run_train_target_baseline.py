import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Base project directory
BASE_DIR = Path(__file__).resolve().parents[3]

# File with processed data splits
SPLITS = BASE_DIR / "data/processed/processed_splits.npz"

# Trained models and results directory
RUN_DIR = BASE_DIR / "results/runs"


def evalm(model, X, y):
    """
    Find standard regfression metrics for a trained dataset.


    Returns a dict containing 3 metrics cased as floats:
        - rmse: root mean squared error
        - mae: mean absolute error
        - r2:  coefficient of determination
    """
    p = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    r2 = float(r2_score(y, p))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    # Setup argument parser and parse args
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load train/val/test splits for the target_model
    data = np.load(SPLITS, allow_pickle=True)
    X_tr, y_tr = data["X_t_train"], data["y_t_train"]
    X_va, y_va = data["X_t_val"], data["y_t_val"]
    X_te, y_te = data["X_t_test"], data["y_t_test"]

    # Train a Random Forest Regressor as the target model
    # Pass in hyperparameters from args
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
    ).fit(X_tr, y_tr)

    # Evaluate on train, val, test splits
    metrics = {
        "train": evalm(rf, X_tr, y_tr),
        "val": evalm(rf, X_va, y_va),
        "test": evalm(rf, X_te, y_te),
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "seed": args.seed,
        },
    }

    # Get an output with a timestamp and save it under RUN_DIR
    ts_dir = RUN_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")
    ts_dir.mkdir(parents=True, exist_ok=True)

    # Save the trained model
    joblib.dump(rf, ts_dir / "target_model.joblib")

    # Save the metrics and hyperparameters
    (ts_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Print out the metrics to the console
    print(json.dumps(metrics, indent=2))
    print(f"[OK] saved to {ts_dir}")


if __name__ == "__main__":
    main()

