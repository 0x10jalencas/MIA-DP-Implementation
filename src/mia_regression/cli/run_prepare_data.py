import argparse, json, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_clean(csv_path: str) -> pd.DataFrame:
    """
    Load and lightly clean the UCI white wine quality dataset.

    The original file uses ';' as a separator and double quotes around headers.
    We drop rows with missing values and exact duplicates (Humphries et.al)
    """
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip().strip('"') for c in df.columns]
    # Drop any rows with missing values (as per our Walkthrough section)
    df = df.dropna()
    # Remove exact duplicate records to avoid false membership signals (Humphries et.al)
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def build_preprocessor(feature_cols):
    """
    Standardize all continuous features to zero mean and unit variance.
    All wine attributes are numeric, so we just apply a StandardScaler (Zhao).
    """
    scaler = StandardScaler()
    pre = Pipeline(steps=[("scaler", scaler)])
    return pre

def main():
    ap = argparse.ArgumentParser(
        description="Prepare processed arrays and saved splits for UCI Wine Quality (white)."
    )
    ap.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to winequality-white.csv",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory to save processed_splits.npz",
    )
    ap.add_argument(
        "--runs_dir",
        type=str,
        required=True,
        help="Directory to save run metadata and preprocessor",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits",
    )
    ap.add_argument(
        "--shadow_pool_frac",
        type=float,
        default=0.4,
        help="Fraction of data reserved as hold-out pool (non-members).",
    )
    ap.add_argument(
        "--log_target",
        action="store_true",
        help="Apply log1p transform to target if desired (not typical for wine quality).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.runs_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load and clean data
    df = read_clean(args.csv)
    # Target is wine quality (integer 0–10)
    assert "quality" in df.columns, "Expected 'quality' column in wine dataset."
    y = df["quality"].to_numpy(dtype=float)
    if args.log_target:
        y = np.log1p(y)
    # Features are all columns except 'quality'
    feature_cols = [c for c in df.columns if c != "quality"]
    X_df = df[feature_cols].copy()

    # 2) Build and fit preprocessor (StandardScaler)
    pre = build_preprocessor(feature_cols)
    X = pre.fit_transform(X_df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    feature_names = feature_cols
    
    # 3) Split into target pool (for train/val/test) and shadow pool (hold-out non-members)
    X_target, X_shadow, y_target, y_shadow = train_test_split(
        X, y, test_size=args.shadow_pool_frac, random_state=args.seed
    )
    # Within target pool: train / val / test for the regression model
    X_t_train, X_t_temp, y_t_train, y_t_temp = train_test_split(
        X_target, y_target, test_size=0.4, random_state=args.seed
    )
    X_t_val, X_t_test, y_t_val, y_t_test = train_test_split(
        X_t_temp, y_t_temp, test_size=0.5, random_state=args.seed
    )

    # 4) Save splits
    np.savez_compressed(
        outdir / "processed_splits.npz",
        X_t_train=X_t_train,
        y_t_train=y_t_train,
        X_t_val=X_t_val,
        y_t_val=y_t_val,
        X_t_test=X_t_test,
        y_t_test=y_t_test,
        X_shadow=X_shadow,
        y_shadow=y_shadow,
        feature_names=np.array(feature_names, dtype=object),
        log_target=np.array([args.log_target]),
    )

    # 5) Save preprocessor
    joblib.dump(pre, run_dir / "preprocessor.joblib")

    meta = {
        "rows_total": int(df.shape[0]),
        "feature_count": int(len(feature_names)),
        "log_target": bool(args.log_target),
        "sizes": {
            "X_t_train": int(X_t_train.shape[0]),
            "X_t_val": int(X_t_val.shape[0]),
            "X_t_test": int(X_t_test.shape[0]),
            "X_shadow": int(X_shadow.shape[0]),
        },
        "seed": args.seed,
        "shadow_pool_frac": args.shadow_pool_frac,
        "feature_names": feature_names,
    }
    with open(run_dir / "prepare_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"[OK] saved arrays => {outdir / 'processed_splits.npz'}")
    print(f"[OK] saved preprocessor => {run_dir/'preprocessor.joblib'}")
if __name__ == "__main__":
    main()
