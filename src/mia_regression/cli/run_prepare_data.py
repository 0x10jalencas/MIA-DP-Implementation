import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Columns required for processing
REQ_COLS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]

# Group columns by type
CATEGORICAL = ["sex", "smoker", "region"]
NUMERIC = ["age", "bmi", "children"]


def read_clean(csv_path: str) -> pd.DataFrame:
    """
    Read and clean up the CSV data.

    Returns:
        - DataFrame: Cleaned DataFrame.
    """
    # Read in the raw data
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Replace any common null representations with np.nan
    df = df.replace({"nan": np.nan, "NaN": np.nan, "NULL": np.nan})

    # Remove rows with missing required columns and duplicates
    df = df.dropna(subset=REQ_COLS)
    df = df.drop_duplicates()

    # Drop bad rows and ensure rows are correctly typed
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["children"] = pd.to_numeric(df["children"], errors="coerce").astype(int)
    df = df.dropna(subset=["age", "bmi", "children"])

    # Normalize categorical columns
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip().str.lower()

    # Ensures charges are valid numerics
    df["charges"] = pd.to_numeric(df["charges"], errors="coerce")
    df = df.dropna(subset=["charges"])

    # Return cleaned DataFrame with reset index
    return df.reset_index(drop=True)


def build_preprocessor():
    """
    Create a preprocessing pipeline for the data.

    Returns:
        - Pipeline: Preprocessing pipeline.
    """
    cat = OneHotEncoder(handle_unknown="ignore")
    num = StandardScaler()
    pre = ColumnTransformer(
        transformers=[
            ("num", num, NUMERIC),
            ("cat", cat, CATEGORICAL),
        ]
    )
    return Pipeline(steps=[("pre", pre)])


def main():
    # Create argument parser
    ap = argparse.ArgumentParser(
        description="Prepare processed arrays and saved splits."
    )
    args = ap.parse_args()

    # Make sure output directories exist
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Create run directory with timestamp
    run_dir = Path(args.runs_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load and clean data
    df = read_clean(args.csv)
    y = df["charges"].to_numpy(dtype=float)

    # Log transform target if specified
    if args.log_target:
        y = np.log1p(y)

    # Get the dataframe with only feature columns
    X_df = df[NUMERIC + CATEGORICAL].copy()

    # Fit in the preprocessing pipeline and transform data
    pre = build_preprocessor()
    X = pre.fit_transform(X_df)

    # Convert to dense if sparse
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Extract feature names
    ct = pre.named_steps["pre"]
    ohe = ct.named_transformers_["cat"]

    # Get categorical feature names
    if hasattr(ohe, "get_feature_names_out"):
        cat_feat_names = ohe.get_feature_names_out(CATEGORICAL).tolist()
    else:
        cat_feat_names = ohe.get_feature_names(CATEGORICAL).tolist()
    feature_names = NUMERIC + cat_feat_names

    # Build target and shadow splits
    X_target, X_shadow, y_target, y_shadow = train_test_split(
        X, y, test_size=args.shadow_pool_frac, random_state=args.seed
    )

    # Split target data into train and temp
    X_t_train, X_t_temp, y_t_train, y_t_temp = train_test_split(
        X_target, y_target, test_size=0.4, random_state=args.seed
    )

    # Split temp into val and test
    X_t_val, X_t_test, y_t_val, y_t_test = train_test_split(
        X_t_temp, y_t_temp, test_size=0.5, random_state=args.seed
    )

    # Compress and save all arrays
    np.savez_compressed(
        Path(args.outdir) / "processed_splits.npz",
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

    # Save fitted preprocessor
    joblib.dump(pre, run_dir / "preprocessor.joblib")

    # Save metadata
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

    # Print out summary
    print(json.dumps(meta, indent=2))
    print(f"[OK] saved arrays → data/processed/processed_splits.npz")
    print(f"[OK] saved preprocessor → {run_dir/'preprocessor.joblib'}")


if __name__ == "__main__":
    main()
