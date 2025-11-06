import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

REQ_COLS = ["age","sex","bmi","children","smoker","region","charges"]
CATEGORICAL = ["sex","smoker","region"]
NUMERIC = ["age","bmi","children"]

def read_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace({"nan": np.nan, "NaN": np.nan, "NULL": np.nan})
    df = df.dropna(subset=REQ_COLS)
    df = df.drop_duplicates()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["children"] = pd.to_numeric(df["children"], errors="coerce").astype(int)
    df = df.dropna(subset=["age","bmi","children"])
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["charges"] = pd.to_numeric(df["charges"], errors="coerce")
    df = df.dropna(subset=["charges"])
    return df.reset_index(drop=True)

def build_preprocessor():
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
    ap = argparse.ArgumentParser(description="Prepare processed arrays and saved splits.")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.runs_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    df = read_clean(args.csv)
    y = df["charges"].to_numpy(dtype=float)
    if args.log_target:
        y = np.log1p(y)

    X_df = df[NUMERIC + CATEGORICAL].copy()

    pre = build_preprocessor()
    X = pre.fit_transform(X_df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    ct = pre.named_steps["pre"]
    ohe = ct.named_transformers_["cat"]
    if hasattr(ohe, "get_feature_names_out"):
        cat_feat_names = ohe.get_feature_names_out(CATEGORICAL).tolist()
    else:
        cat_feat_names = ohe.get_feature_names(CATEGORICAL).tolist()
    feature_names = NUMERIC + cat_feat_names

    X_target, X_shadow, y_target, y_shadow = train_test_split(
        X, y, test_size=args.shadow_pool_frac, random_state=args.seed
    )
    X_t_train, X_t_temp, y_t_train, y_t_temp = train_test_split(
        X_target, y_target, test_size=0.4, random_state=args.seed
    )
    X_t_val, X_t_test, y_t_val, y_t_test = train_test_split(
        X_t_temp, y_t_temp, test_size=0.5, random_state=args.seed
    )

    np.savez_compressed(
        Path(args.outdir) / "processed_splits.npz",
        X_t_train=X_t_train, y_t_train=y_t_train,
        X_t_val=X_t_val, y_t_val=y_t_val,
        X_t_test=X_t_test, y_t_test=y_t_test,
        X_shadow=X_shadow, y_shadow=y_shadow,
        feature_names=np.array(feature_names, dtype=object),
        log_target=np.array([args.log_target]),
    )

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
    print(f"[OK] saved arrays → data/processed/processed_splits.npz")
    print(f"[OK] saved preprocessor → {run_dir/'preprocessor.joblib'}")

if __name__ == "__main__":
    main()
