import argparse, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib, json

def rf_pred_variance(model: RandomForestRegressor, X):
    tree_preds = np.stack([t.predict(X) for t in model.estimators_], axis=1)
    pred_mean = tree_preds.mean(axis=1)
    pred_var = tree_preds.var(axis=1)
    return pred_mean, pred_var

def build_attack_features(y_true, y_pred, pred_var):
    abs_err = np.abs(y_true - y_pred)
    sq_err  = (y_true - y_pred) ** 2
    X_attack = np.column_stack([y_pred, abs_err, sq_err, pred_var])
    return X_attack

def main():
    ap = argparse.ArgumentParser(description="Train shadow regressors and build attack dataset.")
    args = ap.parse_args()

    base = Path(".")
    splits_path = Path(args.splits)
    assert splits_path.exists(), f"Missing splits npz at {splits_path}"
    data = np.load(splits_path, allow_pickle=True)
    X_shadow, y_shadow = data["X_shadow"], data["y_shadow"]

    rng = np.random.RandomState(args.seed)
    n = X_shadow.shape[0]

    X_attack_list, y_attack_list = [], []
    shadow_models_meta = []

    for i in range(args.n_shadows):
        idx = rng.permutation(n)
        m_sz = int(args.shadow_train_frac * n)
        memb_idx, nonm_idx = idx[:m_sz], idx[m_sz:]

        Xm, ym = X_shadow[memb_idx], y_shadow[memb_idx]
        Xn, yn = X_shadow[nonm_idx], y_shadow[nonm_idx]

        shadow = RandomForestRegressor(
            n_estimators=args.n_estimators, random_state=args.seed + i, n_jobs=-1
        ).fit(Xm, ym)

        pm, vm = rf_pred_variance(shadow, Xm)
        Xm_attack = build_attack_features(ym, pm, vm)
        ym_attack = np.ones(Xm_attack.shape[0], dtype=np.int32)

        pn, vn = rf_pred_variance(shadow, Xn)
        Xn_attack = build_attack_features(yn, pn, vn)
        yn_attack = np.zeros(Xn_attack.shape[0], dtype=np.int32)

        X_attack_list.extend([Xm_attack, Xn_attack])
        y_attack_list.extend([ym_attack, yn_attack])

        ts_dir = Path(args.save_models_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
        ts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(shadow, ts_dir / f"shadow_{i}.joblib")
        shadow_models_meta.append(str(ts_dir / f"shadow_{i}.joblib"))

    X_attack = np.vstack(X_attack_list)
    y_attack = np.concatenate(y_attack_list)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X_attack=X_attack, y_attack=y_attack,
                        feature_names=np.array(["y_pred","abs_error","sq_error","pred_var"], dtype=object))

    meta = {
        "n_shadows": args.n_shadows,
        "shadow_train_frac": args.shadow_train_frac,
        "n_estimators": args.n_estimators,
        "rows_attack": int(X_attack.shape[0]),
        "dims_attack": int(X_attack.shape[1]),
        "saved_shadow_models": shadow_models_meta[:3] + (["..."] if len(shadow_models_meta) > 3 else [])
    }
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
