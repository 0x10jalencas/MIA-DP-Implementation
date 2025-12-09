import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_latest_csv(explicit_path: str | None) -> Path:
    """
    Choose a CSV file to generate a plot.

    If we are given a explicit path, use that file.
    Otherwise, we search for the latest sweep CSV in results/sweeps/.
    Specifically, we search for the most recent run.

    Raises:
        SystemExit: If no CSV files are found and no explicit path is given.
    """
    if explicit_path:
        # CSV was given, return it
        return Path(explicit_path)

    # Use smoothed out sweeps first if we are able to find them
    paths = sorted(glob.glob("results/sweeps/sweep_rf_privacy_utility_smoothed_*.csv"))
    if not paths:
        # Fallback to non-smoothed sweeps if none found
        paths = sorted(glob.glob("results/sweeps/sweep_rf_privacy_utility_*.csv"))
    if not paths:
        # Nothing is found, raise an error
        raise SystemExit(
            "No sweep CSVs found under results/sweeps/. Pass --csv <path>."
        )

    # Return the last file
    return Path(paths[-1])


def build_aggregated_df(csv_path: Path) -> pd.DataFrame:
    """
    Load in a CSV file and convert that into a normalized DataFrame for plotting.

    Returns a dataframe with at least the following columns:
        - cfg: tuple (n_estimators, max_depth, min_samples_leaf)
        - x:    mean test RMSE      (float)
        - x_std: std-dev of RMSE    (float, 0.0 if unknown)
        - y:    mean attack AUC     (float)
        - y_std: std-dev of AUC     (float, 0.0 if unknown)
        - seeds: number of seeds used for aggregation (float)
        - ne, md, leaf: hyperparameter columns for annotation
        - orig_idx: original row index in the raw CSV
    """

    # Grab raw CSV data
    raw = pd.read_csv(csv_path)

    # Ensure required columns exist
    if "rmse_mean" not in raw.columns and "rmse_test" in raw.columns:
        raw["rmse_mean"] = raw["rmse_test"]
        raw["rmse_std"] = 0.0
    if "auc_mean" not in raw.columns and "attack_auc" in raw.columns:
        raw["auc_mean"] = raw["attack_auc"]
        raw["auc_std"] = 0.0

    # Ensure seeds column exists
    if "seeds" not in raw.columns:
        raw["seeds"] = 1

    # Group by hyperparameters and aggregate
    raw["cfg"] = list(
        zip(raw["n_estimators"], raw["max_depth"], raw["min_samples_leaf"])
    )

    # Construct DataFrame
    df = pd.DataFrame(
        {
            "cfg": raw["cfg"],
            "x": raw["rmse_mean"].astype(float),
            "x_std": raw.get("rmse_std", pd.Series(0.0, index=raw.index)).astype(float),
            "y": raw["auc_mean"].astype(float),
            "y_std": raw.get("auc_std", pd.Series(0.0, index=raw.index)).astype(float),
            "seeds": raw["seeds"].astype(float),
            "ne": raw["n_estimators"],
            "md": raw["max_depth"],
            "leaf": raw["min_samples_leaf"],
        }
    )

    # Sort by x (RMSE) and reset index to keep track of original rows
    df = (
        df.sort_values("x")
        .reset_index(drop=False)
        .rename(columns={"index": "orig_idx"})
    )

    # Return the constructed and sorted DataFrame
    return df


def pareto_frontier_df(df: pd.DataFrame) -> list[int]:
    """
    Compute Pareto frontier indices on THIS df (minimize x and y).
    Returns a list of df indices (not positions from pre-sort elsewhere).
    """

    # First sort by utility (x), then by privacy (y)
    sdf = df.sort_values(["x", "y"], ascending=[True, True])

    best_y = np.inf
    front = []

    # From best to worst utility, we add points to the fountier if they improve privacy
    for idx, row in sdf.iterrows():
        y = row["y"]
        if y < best_y - 1e-12:
            best_y = y
            front.append(idx)

    # Return the list of indices on the frontier
    return front


def main():
    # Set up argument parser
    ap = argparse.ArgumentParser(
        description="Privacy-utility plot (AUC vs RMSE) with aligned Pareto overlay."
    )
    args = ap.parse_args()

    # Grab CSV path and build aggregated DataFrame
    csv_path = load_latest_csv(args.csv)
    df = build_aggregated_df(csv_path)

    # Determine the vertical error bars based on user choice
    if args.ci == "95ci":
        # Calculate 95% confidence intervals for error bars
        df["y_err"] = 1.96 * (
            df["y_std"] / np.sqrt(np.maximum(df["seeds"].to_numpy(), 1))
        )
    elif args.ci == "std":
        # Use the raw standard deviation as error bars
        df["y_err"] = df["y_std"]
    else:
        # Hide error bars
        df["y_err"] = 0.0

    # Recompute and sort the Pareto frontier by x, and then y
    # Then, compute running minimum on y to identify frontier points
    d = df.sort_values(["x", "y"]).reset_index()
    mask = d["y"] <= d["y"].cummin() + 1e-12
    front_ids = set(d.loc[mask, "index"])
    df["on_frontier"] = df.index.isin(front_ids)

    # At least one frontier point must exist
    assert df["on_frontier"].sum() > 0

    # Create the plot
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8.0, 5.5))
    ax.margins(x=0.05, y=0.06)
    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        color="gray",
        label="random baseline (AUC=0.5)",
        zorder=0,
    )
    ax.errorbar(
        df["x"],
        df["y"],
        yerr=df["y_err"],
        fmt="none",
        capsize=3,
        lw=1,
        color="#1f77b4",
        alpha=0.7,
        zorder=1,
    )
    ax.scatter(df["x"], df["y"], s=36, color="#1f77b4", alpha=0.9, zorder=2)
    front = df[df["on_frontier"]]
    ax.scatter(
        front["x"],
        front["y"],
        s=120,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        zorder=3,
        label="Pareto frontier",
    )

    # Add text labels to specific points if requested
    if args.annotate != "none":
        if args.annotate == "pareto":
            to_annotate = front
        else:
            i_best_util = df["x"].idxmin()
            i_best_priv = df["y"].idxmin()
            x_n = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min() + 1e-12)
            y_n = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min() + 1e-12)
            i_elbow = (x_n + y_n).idxmin()
            to_annotate = df.loc[sorted({i_best_util, i_best_priv, i_elbow})]

        texts = []
        try:
            from adjustText import adjust_text

            for _, r in to_annotate.iterrows():
                label = f"ne={r['ne']},md={r['md']},leaf={r['leaf']}"
                texts.append(ax.text(r["x"], r["y"], label, fontsize=9, clip_on=True))
            adjust_text(
                texts,
                ax=ax,
                only_move={"points": "xy", "texts": "xy"},
                expand_points=(1.1, 1.2),
                expand_text=(1.02, 1.05),
            )
        except Exception:
            dx = 0.002 * (df["x"].max() - df["x"].min() + 1e-12)
            dy = 0.010
            for j, (_, r) in enumerate(to_annotate.iterrows()):
                offx = dx * ((j % 3) - 1)
                offy = dy * (1 if j % 2 == 0 else -1)
                ax.annotate(
                    f"ne={r['ne']},md={r['md']},leaf={r['leaf']}",
                    (r["x"], r["y"]),
                    xytext=(offx, offy),
                    textcoords="data",
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    clip_on=True,
                )

    # Setting the plot's grid, labels, title, and legend
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_xlabel("Test RMSE  (↓ better utility)")
    ax.set_ylabel("Attack AUC  (↓ better privacy)")
    ax.set_title("Privacy-Utility (Random Forest + MIA)")
    ax.legend(loc="upper right", frameon=False)

    # Save the plot
    out_png = csv_path.with_suffix(".clean.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved clean plot → {out_png}")
    print(f"[OK] Source CSV       → {csv_path}")


if __name__ == "__main__":
    main()

