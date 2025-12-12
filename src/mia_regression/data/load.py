"""Extra utilities for loading and splitting the UCI white wine quality dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_CSV = Path("data/raw/winequality-white.csv")
TARGET_COLUMN = "quality"


@dataclass
class WineDataSplits:
    """Container for train/val/test splits produced from the wine dataset."""

    X_t_train: np.ndarray
    y_t_train: np.ndarray
    X_t_val: np.ndarray
    y_t_val: np.ndarray
    X_t_test: np.ndarray
    y_t_test: np.ndarray
    feature_names: Tuple[str, ...]
    scaler: StandardScaler

def read_winequality(csv_path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    """Load, clean, and return the winequality-white dataset."""

    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.dropna().drop_duplicates()
    return df.reset_index(drop=True)


def build_features_targets(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, np.ndarray, Tuple[str, ...]]:
    """Split a dataframe into feature matrix, target vector, and feature names."""

    if target_col not in df.columns:
        raise ValueError(f"Expected '{target_col}' column in dataframe.")

    X_df = df.drop(columns=[target_col])
    y = df[target_col].astype(float).to_numpy()
    feature_names = tuple(X_df.columns.tolist())
    return X_df, y, feature_names


def preprocess_features(
    X_df: pd.DataFrame, scaler: StandardScaler | None = None
) -> Tuple[np.ndarray, StandardScaler]:
    """Standardize numeric features (all wine attributes are numeric)."""

    scaler = scaler or StandardScaler()
    X = scaler.fit_transform(X_df)
    return X, scaler


def create_splits(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Create train/val/test splits for the target model."""

    X_t_train, X_t_temp, y_t_train, y_t_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed
    )
    X_t_val, X_t_test, y_t_val, y_t_test = train_test_split(
        X_t_temp, y_t_temp, test_size=0.5, random_state=seed
    )

    return {
        "X_t_train": X_t_train,
        "y_t_train": y_t_train,
        "X_t_val": X_t_val,
        "y_t_val": y_t_val,
        "X_t_test": X_t_test,
        "y_t_test": y_t_test,
    }


def load_processed_winequality(
    csv_path: Path | str = DEFAULT_CSV,
    *,
    seed: int = 42,
) -> WineDataSplits:
    """High-level helper matching what our CLI pipeline saving."""

    df = read_winequality(csv_path)
    X_df, y, feature_names = build_features_targets(df)
    X, scaler = preprocess_features(X_df)
    splits = create_splits(X, y, seed=seed)

    return WineDataSplits(
        feature_names=feature_names,
        scaler=scaler,
        **splits,
    )

if __name__ == "__main__":
    splits = load_processed_winequality()
    print("Feature count:", len(splits.feature_names))
    print("Train/Val/Test shapes:", splits.X_t_train.shape, splits.X_t_val.shape, splits.X_t_test.shape)
