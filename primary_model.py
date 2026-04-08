"""Phase 3: Primary Model – LightGBM multi-class classifier.

Trains a LightGBM model to predict Triple-Barrier labels
(``1``, ``0``, ``-1``) using a **strictly chronological** train/test split
to prevent look-ahead bias and data leakage.
"""

from __future__ import annotations

from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


# Columns that are *not* model features
_NON_FEATURE_COLS = {"primary_label", "meta_label"}


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names from *df*.

    Args:
        df: Labelled DataFrame produced by the feature and labeling pipelines.

    Returns:
        A sorted list of column names to use as model input features.
    """
    return sorted(c for c in df.columns if c not in _NON_FEATURE_COLS)


def train_primary_model(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    num_leaves: int = 31,
    max_depth: int = 4,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Tuple[lgb.Booster, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Train a LightGBM multi-class classifier on chronologically split data.

    The dataset is split in strict chronological order: the first
    ``train_ratio`` fraction of rows becomes the training set and the
    remainder becomes the held-out test set.  No shuffling is performed.

    Args:
        df: DataFrame that must contain a ``primary_label`` column (values in
            ``{-1, 0, 1}``) as well as feature columns produced by
            ``generate_features``.
        train_ratio: Fraction of rows to use for training (default ``0.8``).
        num_leaves: Maximum number of leaves in one tree (default ``31``).
        max_depth: Maximum tree depth (default ``4``).
        n_estimators: Number of boosting rounds (default ``300``).
        learning_rate: Learning rate for gradient boosting (default ``0.05``).
        random_state: Random seed for reproducibility (default ``42``).

    Notes:
        Per-sample class weights are computed from the training set to
        compensate for the typical class-imbalance produced by Triple-Barrier
        labeling (many more stop-loss events than profit-take events).

    Returns:
        A tuple of:
        - ``booster``: The trained ``lgb.Booster`` object.
        - ``X_test``: Out-of-sample feature DataFrame.
        - ``y_test``: Out-of-sample true labels (original encoding).
        - ``y_pred``: Out-of-sample predictions (original encoding).
        - ``feature_cols``: List of feature column names used by the model.

    Raises:
        ValueError: If ``primary_label`` column is absent from ``df``.
    """
    if "primary_label" not in df.columns:
        raise ValueError("DataFrame must contain a 'primary_label' column.")

    feature_cols = _get_feature_columns(df)
    X = df[feature_cols]
    y_raw = df["primary_label"]

    # LightGBM multi-class requires non-negative integer labels
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), index=y_raw.index, name="primary_label")

    # Chronological split
    split_idx = int(len(df) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test_enc = y.iloc[:split_idx], y.iloc[split_idx:]

    params = {
        "objective": "multiclass",
        "num_class": len(le.classes_),
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "verbose": -1,
        "random_state": random_state,
        "n_jobs": -1,
    }

    # Compute balanced per-sample weights from the training labels
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train.to_numpy(),
    )
    weight_map = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
    sample_weights = y_train.map(weight_map).to_numpy()

    train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    valid_set = lgb.Dataset(X_test, label=y_test_enc, reference=train_set)

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # Predictions → class indices → original labels
    proba = booster.predict(X_test)
    y_pred_enc = np.argmax(proba, axis=1)
    y_pred = pd.Series(le.inverse_transform(y_pred_enc), index=X_test.index, name="pred")
    y_test = pd.Series(le.inverse_transform(y_test_enc), index=X_test.index, name="primary_label")

    print("=== Primary Model (LightGBM) – Classification Report ===")
    original_labels = sorted(y_raw.unique())
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in original_labels]))

    return booster, X_test, y_test, y_pred, feature_cols
