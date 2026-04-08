"""Phase 4: Meta-Labeling Filter – XGBoost binary classifier.

Trains a secondary (meta) model that filters out false positives from the
primary LightGBM model.  Only bars where the primary model predicted *Buy*
(label ``1``) are used.  The meta-model learns to distinguish true positives
(actual outcome ``1``) from false positives (actual outcome ``-1`` or ``0``).

Reference: Lopez de Prado, *Advances in Financial Machine Learning*, Ch. 4.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score


def train_meta_model(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred_primary: pd.Series,
    feature_cols: list[str],
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Tuple[xgb.XGBClassifier, np.ndarray]:
    """Train a binary XGBoost meta-labeling model on primary model predictions.

    Filtering step:
        Only rows where ``y_pred_primary == 1`` (primary model predicted *Buy*)
        are retained for meta-model training.

    Meta-label encoding:
        - ``1`` – true positive (primary predicted Buy **and** actual was Buy)
        - ``0`` – false positive (primary predicted Buy **but** actual was -1 or 0)

    Args:
        X_test: Out-of-sample feature DataFrame from the primary model test set.
        y_test: True labels for the test set (values in ``{-1, 0, 1}``).
        y_pred_primary: Primary model predictions aligned with ``X_test``.
        feature_cols: List of feature column names to use as model inputs.
        n_estimators: Number of boosting rounds (default ``300``).
        max_depth: Maximum tree depth (default ``4``).
        learning_rate: Learning rate for gradient boosting (default ``0.05``).
        random_state: Random seed for reproducibility (default ``42``).

    Returns:
        A tuple of:
        - ``meta_clf``: Trained ``xgb.XGBClassifier`` instance.
        - ``meta_proba``: Predicted probability of ``meta_label == 1`` for each
          row in the filtered set (shape ``(n_filtered,)``).

    Raises:
        ValueError: If no rows satisfy the filter condition
            (primary predicted ``1``).
    """
    # --- Filter to Buy predictions only ---
    buy_mask = y_pred_primary == 1
    X_meta = X_test.loc[buy_mask, feature_cols]
    y_actual_meta = y_test.loc[buy_mask]

    if X_meta.empty:
        raise ValueError(
            "No rows with primary_model prediction == 1 found. "
            "Cannot train the meta-model."
        )

    # Binary meta_label: 1 → true positive, 0 → false positive
    meta_label = (y_actual_meta == 1).astype(int)

    meta_clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    meta_clf.fit(X_meta, meta_label)

    meta_proba: np.ndarray = meta_clf.predict_proba(X_meta)[:, 1]
    meta_pred = (meta_proba >= 0.5).astype(int)

    print("=== Meta-Model (XGBoost) – Classification Report ===")
    print(classification_report(meta_label, meta_pred, target_names=["False Pos", "True Pos"]))
    if len(np.unique(meta_label)) > 1:
        auc = roc_auc_score(meta_label, meta_proba)
        print(f"ROC-AUC: {auc:.4f}")

    return meta_clf, meta_proba
