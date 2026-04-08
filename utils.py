"""Phase 5: Model serialization utilities.

Provides a thin wrapper around ``joblib`` to persist trained model artefacts
(primary LightGBM booster, meta XGBoost classifier, and the ordered list of
feature column names) to disk, and to reload them for live inference.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import lightgbm as lgb
import xgboost as xgb


def save_artefacts(
    primary_model: lgb.Booster,
    meta_model: xgb.XGBClassifier,
    feature_cols: list[str],
    output_dir: str = "models",
) -> None:
    """Persist model artefacts to ``output_dir`` using ``joblib``.

    Three files are written:
    - ``primary_model.joblib`` – the trained LightGBM Booster.
    - ``meta_model.joblib``    – the trained XGBoost classifier.
    - ``feature_cols.joblib``  – the ordered list of feature column names.

    Args:
        primary_model: Trained ``lgb.Booster`` from Phase 3.
        meta_model: Trained ``xgb.XGBClassifier`` from Phase 4.
        feature_cols: Ordered list of feature column names used during
            training (required to align live data before inference).
        output_dir: Directory in which to write the artefact files.
            Created automatically if it does not exist (default ``"models"``).
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "primary_model": os.path.join(output_dir, "primary_model.joblib"),
        "meta_model": os.path.join(output_dir, "meta_model.joblib"),
        "feature_cols": os.path.join(output_dir, "feature_cols.joblib"),
    }

    joblib.dump(primary_model, paths["primary_model"])
    joblib.dump(meta_model, paths["meta_model"])
    joblib.dump(feature_cols, paths["feature_cols"])

    print(f"Artefacts saved to '{output_dir}/':")
    for name, path in paths.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"  {name:<20} → {path}  ({size_kb:.1f} KB)")


def load_artefacts(
    output_dir: str = "models",
) -> tuple[Any, Any, list[str]]:
    """Load model artefacts from ``output_dir``.

    Args:
        output_dir: Directory that contains the serialised artefact files
            written by :func:`save_artefacts` (default ``"models"``).

    Returns:
        A tuple of:
        - ``primary_model``: The loaded ``lgb.Booster``.
        - ``meta_model``: The loaded ``xgb.XGBClassifier``.
        - ``feature_cols``: The ordered list of feature column names.

    Raises:
        FileNotFoundError: If any of the expected artefact files are missing.
    """
    paths = {
        "primary_model": os.path.join(output_dir, "primary_model.joblib"),
        "meta_model": os.path.join(output_dir, "meta_model.joblib"),
        "feature_cols": os.path.join(output_dir, "feature_cols.joblib"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artefact '{name}' not found at '{path}'. "
                "Run the training pipeline first."
            )

    primary_model = joblib.load(paths["primary_model"])
    meta_model = joblib.load(paths["meta_model"])
    feature_cols: list[str] = joblib.load(paths["feature_cols"])

    print(f"Artefacts loaded from '{output_dir}/'.")
    return primary_model, meta_model, feature_cols
