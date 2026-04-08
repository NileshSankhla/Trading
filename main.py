"""Main entry-point for the intraday algorithmic trading pipeline.

This script ties together all five phases of the training pipeline:

    Phase 1 – Feature Engineering  (feature_pipeline.py)
    Phase 2 – Triple-Barrier Labeling  (labeling.py)
    Phase 3 – Primary LightGBM Model   (primary_model.py)
    Phase 4 – Meta-Labeling XGBoost Filter  (meta_model.py)
    Phase 5 – Model Serialization  (utils.py)

It can be invoked in two ways:

    1. With a CSV file path::

        python main.py --csv path/to/HINDZINC_5min.csv

    2. Without any arguments (generates dummy OHLCV data for a quick smoke
       test)::

        python main.py

Architecture reference: Marcos Lopez de Prado,
*Advances in Financial Machine Learning* (2018).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from feature_pipeline import generate_features
from labeling import apply_triple_barrier
from meta_model import train_meta_model
from primary_model import train_primary_model
from utils import save_artefacts


# ---------------------------------------------------------------------------
# Dummy data generator
# ---------------------------------------------------------------------------

def generate_dummy_ohlcv(
    n_bars: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic 5-minute OHLCV data for pipeline testing.

    Produces a realistic-looking price series using a geometric random walk
    with intraday volume patterns.

    Args:
        n_bars: Number of 5-minute bars to generate (default ``2000``).
        seed: Random seed for reproducibility (default ``42``).

    Returns:
        A DataFrame with columns ``['open', 'high', 'low', 'close', 'volume']``
        and a ``DatetimeIndex`` at 5-minute frequency starting from
        ``2023-01-02 09:15``.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-02 09:15:00")
    index = pd.date_range(start, periods=n_bars, freq="5min")

    # Geometric random walk for close prices
    log_returns = rng.normal(loc=0.0, scale=0.002, size=n_bars)
    close = 2000.0 * np.exp(np.cumsum(log_returns))

    # Build OHLC around close
    noise = rng.uniform(0.001, 0.005, size=n_bars)
    high = close * (1.0 + noise)
    low = close * (1.0 - noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Volume with intraday seasonality (higher at open/close)
    base_vol = rng.integers(50_000, 200_000, size=n_bars)
    volume = base_vol.astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame, output_dir: str = "models") -> None:
    """Execute all five phases of the trading model training pipeline.

    Args:
        df: Raw OHLCV DataFrame (5-minute bars).
        output_dir: Directory where trained artefacts will be saved.
    """
    print("\n" + "=" * 60)
    print("PHASE 1 – Feature Engineering")
    print("=" * 60)
    df_features = generate_features(df)
    print(f"Feature DataFrame shape: {df_features.shape}")
    print(f"Columns: {list(df_features.columns)}")

    print("\n" + "=" * 60)
    print("PHASE 2 – Triple-Barrier Labeling")
    print("=" * 60)
    df_labeled = apply_triple_barrier(
        df_features,
        target_pct=0.005,   # 0.5 % profit target (symmetric with stop)
        stop_pct=0.005,     # 0.5 % stop-loss
        time_limit_bars=12,  # max 12 bars (1 hour) look-ahead
    )
    label_counts = df_labeled["primary_label"].value_counts().sort_index()
    print(f"Label distribution:\n{label_counts}")

    print("\n" + "=" * 60)
    print("PHASE 3 – Primary Model (LightGBM)")
    print("=" * 60)
    booster, X_test, y_test, y_pred, feature_cols = train_primary_model(
        df_labeled,
        train_ratio=0.8,
        max_depth=4,
    )

    print("\n" + "=" * 60)
    print("PHASE 4 – Meta-Labeling Model (XGBoost)")
    print("=" * 60)
    meta_clf, meta_proba = train_meta_model(
        X_test=X_test,
        y_test=y_test,
        y_pred_primary=y_pred,
        feature_cols=feature_cols,
        max_depth=4,
    )

    print("\n" + "=" * 60)
    print("PHASE 5 – Serialization")
    print("=" * 60)
    save_artefacts(booster, meta_clf, feature_cols, output_dir=output_dir)

    print("\nPipeline completed successfully.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intraday algorithmic trading training pipeline."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "Path to a CSV file containing 5-minute OHLCV data. "
            "If omitted, dummy data is generated automatically."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory for saving trained model artefacts (default: 'models').",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    if args.csv:
        print(f"Loading data from: {args.csv}")
        raw_df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    else:
        print("No CSV provided – generating dummy OHLCV data …")
        raw_df = generate_dummy_ohlcv(n_bars=2000)

    run_pipeline(raw_df, output_dir=args.output_dir)
