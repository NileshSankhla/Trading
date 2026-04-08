"""Phase 2: Triple-Barrier Labeling based on Marcos Lopez de Prado's
*Advances in Financial Machine Learning*, Chapter 3.

Each observation is assigned one of three labels:
    *  1 – upper (profit-take) barrier hit first
    * -1 – lower (stop-loss) barrier hit first
    *  0 – time limit expired before either barrier was touched
"""

import numpy as np
import pandas as pd


def apply_triple_barrier(
    df: pd.DataFrame,
    target_pct: float,
    stop_pct: float,
    time_limit_bars: int,
) -> pd.DataFrame:
    """Label each bar using the Triple-Barrier method.

    For every row *i* in ``df``, two horizontal barriers are placed:
    - **Upper barrier**: ``close[i] * (1 + target_pct)``
    - **Lower barrier**: ``close[i] * (1 - stop_pct)``

    The function then looks forward up to ``time_limit_bars`` bars.  The
    first barrier that is *breached* (via the ``high`` and ``low`` columns)
    determines the label.

    Args:
        df: DataFrame that must contain at least the columns
            ``['close', 'high', 'low']``.
        target_pct: Fractional take-profit target (e.g. ``0.01`` for 1 %).
        stop_pct: Fractional stop-loss level (e.g. ``0.005`` for 0.5 %).
        time_limit_bars: Maximum number of bars to look ahead before
            assigning the neutral label ``0``.

    Returns:
        A copy of ``df`` with an additional integer column ``primary_label``
        containing values in ``{-1, 0, 1}``.

    Raises:
        ValueError: If required columns are absent from ``df``.
    """
    required = {"close", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.copy()
    labels: list[int] = []

    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    n = len(close_arr)

    for i in range(n):
        entry_price = close_arr[i]
        upper = entry_price * (1.0 + target_pct)
        lower = entry_price * (1.0 - stop_pct)

        label = 0  # default: time-limit expires
        end = min(i + time_limit_bars + 1, n)

        for j in range(i + 1, end):
            hit_upper = high_arr[j] >= upper
            hit_lower = low_arr[j] <= lower

            if hit_upper and hit_lower:
                # Both barriers touched in the same bar.  The upper barrier
                # takes precedence (standard AFML convention), giving a
                # bullish tie-break that is consistent regardless of whether
                # the barriers are symmetric or asymmetric.
                label = 1
                break
            elif hit_upper:
                label = 1
                break
            elif hit_lower:
                label = -1
                break

        labels.append(label)

    df["primary_label"] = labels
    return df
