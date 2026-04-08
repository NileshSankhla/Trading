"""Phase 1: Feature Engineering pipeline for intraday OHLCV data.

Generates technical indicators and derived features used by the trading
model, following Marcos Lopez de Prado's methodology described in
*Advances in Financial Machine Learning*.
"""

import pandas as pd
import pandas_ta as ta


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate technical and statistical features from raw OHLCV data.

    Computes the following features on top of the input DataFrame:
    - RSI (14-period)
    - MACD line, signal line, and histogram (12, 26, 9)
    - ATR (14-period)
    - Bollinger Bands upper, middle, and lower (20-period)
    - 5-minute rolling percentage return
    - 15-minute rolling percentage return
    - Volume imbalance (current volume / 10-period SMA of volume)

    All rows that contain NaN values introduced by the rolling windows are
    dropped before the enriched DataFrame is returned.

    Args:
        df: A pandas DataFrame with at minimum the columns
            ``['open', 'high', 'low', 'close', 'volume']`` (case-insensitive).
            The index should be a ``DatetimeIndex`` representing 5-minute bars.

    Returns:
        A copy of ``df`` with additional feature columns and no NaN rows.

    Raises:
        ValueError: If any of the required OHLCV columns are missing.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - {c.lower() for c in df.columns}
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    # Work on a copy; normalise column names to lower-case for pandas_ta
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # --- RSI (14) ---
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # --- MACD (12, 26, 9) ---
    macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd_df is not None:
        df["macd"] = macd_df.iloc[:, 0]        # MACD line
        df["macd_signal"] = macd_df.iloc[:, 1]  # Signal line
        df["macd_hist"] = macd_df.iloc[:, 2]    # Histogram

    # --- ATR (14) ---
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # --- Bollinger Bands (20, 2) ---
    bb_df = ta.bbands(df["close"], length=20, std=2)
    if bb_df is not None:
        df["bb_lower"] = bb_df.iloc[:, 0]   # BBL
        df["bb_mid"] = bb_df.iloc[:, 1]     # BBM
        df["bb_upper"] = bb_df.iloc[:, 2]   # BBU

    # --- Rolling percentage returns ---
    # 5-minute return  = 1-bar pct change  (each bar is 5 min)
    # 15-minute return = 3-bar pct change
    df["ret_5m"] = df["close"].pct_change(periods=1)
    df["ret_15m"] = df["close"].pct_change(periods=3)

    # --- Volume Imbalance (current vol / 10-period SMA vol) ---
    df["vol_sma_10"] = df["volume"].rolling(window=10).mean()
    df["vol_imbalance"] = df["volume"] / df["vol_sma_10"]
    df.drop(columns=["vol_sma_10"], inplace=True)

    # Drop rows with NaN values introduced by rolling / indicator calculations
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
