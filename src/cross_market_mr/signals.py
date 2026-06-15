"""Residual z-score signals and hysteresis state machine."""

from __future__ import annotations

import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Compute a look-ahead-safe rolling z-score."""
    mean = series.rolling(window).mean().shift(1)
    std = series.rolling(window).std(ddof=1).shift(1)
    zscore = (series - mean) / std
    return zscore.replace([float("inf"), float("-inf")], pd.NA)


def build_hysteresis_signal(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """Convert z-scores into a stateful long/short/flat signal.

    Convention:
    +1 => long asset, short factor hedge
    -1 => short asset, long factor hedge
     0 => flat
    """
    signal = pd.Series(index=zscore.index, dtype=float)
    state = 0.0

    for date, value in zscore.items():
        if pd.isna(value):
            signal.loc[date] = state
            continue

        if state == 0.0:
            if value > entry_z:
                state = -1.0
            elif value < -entry_z:
                state = 1.0
        else:
            if abs(value) < exit_z:
                state = 0.0

        signal.loc[date] = state

    return signal


def signal_trade_count(signal: pd.Series) -> int:
    """Count number of position changes."""
    changes = signal.fillna(0).diff().abs()
    return int((changes > 0).sum())

