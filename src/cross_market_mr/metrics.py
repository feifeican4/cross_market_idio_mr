"""Performance, risk, and attribution metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity_curve: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Return max drawdown and its start/end dates."""
    if equity_curve.empty:
        return 0.0, pd.NaT, pd.NaT
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    end = drawdown.idxmin()
    start = equity_curve.loc[:end].idxmax()
    return float(drawdown.min()), start, end


def performance_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Common strategy metrics."""
    clean = returns.dropna()
    if clean.empty:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win_loss": 0.0,
            "avg_daily_return": 0.0,
        }

    equity = (1.0 + clean).cumprod()
    ann_return = equity.iloc[-1] ** (periods_per_year / len(clean)) - 1.0
    ann_vol = clean.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    mdd, _, _ = max_drawdown(equity)
    calmar = ann_return / abs(mdd) if mdd < 0 else 0.0
    wins = clean[clean > 0]
    losses = clean[clean < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else np.inf
    avg_win_loss = wins.mean() / abs(losses.mean()) if not losses.empty else np.inf

    return {
        "annual_return": float(ann_return),
        "annual_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "max_drawdown": float(mdd),
        "win_rate": float((clean > 0).mean()),
        "profit_factor": float(profit_factor),
        "avg_win_loss": float(avg_win_loss),
        "avg_daily_return": float(clean.mean()),
    }


def summarize_drawdown(equity_curve: pd.Series) -> pd.DataFrame:
    """Create a one-row drawdown table."""
    mdd, start, end = max_drawdown(equity_curve)
    return pd.DataFrame(
        [
            {
                "start": start,
                "end": end,
                "max_drawdown": mdd,
            }
        ]
    )


def factor_exposure_summary(weights: pd.DataFrame, factor_symbols: list[str]) -> pd.DataFrame:
    """Daily factor exposure from explicit hedge legs."""
    columns = [symbol for symbol in factor_symbols if symbol in weights.columns]
    if not columns:
        return pd.DataFrame(index=weights.index)
    exposure = weights[columns].copy()
    exposure.columns = [f"net_{col}" for col in exposure.columns]
    exposure["gross_factor_exposure"] = exposure.abs().sum(axis=1)
    return exposure


def turnover_summary(weights: pd.DataFrame) -> pd.Series:
    """Daily portfolio turnover."""
    if weights.empty:
        return pd.Series(dtype=float)
    return weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))

