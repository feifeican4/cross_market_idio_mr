"""Portfolio construction and risk caps."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import StrategyConfig


def annualized_vol(returns: pd.Series, window: int = 60, periods_per_year: int = 252) -> pd.Series:
    """Rolling annualized volatility."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def build_pair_weight_frame(
    asset: str,
    signal: pd.Series,
    betas: pd.DataFrame,
    residuals: pd.Series,
    factor_symbols: list[str],
    config: StrategyConfig,
) -> pd.DataFrame:
    """Build asset leg plus factor hedge legs for one target asset."""
    settings = config.settings
    beta_cols = [col for col in betas.columns if col != "const"]
    beta_frame = betas.reindex(signal.index).ffill()
    pair_vol = annualized_vol(residuals, window=settings.zscore_window).shift(1)
    notional = settings.capital_per_signal * settings.target_pair_vol / pair_vol
    notional = notional.replace([np.inf, -np.inf], pd.NA).fillna(0.0)
    asset_weight = (signal.astype(float) * notional).clip(
        lower=-settings.max_single_weight,
        upper=settings.max_single_weight,
    )

    frame = pd.DataFrame(index=signal.index)
    frame[asset] = asset_weight
    for factor in factor_symbols:
        if factor in beta_cols:
            beta = beta_frame[factor].fillna(0.0)
            frame[factor] = -asset_weight * beta
        else:
            frame[factor] = 0.0
    return frame.fillna(0.0)


def combine_weight_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sum pair-level weights into a portfolio-level frame."""
    if not frames:
        return pd.DataFrame()

    all_index = sorted({idx for frame in frames.values() for idx in frame.index})
    all_columns = sorted({col for frame in frames.values() for col in frame.columns})
    total = pd.DataFrame(0.0, index=pd.Index(all_index, name="date"), columns=all_columns)
    for frame in frames.values():
        aligned = frame.reindex(index=total.index, columns=total.columns).fillna(0.0)
        total = total.add(aligned, fill_value=0.0)
    return total.fillna(0.0)


def apply_single_name_cap(weights: pd.DataFrame, max_single_weight: float) -> pd.DataFrame:
    """Cap each instrument's absolute exposure."""
    return weights.clip(lower=-max_single_weight, upper=max_single_weight)


def apply_group_cap(
    weights: pd.DataFrame,
    group_map: dict[str, str],
    max_group_weight: float,
) -> pd.DataFrame:
    """Cap gross exposure per group by proportionally scaling the group."""
    capped = weights.copy()
    groups: dict[str, list[str]] = defaultdict(list)
    for symbol, group in group_map.items():
        if symbol in capped.columns:
            groups[group].append(symbol)

    for idx, row in capped.iterrows():
        scaled_row = row.copy()
        for group, columns in groups.items():
            gross = scaled_row[columns].abs().sum()
            if gross > max_group_weight and gross > 0:
                scale = max_group_weight / gross
                scaled_row.loc[columns] = scaled_row[columns] * scale
        capped.loc[idx] = scaled_row
    return capped


def apply_factor_cap(
    weights: pd.DataFrame,
    factor_symbols: list[str],
    max_factor_weight: float,
) -> pd.DataFrame:
    """Cap factor leg exposures directly."""
    capped = weights.copy()
    for symbol in factor_symbols:
        if symbol in capped.columns:
            capped[symbol] = capped[symbol].clip(
                lower=-max_factor_weight,
                upper=max_factor_weight,
            )
    return capped


def apply_gross_leverage_cap(weights: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    """Scale the whole portfolio if gross leverage is too high."""
    gross = weights.abs().sum(axis=1)
    scale = pd.Series(1.0, index=weights.index)
    mask = gross > max_gross
    scale.loc[mask] = max_gross / gross.loc[mask]
    return weights.mul(scale, axis=0)


def apply_risk_caps(
    weights: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Apply single-name, group, factor, and gross leverage constraints."""
    capped = apply_single_name_cap(weights, config.settings.max_single_weight)
    capped = apply_group_cap(capped, config.group_map, config.settings.max_group_weight)
    capped = apply_factor_cap(capped, config.factor_symbols, config.settings.max_factor_weight)
    capped = apply_gross_leverage_cap(capped, config.settings.max_gross_leverage)
    return capped.fillna(0.0)


def annualized_turnover(weights: pd.DataFrame, periods_per_year: int = 252) -> float:
    """Estimate annualized one-way turnover."""
    if weights.empty:
        return 0.0
    daily_turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    return float(daily_turnover.mean() * periods_per_year / 2.0)


def average_holding_period(signals: pd.DataFrame) -> float:
    """Average non-zero holding length in days across all signals."""
    durations: list[int] = []
    for column in signals.columns:
        state = signals[column].fillna(0.0)
        run = 0
        for value in state:
            if value != 0:
                run += 1
            elif run > 0:
                durations.append(run)
                run = 0
        if run > 0:
            durations.append(run)
    if not durations:
        return 0.0
    return float(np.mean(durations))

