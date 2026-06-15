"""Sensitivity, cost stress, and capacity analysis helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import StrategyConfig, StrategySettings
from .metrics import performance_metrics
from .pipeline import run_strategy


def with_settings(config: StrategyConfig, **overrides: object) -> StrategyConfig:
    """Create a config copy with selected setting overrides."""
    settings = replace(config.settings, **overrides)
    return StrategyConfig(settings=settings, instruments=config.instruments)


def parameter_sensitivity(
    config: StrategyConfig,
    series_map: dict[str, pd.Series],
    entry_values: list[float] | None = None,
    exit_values: list[float] | None = None,
    regression_windows: list[int] | None = None,
    zscore_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Grid-search key parameters without changing the strategy logic."""
    entry_values = entry_values or [1.5, 2.0, 2.5]
    exit_values = exit_values or [0.25, 0.5, 1.0]
    regression_windows = regression_windows or [60, 90, 120]
    zscore_windows = zscore_windows or [30, 60, 90]

    rows: list[dict[str, float]] = []
    for entry_z in entry_values:
        for exit_z in exit_values:
            if exit_z >= entry_z:
                continue
            for regression_window in regression_windows:
                for zscore_window in zscore_windows:
                    cfg = with_settings(
                        config,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        regression_window=regression_window,
                        zscore_window=zscore_window,
                    )
                    result = run_strategy(cfg, series_map)
                    metrics = performance_metrics(result.backtest.daily["net_return"])
                    rows.append(
                        {
                            "entry_z": entry_z,
                            "exit_z": exit_z,
                            "regression_window": regression_window,
                            "zscore_window": zscore_window,
                            "annual_return": metrics["annual_return"],
                            "annual_volatility": metrics["annual_volatility"],
                            "sharpe": metrics["sharpe"],
                            "max_drawdown": metrics["max_drawdown"],
                            "avg_gross_leverage": result.backtest.daily["gross_leverage"].mean(),
                        }
                    )
    return pd.DataFrame(rows)


def cost_sensitivity(
    config: StrategyConfig,
    series_map: dict[str, pd.Series],
    multipliers: list[float] | None = None,
) -> pd.DataFrame:
    """Stress-test fees and slippage."""
    multipliers = multipliers or [0.5, 1.0, 2.0, 3.0]
    rows: list[dict[str, float]] = []
    base_fees = config.settings.fee_bps_by_kind or {}
    base_slippage = config.settings.slippage_bps_by_kind or {}

    for multiplier in multipliers:
        fee_map = {key: value * multiplier for key, value in base_fees.items()}
        slippage_map = {key: value * multiplier for key, value in base_slippage.items()}
        cfg = with_settings(
            config,
            fee_bps_by_kind=fee_map,
            slippage_bps_by_kind=slippage_map,
        )
        result = run_strategy(cfg, series_map)
        metrics = performance_metrics(result.backtest.daily["net_return"])
        rows.append(
            {
                "cost_multiplier": multiplier,
                "annual_return": metrics["annual_return"],
                "annual_volatility": metrics["annual_volatility"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "total_transaction_cost": result.backtest.daily["transaction_cost"].sum(),
                "total_borrow_cost": result.backtest.daily["borrow_cost"].sum(),
                "total_funding_cost": result.backtest.daily["funding_cost"].sum(),
            }
        )
    return pd.DataFrame(rows)


def estimate_adv_from_prices(
    prices: pd.DataFrame,
    volume_proxy: pd.DataFrame | None = None,
    default_adv_usd: float = 100_000_000.0,
) -> pd.Series:
    """Estimate average dollar volume.

    If volume data is unavailable, use a conservative placeholder so that the
    capacity report is still produced and clearly labeled as a proxy.
    """
    if volume_proxy is not None and not volume_proxy.empty:
        adv = volume_proxy.reindex(prices.index).mean()
        return adv.astype(float)
    return pd.Series(default_adv_usd, index=prices.columns, dtype=float)


def capacity_analysis(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    volume_proxy: pd.DataFrame | None = None,
    participation_rate: float = 0.05,
    default_adv_usd: float = 100_000_000.0,
) -> pd.DataFrame:
    """Estimate AUM capacity from turnover and ADV participation."""
    if weights.empty:
        return pd.DataFrame()

    adv = estimate_adv_from_prices(
        prices=prices,
        volume_proxy=volume_proxy,
        default_adv_usd=default_adv_usd,
    )
    avg_abs_trade_weight = weights.diff().abs().mean().replace(0.0, np.nan)
    max_trade_dollars = adv * participation_rate
    capacity = max_trade_dollars / avg_abs_trade_weight
    table = pd.DataFrame(
        {
            "adv_usd_proxy": adv,
            "participation_rate": participation_rate,
            "max_trade_usd": max_trade_dollars,
            "avg_abs_trade_weight": avg_abs_trade_weight,
            "capacity_usd_proxy": capacity,
        }
    )
    return table.sort_values("capacity_usd_proxy")

