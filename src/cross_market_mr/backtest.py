"""Close-to-close backtest with explicit transaction and carry costs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import StrategyConfig


@dataclass
class BacktestResult:
    """Daily backtest output."""

    daily: pd.DataFrame

    @property
    def equity_curve(self) -> pd.Series:
        return self.daily["equity_curve"]


def _cost_rate_map(default_map: dict[str, float] | None) -> dict[str, float]:
    return dict(default_map or {})


def estimate_transaction_cost(
    target_weights: pd.DataFrame,
    config: StrategyConfig,
    kind_map: dict[str, str],
) -> pd.Series:
    """Estimate turnover cost from target-to-target weight changes."""
    fee_map = _cost_rate_map(config.settings.fee_bps_by_kind)
    slippage_map = _cost_rate_map(config.settings.slippage_bps_by_kind)
    turnover = target_weights.diff().abs().fillna(target_weights.abs())
    cost = pd.Series(0.0, index=target_weights.index)

    for symbol in target_weights.columns:
        kind = kind_map.get(symbol, "equity")
        fee_bps = fee_map.get(kind, 10.0)
        slip_bps = slippage_map.get(kind, 5.0)
        rate = (fee_bps + slip_bps) / 10000.0
        cost = cost + turnover[symbol] * rate

    return cost


def estimate_borrow_cost(
    holdings: pd.DataFrame,
    kind_map: dict[str, str],
    annual_borrow_rate: float,
    periods_per_year: int = 252,
) -> pd.Series:
    """Charge borrow cost only on short equity exposures."""
    cost = pd.Series(0.0, index=holdings.index)
    daily_rate = annual_borrow_rate / periods_per_year
    for symbol in holdings.columns:
        if kind_map.get(symbol, "equity") == "equity":
            short_exposure = holdings[symbol].clip(upper=0.0).abs()
            cost = cost + short_exposure * daily_rate
    return cost


def estimate_funding_cost(
    holdings: pd.DataFrame,
    kind_map: dict[str, str],
    annual_funding_rate: float,
    periods_per_year: int = 252,
) -> pd.Series:
    """Conservative funding cost proxy for perp legs."""
    cost = pd.Series(0.0, index=holdings.index)
    daily_rate = annual_funding_rate / periods_per_year
    for symbol in holdings.columns:
        if kind_map.get(symbol, "equity") == "crypto_perp":
            cost = cost + holdings[symbol].abs() * daily_rate
    return cost


def run_backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: StrategyConfig,
) -> BacktestResult:
    """Run a one-day-lag close-to-close backtest."""
    if returns.empty or target_weights.empty:
        daily = pd.DataFrame(
            columns=[
                "gross_return",
                "transaction_cost",
                "borrow_cost",
                "funding_cost",
                "net_return",
                "gross_leverage",
                "equity_curve",
            ]
        )
        return BacktestResult(daily=daily)

    index = sorted(set(returns.index).union(target_weights.index))
    aligned_returns = returns.reindex(index).fillna(0.0)
    aligned_targets = target_weights.reindex(index).fillna(0.0)
    holdings = aligned_targets.shift(1).fillna(0.0)

    gross_return = (holdings * aligned_returns).sum(axis=1)
    transaction_cost = estimate_transaction_cost(aligned_targets, config, config.kind_map)
    borrow_cost = estimate_borrow_cost(
        holdings,
        kind_map=config.kind_map,
        annual_borrow_rate=config.settings.annual_borrow_rate,
    )
    funding_cost = estimate_funding_cost(
        holdings,
        kind_map=config.kind_map,
        annual_funding_rate=config.settings.annual_funding_rate,
    )
    net_return = gross_return - transaction_cost - borrow_cost - funding_cost
    gross_leverage = holdings.abs().sum(axis=1)
    equity_curve = (1.0 + net_return.fillna(0.0)).cumprod()

    daily = pd.DataFrame(
        {
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "borrow_cost": borrow_cost,
            "funding_cost": funding_cost,
            "net_return": net_return,
            "gross_leverage": gross_leverage,
            "equity_curve": equity_curve,
        }
    )
    return BacktestResult(daily=daily)

