"""Synthetic market generator for offline smoke testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig


@dataclass
class SyntheticDataset:
    """Synthetic price and return data."""

    series_map: dict[str, pd.Series]
    factor_returns: pd.DataFrame
    target_returns: pd.DataFrame


def _make_price_series(returns: pd.Series, start_price: float = 100.0) -> pd.Series:
    clean = returns.fillna(0.0)
    price = start_price * np.exp(clean.cumsum())
    price.name = returns.name
    return price


def _business_calendar(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start_date, end_date, freq="C")


def generate_synthetic_dataset(
    config: StrategyConfig,
    seed: int = 7,
) -> SyntheticDataset:
    """Generate a realistic but controlled synthetic cross-market dataset."""
    rng = np.random.default_rng(seed)

    daily_index = pd.date_range(config.settings.start_date, config.settings.end_date, freq="D")
    business_index = _business_calendar(config.settings.start_date, config.settings.end_date)

    btc = pd.Series(0.0002 + 0.030 * rng.standard_normal(len(daily_index)), index=daily_index, name="BTC")
    eth = pd.Series(0.0003 + 0.025 * rng.standard_normal(len(daily_index)) + 0.65 * btc.values,
                    index=daily_index, name="ETH")
    spy = pd.Series(0.0001 + 0.010 * rng.standard_normal(len(business_index)), index=business_index, name="SPY")
    qqq = pd.Series(0.00015 + 0.011 * rng.standard_normal(len(business_index)) + 0.90 * spy.values,
                    index=business_index, name="QQQ")
    smh = pd.Series(0.00018 + 0.013 * rng.standard_normal(len(business_index)) + 0.75 * qqq.values,
                    index=business_index, name="SMH")

    factor_returns = pd.concat([btc, eth, spy, qqq, smh], axis=1)

    series_map: dict[str, pd.Series] = {
        "BTC": _make_price_series(btc),
        "ETH": _make_price_series(eth),
        "SPY": _make_price_series(spy),
        "QQQ": _make_price_series(qqq),
        "SMH": _make_price_series(smh),
    }

    target_returns: dict[str, pd.Series] = {}
    base_betas = {
        "MSTR": {"BTC": 2.1, "SPY": 0.35},
        "COIN": {"BTC": 1.6, "QQQ": 0.45},
        "MARA": {"BTC": 1.4, "SMH": 0.25},
        "RIOT": {"BTC": 1.35, "SMH": 0.20},
        "CLSK": {"BTC": 1.25, "SMH": 0.18},
        "HOOD": {"BTC": 0.9, "QQQ": 0.55},
        "SQ": {"BTC": 0.8, "QQQ": 0.50},
        "SOL": {"BTC": 1.15},
        "BNB": {"BTC": 1.05},
        "ADA": {"BTC": 0.95},
        "XRP": {"BTC": 0.90},
        "DOGE": {"BTC": 0.80},
        "AVAX": {"BTC": 1.05},
        "LINK": {"BTC": 0.90, "ETH": 0.25},
        "LTC": {"BTC": 0.85},
        "BCH": {"BTC": 0.82},
        "DOT": {"BTC": 0.88},
        "AAVE": {"BTC": 0.65, "ETH": 0.35},
        "NVDA": {"QQQ": 1.20, "SMH": 0.60},
        "TSLA": {"QQQ": 1.10},
        "AMD": {"QQQ": 0.95, "SMH": 0.70},
        "META": {"QQQ": 1.00},
    }

    for symbol in config.target_symbols:
        instrument = config.instruments[symbol]
        index = daily_index if instrument.kind == "crypto_perp" else business_index
        residual = np.zeros(len(index))
        shock = rng.normal(scale=0.010 if instrument.kind == "equity" else 0.018, size=len(index))
        phi = 0.78 if instrument.kind == "equity" else 0.72
        for i in range(1, len(index)):
            residual[i] = phi * residual[i - 1] + shock[i]

        beta_map = base_betas.get(symbol, {})
        factor_component = np.zeros(len(index))
        for factor_symbol, beta in beta_map.items():
            factor_series = factor_returns[factor_symbol].reindex(index)
            factor_component += beta * factor_series.fillna(0.0).values

        alpha = 0.00005 if instrument.kind == "equity" else 0.00010
        ret = alpha + factor_component + residual
        ret_series = pd.Series(ret, index=index, name=symbol)
        target_returns[symbol] = ret_series
        series_map[symbol] = _make_price_series(ret_series)

    target_return_frame = pd.concat(target_returns, axis=1).sort_index()
    return SyntheticDataset(
        series_map=series_map,
        factor_returns=factor_returns,
        target_returns=target_return_frame,
    )

