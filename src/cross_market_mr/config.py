"""Configuration loading and typed accessors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InstrumentConfig:
    """One tradable or benchmark instrument."""

    symbol: str
    ticker: str
    source: str
    market_type: str
    kind: str
    group: str
    role: str
    factors: tuple[str, ...]


@dataclass(frozen=True)
class StrategySettings:
    """Strategy-level hyperparameters."""

    start_date: str
    end_date: str
    regression_window: int = 90
    zscore_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    target_pair_vol: float = 0.10
    capital_per_signal: float = 0.02
    max_single_weight: float = 0.03
    max_group_weight: float = 0.15
    max_gross_leverage: float = 3.0
    max_factor_weight: float = 0.05
    calendar_ffill_limit: int = 3
    annual_borrow_rate: float = 0.03
    annual_funding_rate: float = 0.05
    fee_bps_by_kind: dict[str, float] | None = None
    slippage_bps_by_kind: dict[str, float] | None = None


@dataclass(frozen=True)
class StrategyConfig:
    """Full project configuration."""

    settings: StrategySettings
    instruments: dict[str, InstrumentConfig]

    @property
    def benchmark_symbols(self) -> list[str]:
        return [
            symbol
            for symbol, meta in self.instruments.items()
            if meta.role == "benchmark"
        ]

    @property
    def target_symbols(self) -> list[str]:
        return [
            symbol
            for symbol, meta in self.instruments.items()
            if meta.role == "target"
        ]

    @property
    def factor_symbols(self) -> list[str]:
        return self.benchmark_symbols

    @property
    def group_map(self) -> dict[str, str]:
        return {symbol: meta.group for symbol, meta in self.instruments.items()}

    @property
    def kind_map(self) -> dict[str, str]:
        return {symbol: meta.kind for symbol, meta in self.instruments.items()}

    def factor_list_for(self, symbol: str) -> tuple[str, ...]:
        return self.instruments[symbol].factors


def _to_instrument(symbol: str, payload: dict[str, Any]) -> InstrumentConfig:
    return InstrumentConfig(
        symbol=symbol,
        ticker=str(payload["ticker"]),
        source=str(payload["source"]),
        market_type=str(payload["market_type"]),
        kind=str(payload["kind"]),
        group=str(payload["group"]),
        role=str(payload["role"]),
        factors=tuple(payload.get("factors", [])),
    )


def load_config(path: str | Path) -> StrategyConfig:
    """Load YAML configuration into typed dataclasses."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    settings_raw = raw["settings"]
    settings = StrategySettings(
        start_date=str(settings_raw["start_date"]),
        end_date=str(settings_raw["end_date"]),
        regression_window=int(settings_raw.get("regression_window", 90)),
        zscore_window=int(settings_raw.get("zscore_window", 60)),
        entry_z=float(settings_raw.get("entry_z", 2.0)),
        exit_z=float(settings_raw.get("exit_z", 0.5)),
        target_pair_vol=float(settings_raw.get("target_pair_vol", 0.10)),
        capital_per_signal=float(settings_raw.get("capital_per_signal", 0.02)),
        max_single_weight=float(settings_raw.get("max_single_weight", 0.03)),
        max_group_weight=float(settings_raw.get("max_group_weight", 0.15)),
        max_gross_leverage=float(settings_raw.get("max_gross_leverage", 3.0)),
        max_factor_weight=float(settings_raw.get("max_factor_weight", 0.05)),
        calendar_ffill_limit=int(settings_raw.get("calendar_ffill_limit", 3)),
        annual_borrow_rate=float(settings_raw.get("annual_borrow_rate", 0.03)),
        annual_funding_rate=float(settings_raw.get("annual_funding_rate", 0.05)),
        fee_bps_by_kind=dict(settings_raw.get("fee_bps_by_kind", {})),
        slippage_bps_by_kind=dict(settings_raw.get("slippage_bps_by_kind", {})),
    )

    instruments = {
        symbol: _to_instrument(symbol, payload)
        for symbol, payload in raw["instruments"].items()
    }

    return StrategyConfig(settings=settings, instruments=instruments)

