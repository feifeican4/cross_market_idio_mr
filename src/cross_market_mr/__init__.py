"""Cross-market idiosyncratic mean reversion research package."""

from .backtest import BacktestResult, run_backtest
from .config import InstrumentConfig, StrategyConfig, StrategySettings, load_config
from .factor_model import RollingFactorModelResult, adf_test, fit_rolling_factor_model
from .pipeline import StrategyRunResult, run_strategy
from .signals import build_hysteresis_signal, rolling_zscore

__all__ = [
    "adf_test",
    "BacktestResult",
    "build_hysteresis_signal",
    "fit_rolling_factor_model",
    "InstrumentConfig",
    "load_config",
    "RollingFactorModelResult",
    "run_backtest",
    "run_strategy",
    "StrategyConfig",
    "StrategyRunResult",
    "StrategySettings",
    "rolling_zscore",
]

