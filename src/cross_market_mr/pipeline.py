"""End-to-end research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import StrategyConfig
from .data import build_price_panel, compute_return_panel
from .factor_model import RollingFactorModelResult, fit_rolling_factor_model, model_diagnostics
from .metrics import (
    factor_exposure_summary,
    performance_metrics,
    summarize_drawdown,
    turnover_summary,
)
from .portfolio import (
    apply_risk_caps,
    average_holding_period,
    build_pair_weight_frame,
    combine_weight_frames,
)
from .signals import build_hysteresis_signal, rolling_zscore, signal_trade_count


@dataclass
class StrategyRunResult:
    """All outputs from one strategy run."""

    config: StrategyConfig
    prices: pd.DataFrame
    returns: pd.DataFrame
    models: dict[str, RollingFactorModelResult]
    signals: pd.DataFrame
    pair_weights: dict[str, pd.DataFrame]
    raw_weights: pd.DataFrame
    capped_weights: pd.DataFrame
    backtest: BacktestResult
    diagnostics: pd.DataFrame
    summary: dict[str, float]
    drawdown_table: pd.DataFrame
    factor_exposure: pd.DataFrame
    turnover: pd.Series

    def save(self, output_dir: str | Path) -> None:
        """Persist the main research artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.prices.to_parquet(output_path / "prices.parquet")
        self.returns.to_parquet(output_path / "returns.parquet")
        self.signals.to_parquet(output_path / "signals.parquet")
        self.raw_weights.to_parquet(output_path / "raw_weights.parquet")
        self.capped_weights.to_parquet(output_path / "capped_weights.parquet")
        self.backtest.daily.to_csv(output_path / "backtest_daily.csv")
        self.diagnostics.to_csv(output_path / "diagnostics.csv", index=False)
        self.drawdown_table.to_csv(output_path / "drawdown_table.csv", index=False)
        self.factor_exposure.to_csv(output_path / "factor_exposure.csv")


def run_strategy(
    config: StrategyConfig,
    series_map: dict[str, pd.Series],
) -> StrategyRunResult:
    """Run the full cross-market residual mean-reversion strategy."""
    prices = build_price_panel(series_map)
    returns = compute_return_panel(series_map)

    models: dict[str, RollingFactorModelResult] = {}
    pair_weights: dict[str, pd.DataFrame] = {}
    signal_frame: dict[str, pd.Series] = {}
    diagnostics_rows: list[dict[str, float | str | int]] = []

    for symbol in config.target_symbols:
        if symbol not in returns.columns:
            continue
        factor_symbols = [factor for factor in config.factor_list_for(symbol) if factor in returns.columns]
        if not factor_symbols:
            continue
        asset_returns = returns[symbol]
        factor_returns = returns[factor_symbols]
        if asset_returns.dropna().empty or factor_returns.dropna(how="all").empty:
            continue

        model = fit_rolling_factor_model(
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            window=config.settings.regression_window,
        )
        zscore = rolling_zscore(model.residuals, window=config.settings.zscore_window)
        signal = build_hysteresis_signal(
            zscore,
            entry_z=config.settings.entry_z,
            exit_z=config.settings.exit_z,
        )
        weights = build_pair_weight_frame(
            asset=symbol,
            signal=signal,
            betas=model.betas,
            residuals=model.residuals,
            factor_symbols=factor_symbols,
            config=config,
        )

        models[symbol] = model
        pair_weights[symbol] = weights
        signal_frame[symbol] = signal

        diag = model_diagnostics(model)
        diag.update(
            {
                "asset": symbol,
                "trades": signal_trade_count(signal),
                "active_days": int((signal != 0).sum()),
            }
        )
        diagnostics_rows.append(diag)

    raw_weights = combine_weight_frames(pair_weights)
    capped_weights = apply_risk_caps(raw_weights, config)
    backtest = run_backtest(returns=returns, target_weights=capped_weights, config=config)

    summary = performance_metrics(backtest.daily["net_return"])
    summary["average_holding_period"] = average_holding_period(pd.DataFrame(signal_frame))
    summary["annualized_turnover"] = float(
        turnover_summary(capped_weights).mean() * 252 / 2.0 if not capped_weights.empty else 0.0
    )

    drawdown_table = summarize_drawdown(backtest.daily["equity_curve"])
    factor_exposure = factor_exposure_summary(capped_weights, config.factor_symbols)

    diagnostics = pd.DataFrame(diagnostics_rows)
    signal_panel = pd.DataFrame(signal_frame).sort_index()
    turnover = turnover_summary(capped_weights)

    return StrategyRunResult(
        config=config,
        prices=prices,
        returns=returns,
        models=models,
        signals=signal_panel,
        pair_weights=pair_weights,
        raw_weights=raw_weights,
        capped_weights=capped_weights,
        backtest=backtest,
        diagnostics=diagnostics,
        summary=summary,
        drawdown_table=drawdown_table,
        factor_exposure=factor_exposure,
        turnover=turnover,
    )
