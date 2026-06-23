"""Run a real 4-hour intraday bonus experiment on Binance historical data."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cross_market_mr.config import StrategyConfig, load_config
from cross_market_mr.data import download_binance_archive_close_series
from cross_market_mr.pipeline import run_strategy
from cross_market_mr.report import generate_report


def _build_intraday_config() -> StrategyConfig:
    base = load_config("configs/universe.yaml")
    intraday_settings = replace(
        base.settings,
        start_date="2024-01-01",
        end_date="2024-06-30",
        regression_window=120,
        zscore_window=60,
        entry_z=1.5,
        exit_z=0.3,
    )
    return StrategyConfig(settings=intraday_settings, instruments=base.instruments)


def _load_real_4h_series(config: StrategyConfig) -> dict[str, object]:
    symbols = ["BTC", "ETH", "SOL", "BNB", "AAVE", "ARB", "OP", "UNI", "LINK", "AVAX"]
    series_map = {}
    for symbol in symbols:
        ticker = "BTC/USDT" if symbol == "BTC" else f"{symbol}/USDT"
        series = download_binance_archive_close_series(
            ticker,
            start_date=config.settings.start_date,
            end_date=config.settings.end_date,
            name=symbol,
            interval="4h",
            market="spot",
            normalize_dates=False,
        )
        series_map[symbol] = series
    return series_map


def main() -> None:
    config = _build_intraday_config()
    series_map = _load_real_4h_series(config)
    result = run_strategy(config, series_map)

    output_dir = Path("reports/intraday_real")
    result.save(output_dir)
    generate_report(result, output_dir / "strategy_report.md")

    print("Intraday real-data experiment completed.")
    print(f"Report: {output_dir / 'strategy_report.md'}")
    for key, value in result.summary.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
