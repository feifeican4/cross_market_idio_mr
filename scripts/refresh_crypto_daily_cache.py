"""Refresh Binance crypto daily cache through the configured end date."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cross_market_mr.config import load_config
from cross_market_mr.data import (
    _safe_name,
    build_price_panel,
    download_binance_archive_close_series,
    save_panel,
)


def main() -> None:
    config = load_config("configs/universe.yaml")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    crypto_symbols = [
        symbol
        for symbol, instrument in config.instruments.items()
        if instrument.kind == "crypto_spot" or symbol in {"BTC", "ETH"}
    ]

    updated = []
    missing = []
    for symbol in crypto_symbols:
        instrument = config.instruments[symbol]
        ticker = instrument.ticker.replace("-USD", "/USDT")
        try:
            series = download_binance_archive_close_series(
                ticker,
                start_date=config.settings.start_date,
                end_date=config.settings.end_date,
                name=symbol,
                interval="1d",
                market="spot",
            )
            series.to_frame(name=symbol).to_parquet(raw_dir / f"{_safe_name(symbol)}.parquet")
            updated.append((symbol, series.index.min(), series.index.max(), len(series)))
        except Exception as exc:
            missing.append((symbol, str(exc)))

    series_map = {}
    for symbol in config.instruments:
        cache_file = raw_dir / f"{_safe_name(symbol)}.parquet"
        if cache_file.exists():
            series_map[symbol] = pd.read_parquet(cache_file).iloc[:, 0].rename(symbol)

    prices = build_price_panel(series_map)
    save_panel(prices, "data/processed/prices.parquet")

    print("Updated crypto symbols:")
    for row in updated:
        print(f"  {row[0]}: {row[1]} -> {row[2]} ({row[3]} rows)")
    if missing:
        print("Missing crypto symbols:")
        for row in missing:
            print(f"  {row[0]}: {row[1]}")
    print(f"Saved panel: {prices.shape}, {prices.index.min()} -> {prices.index.max()}")


if __name__ == "__main__":
    main()
