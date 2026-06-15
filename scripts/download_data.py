"""Download daily price data for the configured universe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cross_market_mr.config import load_config
from cross_market_mr.data import build_price_panel, load_or_download_series_map, save_panel


def main() -> None:
    config = load_config("configs/universe.yaml")
    series_map, missing = load_or_download_series_map(config, cache_dir="data/raw")
    prices = build_price_panel(series_map)
    save_panel(prices, Path("data/processed/prices.parquet"))

    if missing:
        print("Missing instruments:")
        for symbol in missing:
            print(f"  {symbol}")
    print(f"Saved {len(prices.columns)} columns to data/processed/prices.parquet")


if __name__ == "__main__":
    main()
