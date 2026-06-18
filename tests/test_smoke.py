import unittest
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cross_market_mr.config import StrategyConfig, load_config
from cross_market_mr.pipeline import run_strategy
from cross_market_mr.synthetic import generate_synthetic_dataset
from cross_market_mr.bonus import run_bonus_suite


class SmokeTest(unittest.TestCase):
    def test_synthetic_pipeline_runs(self) -> None:
        base = load_config("configs/universe.yaml")
        instruments = {
            symbol: base.instruments[symbol]
            for symbol in ["BTC", "ETH", "SPY", "QQQ", "SMH", "MSTR", "COIN", "SOL", "AAVE", "NVDA"]
        }
        config = StrategyConfig(
            settings=replace(
                base.settings,
                start_date="2024-01-01",
                end_date="2024-06-30",
                regression_window=60,
                zscore_window=30,
            ),
            instruments=instruments,
        )
        dataset = generate_synthetic_dataset(config, seed=11)
        result = run_strategy(config, dataset.series_map)

        self.assertFalse(result.backtest.daily.empty)
        self.assertIn("net_return", result.backtest.daily.columns)
        self.assertFalse(result.diagnostics.empty)
        self.assertGreaterEqual(result.capped_weights.abs().sum(axis=1).max(), 0.0)

    def test_bonus_suite_runs(self) -> None:
        base = load_config("configs/universe.yaml")
        instruments = {
            symbol: base.instruments[symbol]
            for symbol in ["BTC", "ETH", "SPY", "QQQ", "SMH", "MSTR", "COIN", "SOL", "AAVE", "NVDA"]
        }
        config = StrategyConfig(
            settings=replace(base.settings, start_date="2023-01-01", end_date="2024-12-31"),
            instruments=instruments,
        )
        dataset = generate_synthetic_dataset(config, seed=13)
        result = run_strategy(config, dataset.series_map)
        bonus = run_bonus_suite(result, config, target_asset="MSTR", basis_reference="BTC")

        self.assertFalse(bonus.basis_daily.empty)
        self.assertFalse(bonus.dynamic_daily.empty)
        self.assertFalse(bonus.ml_daily.empty)


if __name__ == "__main__":
    unittest.main()
