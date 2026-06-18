"""Run a 4-hour synthetic intraday demo for the bonus frequency requirement."""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cross_market_mr.config import StrategyConfig, load_config
from cross_market_mr.pipeline import run_strategy
from cross_market_mr.report import generate_report
from cross_market_mr.synthetic import generate_synthetic_intraday_dataset


def main() -> None:
    base = load_config("configs/universe.yaml")
    intraday_settings = replace(
        base.settings,
        start_date="2024-01-01",
        end_date="2024-06-30",
        regression_window=180,
        zscore_window=90,
    )
    config = StrategyConfig(settings=intraday_settings, instruments=base.instruments)
    dataset = generate_synthetic_intraday_dataset(config, seed=19, freq="4h")
    result = run_strategy(config, dataset.series_map)

    output_dir = Path("reports/intraday_demo")
    result.save(output_dir)
    generate_report(result, output_dir / "strategy_report.md")

    print("Intraday demo completed.")
    print(f"Report: {output_dir / 'strategy_report.md'}")
    for key, value in result.summary.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
