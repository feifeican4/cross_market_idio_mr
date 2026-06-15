"""Run the full pipeline on synthetic data.

This is the fastest way to understand the code without relying on network
access or third-party data APIs.
"""

import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cross_market_mr.analysis import capacity_analysis, cost_sensitivity, parameter_sensitivity
from cross_market_mr.config import StrategyConfig, load_config
from cross_market_mr.pipeline import run_strategy
from cross_market_mr.report import append_analysis_tables, generate_report
from cross_market_mr.synthetic import generate_synthetic_dataset


def _build_demo_config() -> StrategyConfig:
    base = load_config("configs/universe.yaml")
    demo_instruments = {
        symbol: base.instruments[symbol]
        for symbol in [
            "BTC",
            "ETH",
            "SPY",
            "QQQ",
            "SMH",
            "MSTR",
            "COIN",
            "SOL",
            "AAVE",
            "NVDA",
        ]
    }
    demo_settings = replace(
        base.settings,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    return StrategyConfig(settings=demo_settings, instruments=demo_instruments)


def main() -> None:
    config = _build_demo_config()
    dataset = generate_synthetic_dataset(config, seed=7)
    result = run_strategy(config, dataset.series_map)

    output_dir = Path("reports/demo")
    result.save(output_dir)
    report_path = output_dir / "strategy_report.md"
    generate_report(result, report_path)

    sensitivity = parameter_sensitivity(
        config,
        dataset.series_map,
        entry_values=[1.5, 2.0],
        exit_values=[0.5],
        regression_windows=[60, 90],
        zscore_windows=[30],
    )
    sensitivity.to_csv(output_dir / "parameter_sensitivity.csv", index=False)

    cost = cost_sensitivity(config, dataset.series_map, multipliers=[1.0, 2.0])
    cost.to_csv(output_dir / "cost_sensitivity.csv", index=False)

    capacity = capacity_analysis(
        weights=result.capped_weights,
        prices=result.prices,
        default_adv_usd=100_000_000.0,
    )
    capacity.to_csv(output_dir / "capacity_analysis.csv")
    append_analysis_tables(report_path, sensitivity, cost, capacity)

    print("Demo completed.")
    print(f"Report: {output_dir / 'strategy_report.md'}")
    print("Summary:")
    for key, value in result.summary.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
