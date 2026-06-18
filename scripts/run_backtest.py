"""Run the full strategy on downloaded real data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cross_market_mr.analysis import capacity_analysis, cost_sensitivity, parameter_sensitivity
from cross_market_mr.bonus import run_bonus_suite
from cross_market_mr.config import load_config
from cross_market_mr.pipeline import run_strategy
from cross_market_mr.report import append_analysis_tables, append_bonus_report, generate_report
from make_pdf_report import write_pdf_report


def _series_map_from_prices(prices: pd.DataFrame) -> dict[str, pd.Series]:
    return {column: prices[column].dropna() for column in prices.columns}


def main() -> None:
    config = load_config("configs/universe.yaml")
    price_path = Path("data/processed/prices.parquet")
    if not price_path.exists():
        raise FileNotFoundError("Run python scripts/download_data.py first.")

    prices = pd.read_parquet(price_path)
    series_map = _series_map_from_prices(prices)
    result = run_strategy(config, series_map)

    output_dir = Path("reports/live")
    result.save(output_dir)
    report_path = output_dir / "strategy_report.md"
    generate_report(result, report_path)

    sensitivity = parameter_sensitivity(
        config,
        series_map,
        entry_values=[1.5, 2.0, 2.5],
        exit_values=[0.25, 0.5, 1.0],
        regression_windows=[60, 90, 120],
        zscore_windows=[30, 60, 90],
    )
    sensitivity.to_csv(output_dir / "parameter_sensitivity.csv", index=False)

    cost = cost_sensitivity(config, series_map, multipliers=[0.5, 1.0, 2.0, 3.0])
    cost.to_csv(output_dir / "cost_sensitivity.csv", index=False)

    capacity = capacity_analysis(
        weights=result.capped_weights,
        prices=result.prices,
        default_adv_usd=100_000_000.0,
    )
    capacity.to_csv(output_dir / "capacity_analysis.csv")
    append_analysis_tables(report_path, sensitivity, cost, capacity)

    bonus = run_bonus_suite(result, config, target_asset="MSTR", basis_reference="BTC")
    bonus.save(output_dir)
    append_bonus_report(report_path, bonus)
    pdf_path = write_pdf_report(output_dir)

    print("Backtest completed.")
    print(f"Report: {report_path}")
    print(f"PDF Report: {pdf_path}")


if __name__ == "__main__":
    main()
