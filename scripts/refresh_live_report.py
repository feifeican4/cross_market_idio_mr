"""Regenerate live Markdown and PDF reports from existing real-data outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cross_market_mr.config import load_config
from cross_market_mr.pipeline import run_strategy
from cross_market_mr.report import append_analysis_tables, generate_report
from make_pdf_report import write_pdf_report


def main() -> None:
    config = load_config("configs/universe.yaml")
    prices = pd.read_parquet("data/processed/prices.parquet")
    series_map = {column: prices[column].dropna() for column in prices.columns}
    result = run_strategy(config, series_map)

    output_dir = Path("reports/live")
    report_path = output_dir / "strategy_report.md"
    generate_report(result, report_path)

    parameter = pd.read_csv(output_dir / "parameter_sensitivity.csv")
    cost = pd.read_csv(output_dir / "cost_sensitivity.csv")
    capacity = pd.read_csv(output_dir / "capacity_analysis.csv", index_col=0)
    append_analysis_tables(report_path, parameter, cost, capacity)

    # Keep already computed real-data bonus CSVs; the PDF reads them directly.
    pdf_path = write_pdf_report(output_dir)
    print(f"Markdown report written to {report_path}")
    print(f"PDF report written to {pdf_path}")


if __name__ == "__main__":
    main()
