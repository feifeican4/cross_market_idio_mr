# Cross-Market Idiosyncratic Mean-Reversion Strategy

This repository implements the core tasks in the interview assignment:

1. Daily data pipeline for crypto, crypto-related equities, and technology equities
2. Rolling multi-factor model for each target asset
3. Residual z-score signal generation
4. Multi-leg factor-hedged portfolio construction
5. Transaction cost, slippage, borrow cost, and funding cost modeling
6. Performance, risk, sensitivity, and capacity analysis
7. Reproducible Markdown report generation

The bonus items are intentionally not implemented in the main path. The code is
kept simple so that every step can be explained in an interview.

## Core Idea

For each asset, estimate:

```text
r_asset = alpha + beta_1 * factor_1 + beta_2 * factor_2 + residual
```

The strategy trades the residual, not the raw asset return:

- if residual z-score is too high, short the asset and buy the hedge factors
- if residual z-score is too low, buy the asset and short the hedge factors
- exit when the residual normalizes

This follows the residual-stat-arb logic used in professional quant research.
Useful references:

- Avellaneda and Lee, "Statistical Arbitrage in the US Equities Market"
- "On statistical arbitrage under a conditional factor model of equity returns"
- "Deep Learning Statistical Arbitrage"

## Quick Start

Use the synthetic demo first. It needs no network and is meant for code review.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_demo.py
```

Run unit smoke test:

```powershell
python -m unittest
```

Run real-data daily backtest:

```powershell
python scripts/download_data.py
python scripts/run_backtest.py
python scripts/make_report.py
```

## Outputs

Synthetic demo outputs:

```text
reports/demo/backtest_daily.csv
reports/demo/diagnostics.csv
reports/demo/factor_exposure.csv
reports/demo/parameter_sensitivity.csv
reports/demo/cost_sensitivity.csv
reports/demo/capacity_analysis.csv
reports/demo/strategy_report.md
```

Real-data outputs go to `reports/live/`.

## Code Map

```text
configs/universe.yaml              universe, factors, costs, risk caps
scripts/download_data.py           downloads daily prices
scripts/run_backtest.py            real-data full pipeline
scripts/run_demo.py                offline synthetic full pipeline
src/cross_market_mr/config.py      config parser
src/cross_market_mr/data.py        data download and return calculation
src/cross_market_mr/factor_model.py rolling OLS and ADF diagnostics
src/cross_market_mr/signals.py     residual z-score and entry/exit logic
src/cross_market_mr/portfolio.py   hedge weights and risk caps
src/cross_market_mr/backtest.py    close-to-close backtest with costs
src/cross_market_mr/analysis.py    sensitivity and capacity analysis
src/cross_market_mr/report.py      Markdown report writer
```

## Important Research Choices

- Rolling regression uses only past data to avoid look-ahead bias.
- Target weights are shifted by one day in the backtest.
- z-score mean and standard deviation are shifted by one day.
- Costs are charged on turnover.
- Equity short exposure pays borrow cost.
- Perp exposure can pay a conservative funding proxy.
- Capacity is estimated from ADV participation and turnover.

## Data Caveat

The real-data path uses yfinance for many equities and crypto proxies. Binance
TradFi perp data should replace these proxies before live trading. The report
must state this clearly.
