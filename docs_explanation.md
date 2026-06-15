# Detailed Explanation for the Assignment

This document explains the project in plain language. Use it to understand the
logic before reading the code.

## 1. What the Strategy Is

The strategy is a market-neutral relative-value strategy.

It does not ask:

```text
Will MSTR go up tomorrow?
```

It asks:

```text
Did MSTR move too much compared with BTC and SPY?
```

If MSTR is too high relative to its factors, the strategy shorts MSTR and buys
the factors. If MSTR is too low relative to its factors, it buys MSTR and shorts
the factors.

## 2. Why This Is Called Idiosyncratic

An asset return can be decomposed as:

```text
asset return = systematic return + idiosyncratic return
```

Systematic return is explained by factors such as BTC, ETH, SPY, QQQ, and SMH.
Idiosyncratic return is the residual left after removing those factor effects.

The strategy wants to trade this residual.

## 3. Why Use a Factor Model

Suppose BTC rises 5% and MSTR rises 12%. If MSTR historically has beta 2 to BTC,
then 10% of the move is explained by BTC. Only the remaining 2% is idiosyncratic.

The factor model estimates this relationship:

```text
MSTR return = alpha + 2.0 * BTC return + 0.3 * SPY return + residual
```

This is implemented in `src/cross_market_mr/factor_model.py`.

## 4. Why Use Rolling Regression

Betas change over time. MSTR's BTC beta in a bull market is not necessarily the
same as in a bear market.

So the code uses rolling OLS:

```text
Use the past 90 days to estimate beta.
Use that beta to predict today's return.
Residual = actual return - predicted return.
```

The current day is not used in the training window. This avoids look-ahead bias.

## 5. Why Use ADF Test

Mean reversion requires some form of stationarity. The ADF test is a statistical
check on whether the residual series behaves more like a stationary process or a
random walk.

In the output:

```text
adf_p_value < 0.05
```

is better evidence that residual mean reversion may be plausible.

It is not a guarantee of profitability.

## 6. How the Signal Works

The strategy converts residuals into z-scores:

```text
z = (residual - rolling mean) / rolling std
```

Rules:

```text
z > +2.0  => residual too high => short asset, long factors
z < -2.0  => residual too low  => long asset, short factors
|z| < 0.5 => residual normalized => exit
```

This is implemented in `src/cross_market_mr/signals.py`.

## 7. How the Hedge Is Constructed

If we long one unit of MSTR and beta is:

```text
BTC beta = 2.0
SPY beta = 0.3
```

then we hedge by shorting:

```text
2.0 units BTC
0.3 units SPY
```

In code:

```text
factor weight = -asset weight * beta
```

This is implemented in `src/cross_market_mr/portfolio.py`.

## 8. How Risk Budgeting Works

The same dollar position is not equally risky for all assets. A volatile residual
needs a smaller position.

The code scales position size by residual volatility:

```text
asset weight = signal * capital_per_signal * target_pair_vol / residual_vol
```

Then it applies assignment constraints:

- single instrument cap: 3%
- group cap: 15%
- factor leg cap: 5%
- gross leverage cap: 3x

## 9. How the Backtest Avoids Look-Ahead Bias

Signals are calculated using today's close, so they cannot earn today's return.
They can only be traded from the next day.

The key line is:

```python
holdings = aligned_targets.shift(1).fillna(0.0)
```

This is in `src/cross_market_mr/backtest.py`.

## 10. Costs Included

The backtest includes:

- taker fee
- slippage
- equity borrow cost for short equity legs
- funding proxy for perp legs

Costs are important because residual mean-reversion strategies can have high
turnover. A strategy that looks good before costs may disappear after costs.

## 11. How to Interpret the Main Outputs

`diagnostics.csv`:

- average R2
- median R2
- ADF statistic
- ADF p-value
- trade count
- active days

`backtest_daily.csv`:

- gross return
- transaction cost
- borrow cost
- funding cost
- net return
- gross leverage
- equity curve

`parameter_sensitivity.csv`:

- tests whether performance survives different windows and z thresholds

`cost_sensitivity.csv`:

- tests whether performance survives higher trading costs

`capacity_analysis.csv`:

- rough AUM capacity estimate from turnover and ADV participation

## 12. What to Say in an Interview

Say this:

```text
I model each asset's return as factor return plus idiosyncratic residual.
The strategy only trades residual extremes after removing BTC, ETH, SPY, QQQ,
and SMH exposure. I estimate beta using rolling OLS and generate z-score signals
from the residual. Portfolio weights are volatility-scaled and beta-hedged.
The backtest shifts positions by one day and includes fees, slippage, borrow,
and funding costs.
```

Then be honest:

```text
The real limitation is data quality and tradability. yfinance is only a research
proxy for some markets. True Binance TradFi perp prices, funding rates, and
order book depth are needed before production deployment.
```

## 13. What Is Completed

The project covers the non-bonus requirements:

- daily data pipeline
- three asset categories
- factor model
- rolling regression
- R2 report
- ADF test
- residual z-score
- entry and exit rules
- beta hedge
- volatility-based risk budgeting
- single-name, group, factor, and leverage constraints
- fee, slippage, borrow, and funding costs
- no look-ahead backtest
- performance metrics
- drawdown analysis
- factor exposure report
- parameter sensitivity
- cost sensitivity
- capacity analysis
- Markdown report

