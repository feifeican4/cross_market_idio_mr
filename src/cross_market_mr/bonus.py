"""Bonus research modules: basis arbitrage, dynamic factor selection, and ML gating.

These modules are intentionally lightweight and explanatory rather than
over-optimized. They are meant to show research direction and limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .backtest import run_backtest
from .config import StrategyConfig
from .data import download_binance_archive_close_series
from .metrics import performance_metrics
from .portfolio import apply_risk_caps, build_pair_weight_frame
from .signals import build_hysteresis_signal, rolling_zscore


@dataclass
class BonusSuiteResult:
    """Outputs from the bonus experiments."""

    basis_daily: pd.DataFrame
    basis_summary: dict[str, float]
    dynamic_daily: pd.DataFrame
    dynamic_summary: dict[str, float]
    dynamic_selection: pd.DataFrame
    ml_daily: pd.DataFrame
    ml_summary: dict[str, float]
    ml_coefficients: pd.DataFrame
    ml_classification: dict[str, float]

    def save(self, output_dir: str | Path) -> None:
        """Persist all bonus outputs."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.basis_daily.to_csv(path / "basis_daily.csv")
        self.dynamic_daily.to_csv(path / "dynamic_daily.csv")
        self.dynamic_selection.to_csv(path / "dynamic_selection.csv", index=False)
        self.ml_daily.to_csv(path / "ml_daily.csv")
        self.ml_coefficients.to_csv(path / "ml_coefficients.csv", index=False)
        pd.DataFrame(
            [
                {"module": "basis", **self.basis_summary},
                {"module": "dynamic", **self.dynamic_summary},
                {"module": "ml", **self.ml_summary},
                {"module": "ml_classifier", **self.ml_classification},
            ]
        ).to_csv(path / "bonus_summary.csv", index=False)


def generate_synthetic_basis_pair(
    spot: pd.Series,
    seed: int = 7,
    mean_basis: float = 0.0015,
    phi: float = 0.96,
    sigma: float = 0.004,
) -> pd.DataFrame:
    """Create a mean-reverting perp proxy from a spot series."""
    rng = np.random.default_rng(seed)
    clean_spot = spot.dropna().sort_index()
    basis = np.zeros(len(clean_spot), dtype=float)
    basis[0] = mean_basis
    shocks = rng.normal(scale=sigma, size=len(clean_spot))

    for i in range(1, len(clean_spot)):
        basis[i] = mean_basis + phi * (basis[i - 1] - mean_basis) + shocks[i]

    perp = clean_spot * np.exp(basis)
    return pd.DataFrame(
        {
            "spot": clean_spot,
            "perp": perp,
            "basis": pd.Series(basis, index=clean_spot.index),
        }
    )


def _basis_signal(
    basis_z: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """Basis signal: +1 means long spot / short perp, -1 means the reverse."""
    signal = pd.Series(index=basis_z.index, dtype=float)
    state = 0.0

    for dt, value in basis_z.items():
        if pd.isna(value):
            signal.loc[dt] = state
            continue

        if state == 0.0:
            if value > entry_z:
                state = 1.0
            elif value < -entry_z:
                state = -1.0
        else:
            if abs(value) < exit_z:
                state = 0.0

        signal.loc[dt] = state

    return signal


def run_basis_arbitrage(
    spot: pd.Series,
    perp: pd.Series,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    notional: float = 0.02,
    fee_bps_spot: float = 10.0,
    fee_bps_perp: float = 4.0,
    annual_funding_rate: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest a simple spot-perp basis arbitrage spread."""
    panel = pd.concat(
        {
            "spot": spot,
            "perp": perp,
        },
        axis=1,
    ).dropna().sort_index()
    panel["basis"] = np.log(panel["perp"]) - np.log(panel["spot"])
    panel["basis_z"] = rolling_zscore(panel["basis"], window=window)
    panel["signal"] = _basis_signal(panel["basis_z"], entry_z=entry_z, exit_z=exit_z)

    spot_ret = np.log(panel["spot"]).diff().fillna(0.0)
    perp_ret = np.log(panel["perp"]).diff().fillna(0.0)

    spot_weight = panel["signal"] * notional
    perp_weight = -panel["signal"] * notional
    holdings = pd.DataFrame({"spot": spot_weight, "perp": perp_weight}, index=panel.index)
    lagged = holdings.shift(1).fillna(0.0)

    gross_return = lagged["spot"] * spot_ret + lagged["perp"] * perp_ret
    turnover = holdings.diff().abs().sum(axis=1).fillna(holdings.abs().sum(axis=1))
    transaction_cost = turnover * ((fee_bps_spot + fee_bps_perp) / 10000.0)
    funding_cost = lagged["perp"].abs() * annual_funding_rate / 252.0
    net_return = gross_return - transaction_cost - funding_cost
    equity_curve = (1.0 + net_return).cumprod()

    daily = pd.DataFrame(
        {
            "basis": panel["basis"],
            "basis_z": panel["basis_z"],
            "signal": panel["signal"],
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "funding_cost": funding_cost,
            "net_return": net_return,
            "equity_curve": equity_curve,
        }
    )
    summary = performance_metrics(daily["net_return"])
    summary.update(
        {
            "mean_basis": float(panel["basis"].mean()),
            "basis_volatility": float(panel["basis"].std(ddof=1)),
            "active_days": float((panel["signal"] != 0).sum()),
            "turnover": float(turnover.mean()),
        }
    )
    return daily, summary


def _fit_aic_ols(y: pd.Series, x: pd.DataFrame) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, float]:
    """Fit OLS and return the fitted model and AIC."""
    x_const = sm.add_constant(x, has_constant="add")
    fit = sm.OLS(y, x_const).fit()
    return fit, float(fit.aic)


def stepwise_select_factors(
    y: pd.Series,
    x: pd.DataFrame,
    max_factors: int = 4,
    min_aic_improvement: float = 2.0,
) -> list[str]:
    """Forward-backward stepwise selection using AIC."""
    joined = pd.concat([y.rename("y"), x], axis=1).dropna()
    if joined.empty:
        return []

    y_clean = joined["y"]
    x_clean = joined.drop(columns=["y"])
    candidates = list(x_clean.columns)
    selected: list[str] = []
    current_fit, current_aic = _fit_aic_ols(y_clean, pd.DataFrame(index=y_clean.index))

    while True:
        best_candidate = None
        best_fit = None
        best_aic = current_aic

        for factor in candidates:
            if factor in selected:
                continue
            trial_cols = selected + [factor]
            fit, aic = _fit_aic_ols(y_clean, x_clean[trial_cols])
            if current_aic - aic >= min_aic_improvement and aic < best_aic:
                best_candidate = factor
                best_fit = fit
                best_aic = aic

        if best_candidate is None or len(selected) >= max_factors:
            break

        selected.append(best_candidate)
        current_fit = best_fit
        current_aic = best_aic

        while selected:
            removed = False
            for factor in list(selected):
                trial_cols = [col for col in selected if col != factor]
                trial_x = x_clean[trial_cols] if trial_cols else pd.DataFrame(index=y_clean.index)
                fit, aic = _fit_aic_ols(y_clean, trial_x)
                if current_aic - aic >= min_aic_improvement:
                    selected = trial_cols
                    current_fit = fit
                    current_aic = aic
                    removed = True
                    break
            if not removed:
                break

    return selected


def lasso_select_factors(
    y: pd.Series,
    x: pd.DataFrame,
    alpha_grid: Iterable[float] | None = None,
    tolerance: float = 1e-4,
) -> list[str]:
    """Select factors with a small LASSO grid and BIC-based model choice."""
    joined = pd.concat([y.rename("y"), x], axis=1).dropna()
    if joined.empty:
        return []

    y_clean = joined["y"]
    x_clean = joined.drop(columns=["y"])
    if alpha_grid is None:
        alpha_grid = np.logspace(-3, -1, 8)

    y_centered = y_clean - y_clean.mean()
    x_centered = x_clean - x_clean.mean()
    x_scaled = x_centered / x_centered.std(ddof=0).replace(0.0, 1.0)
    best_selected: list[str] = []
    best_bic = np.inf

    for alpha in alpha_grid:
        try:
            fit = sm.OLS(y_centered, x_scaled).fit_regularized(alpha=float(alpha), L1_wt=1.0)
            params = pd.Series(np.asarray(fit.params), index=x_scaled.columns)
        except Exception:
            continue

        selected = params.index[params.abs() > tolerance].tolist()
        if not selected:
            continue

        model, _ = _fit_aic_ols(y_clean, x_clean[selected])
        rss = float(np.sum(model.resid**2))
        n = len(y_clean)
        k = len(selected) + 1
        bic = n * log(max(rss, 1e-12) / n) + k * log(n)
        if bic < best_bic:
            best_bic = bic
            best_selected = selected

    return best_selected


def _dynamic_factor_fit(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int,
    method: str,
    rebalance_every: int = 5,
) -> dict[str, object]:
    """Fit a rolling factor model with dynamic factor selection."""
    joined = pd.concat([asset_returns.rename("asset"), factor_returns], axis=1).dropna().sort_index()
    residuals = pd.Series(index=joined.index, dtype=float)
    predictions = pd.Series(index=joined.index, dtype=float)
    beta_cols = ["const"] + list(factor_returns.columns)
    betas = pd.DataFrame(index=joined.index, columns=beta_cols, dtype=float)
    selection_flags = pd.DataFrame(0.0, index=joined.index, columns=factor_returns.columns)
    selected_counts = pd.Series(index=joined.index, dtype=float)
    selected: list[str] = []

    for pos in range(window, len(joined)):
        train = joined.iloc[pos - window:pos].dropna()
        if len(train) < max(30, int(window * 0.8)):
            continue

        y_train = train["asset"]
        x_train = train.drop(columns=["asset"])
        if (pos - window) % rebalance_every == 0 or not selected:
            if method == "stepwise":
                selected = stepwise_select_factors(y_train, x_train)
            elif method == "lasso":
                selected = lasso_select_factors(y_train, x_train)
            else:
                raise ValueError(f"Unknown method: {method}")

        if selected:
            fit, _ = _fit_aic_ols(y_train, x_train[selected])
            params = fit.params
            current_x = sm.add_constant(joined.iloc[[pos]].drop(columns=["asset"])[selected], has_constant="add")
            pred = float(fit.predict(current_x).iloc[0])
        else:
            params = pd.Series({"const": float(y_train.mean())})
            pred = float(y_train.mean())

        current_date = joined.index[pos]
        predictions.loc[current_date] = pred
        residuals.loc[current_date] = float(joined.iloc[pos]["asset"] - pred)
        selected_counts.loc[current_date] = float(len(selected))
        betas.loc[current_date, "const"] = float(params.get("const", np.nan))
        for factor in factor_returns.columns:
            value = float(params.get(factor, 0.0)) if factor in selected else 0.0
            betas.loc[current_date, factor] = value
            selection_flags.loc[current_date, factor] = 1.0 if factor in selected else 0.0

    return {
        "residuals": residuals,
        "predictions": predictions,
        "betas": betas,
        "selection_flags": selection_flags,
        "selected_counts": selected_counts,
    }


def run_dynamic_selection_strategy(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    config: StrategyConfig,
    method: str = "stepwise",
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Run a single-asset strategy with dynamic factor selection."""
    fit = _dynamic_factor_fit(
        asset_returns=asset_returns,
        factor_returns=factor_returns,
        window=config.settings.regression_window,
        method=method,
    )
    residuals = fit["residuals"]
    betas = fit["betas"]
    zscore = rolling_zscore(residuals, window=config.settings.zscore_window)
    signal = build_hysteresis_signal(zscore, entry_z=config.settings.entry_z, exit_z=config.settings.exit_z)
    weights = build_pair_weight_frame(
        asset=str(asset_returns.name),
        signal=signal,
        betas=betas,
        residuals=residuals,
        factor_symbols=list(factor_returns.columns),
        config=config,
    )
    weights = apply_risk_caps(weights, config)
    returns = pd.concat([asset_returns.rename(str(asset_returns.name)), factor_returns], axis=1).dropna()
    backtest = run_backtest(returns=returns, target_weights=weights, config=config)
    daily = backtest.daily
    summary = performance_metrics(daily["net_return"])
    summary.update(
        {
            "avg_selected_factors": float(fit["selected_counts"].dropna().mean()),
            "selection_days": float((fit["selected_counts"].dropna().index.size)),
            "active_days": float((signal != 0).sum()),
        }
    )
    selection = fit["selection_flags"].copy()
    selection["selected_count"] = fit["selected_counts"]
    selection = selection.reset_index().rename(columns={"index": "date"})
    return daily, summary, selection


def build_ml_feature_frame(
    residuals: pd.Series,
    factor_returns: pd.DataFrame,
    horizon: int = 5,
    z_window: int = 20,
) -> pd.DataFrame:
    """Construct simple, interpretable features for residual mean reversion."""
    frame = pd.DataFrame(index=residuals.index)
    frame["residual"] = residuals
    frame["residual_z"] = rolling_zscore(residuals, window=z_window)
    frame["abs_residual_z"] = frame["residual_z"].abs()
    frame["residual_diff_1"] = residuals.diff(1)
    frame["residual_diff_3"] = residuals.diff(3)
    frame["residual_vol_20"] = residuals.rolling(20).std(ddof=1)
    frame["factor_mean_ret_3"] = factor_returns.mean(axis=1).rolling(3).mean()
    frame["factor_vol_20"] = factor_returns.std(axis=1).rolling(20).mean()
    future_abs = residuals.shift(-horizon).abs()
    frame["label"] = (future_abs < residuals.abs()).where(future_abs.notna()).astype(float)
    return frame


def run_walk_forward_logit(
    feature_frame: pd.DataFrame,
    window: int = 180,
) -> tuple[pd.Series, pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Walk-forward logistic regression for residual reversion probability."""
    data = feature_frame.dropna().copy()
    feature_cols = [col for col in data.columns if col != "label"]
    probabilities = pd.Series(index=data.index, dtype=float)
    last_params = pd.Series(dtype=float)
    rows: list[dict[str, float]] = []

    if len(data) < window + 10:
        window = max(40, min(window, len(data) // 2))

    for pos in range(window, len(data)):
        train = data.iloc[pos - window:pos].dropna()
        if train["label"].nunique() < 2:
            continue

        x_train = train[feature_cols]
        x_mean = x_train.mean()
        x_std = x_train.std(ddof=0).replace(0.0, 1.0)
        x_train_scaled = (x_train - x_mean) / x_std
        x_train_scaled = sm.add_constant(x_train_scaled, has_constant="add")
        y_train = train["label"]

        try:
            fit = sm.Logit(y_train, x_train_scaled).fit(disp=False, maxiter=100)
        except Exception:
            continue

        current = data.iloc[[pos]][feature_cols]
        current_scaled = (current - x_mean) / x_std
        current_scaled = sm.add_constant(current_scaled, has_constant="add")
        prob = float(fit.predict(current_scaled).iloc[0])
        current_date = data.index[pos]
        probabilities.loc[current_date] = prob
        last_params = fit.params.copy()
        rows.append(
            {
                "date": current_date,
                "probability": prob,
                "actual": float(data.iloc[pos]["label"]),
                "predicted": 1.0 if prob >= 0.5 else 0.0,
            }
        )

    prediction_table = pd.DataFrame(rows)
    if prediction_table.empty:
        summary = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "positive_rate": float(data["label"].mean()) if not data.empty else 0.0,
        }
        coef_table = pd.DataFrame(columns=["feature", "coef"])
        return probabilities, prediction_table, summary, coef_table

    actual = prediction_table["actual"]
    predicted = prediction_table["predicted"]
    tp = float(((predicted == 1.0) & (actual == 1.0)).sum())
    fp = float(((predicted == 1.0) & (actual == 0.0)).sum())
    fn = float(((predicted == 0.0) & (actual == 1.0)).sum())
    tn = float(((predicted == 0.0) & (actual == 0.0)).sum())
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)

    summary = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "positive_rate": float(data["label"].mean()),
        "predicted_positive_rate": float(predicted.mean()),
    }

    coef_table = pd.DataFrame(
        {
            "feature": last_params.index,
            "coef": last_params.values,
        }
    ) if not last_params.empty else pd.DataFrame(columns=["feature", "coef"])
    return probabilities, prediction_table, summary, coef_table


def apply_ml_gate(signal: pd.Series, probability: pd.Series, threshold: float = 0.55) -> pd.Series:
    """Keep a trade only when the ML classifier is confident."""
    aligned = pd.concat([signal.rename("signal"), probability.rename("prob")], axis=1).dropna()
    gated = aligned["signal"].where(aligned["prob"] >= threshold, 0.0)
    return gated.reindex(signal.index).fillna(0.0)


def run_bonus_suite(
    base_result: object,
    config: StrategyConfig,
    target_asset: str = "MSTR",
    basis_reference: str = "BTC",
) -> BonusSuiteResult:
    """Run all bonus analyses on top of the core strategy result."""
    prices = getattr(base_result, "prices")
    returns = getattr(base_result, "returns")
    models = getattr(base_result, "models")
    signals = getattr(base_result, "signals")
    pair_weights = getattr(base_result, "pair_weights")

    # 1) Cross-market basis arbitrage on real Binance spot/perp data.
    basis_daily = pd.DataFrame()
    basis_summary: dict[str, float] = {}
    try:
        basis_spot = prices[basis_reference]
        basis_perp = download_binance_archive_close_series(
            f"{basis_reference}/USDT",
            start_date=str(basis_spot.index.min().date()),
            end_date=str(basis_spot.index.max().date()),
            name=f"{basis_reference}_perp",
            market="um_futures",
        )
        basis_daily, basis_summary = run_basis_arbitrage(
            spot=basis_spot,
            perp=basis_perp,
            window=max(30, config.settings.zscore_window),
            entry_z=config.settings.entry_z,
            exit_z=config.settings.exit_z,
            notional=config.settings.capital_per_signal,
        )
        basis_summary["basis_source"] = "binance_real"
    except Exception:
        basis_summary["basis_source"] = "unavailable"

    # 2) Dynamic factor selection on one representative asset.
    candidate_factors = [factor for factor in config.factor_symbols if factor in returns.columns]
    dynamic_daily, dynamic_summary, dynamic_selection = run_dynamic_selection_strategy(
        asset_returns=returns[target_asset],
        factor_returns=returns[candidate_factors],
        config=config,
        method="stepwise",
    )
    # Also evaluate LASSO selection on the same window for comparison.
    dynamic_lasso_daily, dynamic_lasso_summary, dynamic_lasso_selection = run_dynamic_selection_strategy(
        asset_returns=returns[target_asset],
        factor_returns=returns[candidate_factors],
        config=config,
        method="lasso",
    )
    dynamic_summary["lasso_sharpe"] = dynamic_lasso_summary["sharpe"]
    dynamic_summary["lasso_avg_selected_factors"] = dynamic_lasso_summary["avg_selected_factors"]

    # 3) ML gating on the static residuals of the same asset.
    static_model = models[target_asset]
    feature_frame = build_ml_feature_frame(
        residuals=static_model.residuals,
        factor_returns=returns[candidate_factors],
        horizon=5,
        z_window=config.settings.zscore_window,
    )
    probability, prediction_table, ml_classification, ml_coefficients = run_walk_forward_logit(
        feature_frame=feature_frame,
        window=config.settings.regression_window,
    )
    gated_signal = apply_ml_gate(
        signal=signals[target_asset],
        probability=probability,
        threshold=0.50,
    )
    gated_weights = build_pair_weight_frame(
        asset=target_asset,
        signal=gated_signal,
        betas=static_model.betas,
        residuals=static_model.residuals,
        factor_symbols=candidate_factors,
        config=config,
    )
    gated_weights = apply_risk_caps(gated_weights, config)
    ml_returns = pd.concat([returns[target_asset].rename(target_asset), returns[candidate_factors]], axis=1).dropna()
    ml_backtest = run_backtest(returns=ml_returns, target_weights=gated_weights, config=config)
    ml_summary = performance_metrics(ml_backtest.daily["net_return"])
    ml_summary.update(
        {
            "active_days": float((gated_signal != 0).sum()),
            "avg_probability": float(probability.dropna().mean()) if not probability.dropna().empty else 0.0,
        }
    )
    # Compare against the base static pair for the same asset.
    static_pair_weights = pair_weights[target_asset]
    static_returns = pd.concat([returns[target_asset].rename(target_asset), returns[candidate_factors]], axis=1).dropna()
    static_backtest = run_backtest(returns=static_returns, target_weights=static_pair_weights, config=config)
    static_summary = performance_metrics(static_backtest.daily["net_return"])
    ml_summary["static_pair_sharpe"] = static_summary["sharpe"]

    return BonusSuiteResult(
        basis_daily=basis_daily,
        basis_summary=basis_summary,
        dynamic_daily=dynamic_daily,
        dynamic_summary=dynamic_summary,
        dynamic_selection=pd.concat(
            {
                "stepwise": dynamic_selection.assign(method="stepwise"),
                "lasso": dynamic_lasso_selection.assign(method="lasso"),
            },
            ignore_index=True,
        ),
        ml_daily=ml_backtest.daily.assign(probability=probability.reindex(ml_backtest.daily.index)),
        ml_summary=ml_summary,
        ml_coefficients=ml_coefficients,
        ml_classification=ml_classification,
    )
