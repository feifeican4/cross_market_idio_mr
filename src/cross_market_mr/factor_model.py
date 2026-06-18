"""Rolling factor regression and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass
class RollingFactorModelResult:
    """Container for rolling regression outputs."""

    residuals: pd.Series
    predictions: pd.Series
    betas: pd.DataFrame
    r2: pd.Series
    nobs: pd.Series


def fit_rolling_factor_model(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int = 90,
    min_obs: int | None = None,
) -> RollingFactorModelResult:
    """Fit rolling OLS using past data only.

    The regression at date t uses rows [t-window, ..., t-1] and then predicts
    the asset return at t. This avoids look-ahead bias.
    """
    if min_obs is None:
        min_obs = max(30, int(window * 0.8))

    joined = pd.concat([asset_returns.rename("asset"), factor_returns], axis=1).sort_index()
    joined = joined.dropna()

    residuals = pd.Series(index=joined.index, dtype=float, name=f"{asset_returns.name}_resid")
    predictions = pd.Series(index=joined.index, dtype=float, name=f"{asset_returns.name}_pred")
    r2 = pd.Series(index=joined.index, dtype=float, name=f"{asset_returns.name}_r2")
    nobs = pd.Series(index=joined.index, dtype=float, name=f"{asset_returns.name}_nobs")
    beta_columns = ["const"] + list(factor_returns.columns)
    betas = pd.DataFrame(index=joined.index, columns=beta_columns, dtype=float)

    factor_columns = list(factor_returns.columns)
    values = joined[["asset", *factor_columns]].to_numpy(dtype=float)

    for pos in range(window, len(joined)):
        train_values = values[pos - window:pos]
        train = pd.DataFrame(train_values, columns=["asset", *factor_columns]).dropna()
        if len(train) < min_obs:
            continue

        y_train = train["asset"].to_numpy(dtype=float)
        x_raw = train[factor_columns].to_numpy(dtype=float)
        x_train = np.column_stack([np.ones(len(x_raw)), x_raw])
        params, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)

        current_values = values[pos]
        x_current = np.array([1.0, *current_values[1:]], dtype=float)
        pred = float(x_current @ params)

        fitted = x_train @ params
        resid_train = y_train - fitted
        ss_res = float(np.sum(resid_train ** 2))
        ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
        current_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        predictions.iloc[pos] = pred
        residuals.iloc[pos] = float(current_values[0] - pred)
        r2.iloc[pos] = current_r2
        nobs.iloc[pos] = float(len(y_train))
        betas.iloc[pos, :] = params

    return RollingFactorModelResult(
        residuals=residuals,
        predictions=predictions,
        betas=betas,
        r2=r2,
        nobs=nobs,
    )


def adf_test(series: pd.Series) -> dict[str, float]:
    """Run Augmented Dickey-Fuller test on a residual series."""
    clean = series.dropna()
    if len(clean) < 50:
        return {"adf_stat": float("nan"), "p_value": float("nan"), "nobs": float(len(clean))}
    stat, p_value, *_ = adfuller(clean, autolag="AIC")
    return {"adf_stat": float(stat), "p_value": float(p_value), "nobs": float(len(clean))}


def model_diagnostics(model: RollingFactorModelResult) -> dict[str, float]:
    """Summarize one asset's rolling factor model."""
    adf = adf_test(model.residuals)
    clean_r2 = model.r2.dropna()
    return {
        "avg_r2": float(clean_r2.mean()) if not clean_r2.empty else float("nan"),
        "median_r2": float(clean_r2.median()) if not clean_r2.empty else float("nan"),
        "adf_stat": adf["adf_stat"],
        "adf_p_value": adf["p_value"],
        "residual_obs": adf["nobs"],
    }
