"""Rolling factor regression and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
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

    for pos in range(window, len(joined)):
        train = joined.iloc[pos - window:pos].dropna()
        if len(train) < min_obs:
            continue

        y_train = train["asset"]
        x_train = sm.add_constant(train.drop(columns=["asset"]), has_constant="add")
        fit = sm.OLS(y_train, x_train).fit()

        current = joined.iloc[[pos]]
        x_current = sm.add_constant(current.drop(columns=["asset"]), has_constant="add")
        pred = float(fit.predict(x_current).iloc[0])

        predictions.iloc[pos] = pred
        residuals.iloc[pos] = float(current["asset"].iloc[0] - pred)
        r2.iloc[pos] = float(fit.rsquared)
        nobs.iloc[pos] = float(fit.nobs)
        for column in beta_columns:
            betas.loc[joined.index[pos], column] = float(fit.params.get(column, float("nan")))

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

