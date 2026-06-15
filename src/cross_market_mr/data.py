"""Data download, caching, and return panel construction."""

from __future__ import annotations

from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf

from .config import InstrumentConfig, StrategyConfig


def _safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _normalize_series(series: pd.Series, name: str) -> pd.Series:
    clean = series.dropna().copy()
    clean.index = pd.to_datetime(clean.index).tz_localize(None).normalize()
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    clean.name = name
    return clean.astype(float)


def download_yfinance_close_series(
    ticker: str,
    start_date: str,
    end_date: str,
    name: str | None = None,
) -> pd.Series:
    """Download adjusted close data from Yahoo Finance."""
    end_plus_one = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    frame = yf.download(
        ticker,
        start=start_date,
        end=end_plus_one,
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if frame.empty:
        raise ValueError(f"No yfinance data for {ticker}")

    if "Close" in frame.columns:
        close = frame["Close"]
    elif "Adj Close" in frame.columns:
        close = frame["Adj Close"]
    else:
        close = frame.iloc[:, 0]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return _normalize_series(close, name or ticker)


def download_binance_close_series(
    symbol: str,
    start_date: str,
    end_date: str,
    name: str | None = None,
    timeframe: str = "1d",
) -> pd.Series:
    """Download daily close data from Binance via ccxt."""
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ms = exchange.parse8601(f"{end_date}T00:00:00Z") + 24 * 60 * 60 * 1000
    rows: list[list[float]] = []

    while since < end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1

    if not rows:
        raise ValueError(f"No Binance data for {symbol}")

    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    series = frame.groupby(frame["datetime"].dt.normalize())["close"].last()
    return _normalize_series(series, name or symbol)


def fallback_yfinance_symbol(symbol: str) -> str:
    """Map Binance-style symbols to Yahoo Finance tickers."""
    return symbol.replace("/USDT", "-USD").replace("/BUSD", "-USD")


def download_instrument_series(
    instrument: InstrumentConfig,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Download one instrument with source-aware fallback."""
    if instrument.source == "binance":
        try:
            return download_binance_close_series(
                instrument.ticker,
                start_date=start_date,
                end_date=end_date,
                name=instrument.symbol,
            )
        except Exception:
            fallback_ticker = fallback_yfinance_symbol(instrument.ticker)
            return download_yfinance_close_series(
                fallback_ticker,
                start_date=start_date,
                end_date=end_date,
                name=instrument.symbol,
            )

    return download_yfinance_close_series(
        instrument.ticker,
        start_date=start_date,
        end_date=end_date,
        name=instrument.symbol,
    )


def load_or_download_series_map(
    config: StrategyConfig,
    cache_dir: str | Path | None = None,
) -> tuple[dict[str, pd.Series], list[str]]:
    """Load series from cache if present, otherwise download them."""
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    series_map: dict[str, pd.Series] = {}
    missing: list[str] = []
    for symbol in config.instruments:
        instrument = config.instruments[symbol]
        cache_file = None
        if cache_path is not None:
            cache_file = cache_path / f"{_safe_name(symbol)}.parquet"
        try:
            if cache_file is not None and cache_file.exists():
                cached = pd.read_parquet(cache_file).iloc[:, 0]
                series_map[symbol] = _normalize_series(cached, symbol)
            else:
                series = download_instrument_series(
                    instrument,
                    start_date=config.settings.start_date,
                    end_date=config.settings.end_date,
                )
                series_map[symbol] = series
                if cache_file is not None:
                    series.to_frame(name=symbol).to_parquet(cache_file)
        except Exception:
            missing.append(symbol)

    return series_map, missing


def build_price_panel(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Outer-join individual price series into one panel."""
    if not series_map:
        return pd.DataFrame()
    panel = pd.concat(series_map, axis=1).sort_index()
    panel.index.name = "date"
    return panel


def compute_return_panel(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute log returns per instrument before alignment."""
    returns = {}
    for symbol, series in series_map.items():
        clean = series.dropna().sort_index()
        ret = np.log(clean).diff()
        ret.name = symbol
        returns[symbol] = ret.dropna()
    if not returns:
        return pd.DataFrame()
    panel = pd.concat(returns, axis=1).sort_index()
    panel.index.name = "date"
    return panel


def save_panel(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a panel to parquet."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target)


def load_panel(path: str | Path) -> pd.DataFrame:
    """Load a parquet panel if it exists."""
    return pd.read_parquet(path)

