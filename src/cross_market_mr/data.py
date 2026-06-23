"""Data download, caching, and return panel construction."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from zipfile import ZipFile

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests

from .config import InstrumentConfig, StrategyConfig


def _safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _normalize_series(series: pd.Series, name: str, normalize_dates: bool = True) -> pd.Series:
    clean = series.dropna().copy()
    clean.index = pd.to_datetime(clean.index).tz_localize(None)
    if normalize_dates:
        clean.index = clean.index.normalize()
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    clean.name = name
    return clean.astype(float)


def _binance_archive_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace(":USDT", "")


def _binance_archive_months(start_date: str, end_date: str) -> pd.PeriodIndex:
    start = pd.Timestamp(start_date).to_period("M")
    end = pd.Timestamp(end_date).to_period("M")
    return pd.period_range(start=start, end=end, freq="M")


def download_binance_archive_close_series(
    symbol: str,
    start_date: str,
    end_date: str,
    name: str | None = None,
    interval: str = "1d",
    market: Literal["spot", "um_futures"] = "spot",
    normalize_dates: bool = True,
) -> pd.Series:
    """Download Binance public historical kline archive."""
    archive_symbol = _binance_archive_symbol(symbol)
    market_path = "spot" if market == "spot" else "futures/um"
    frames = []
    for month in _binance_archive_months(start_date, end_date):
        url = (
            f"https://data.binance.vision/data/{market_path}/monthly/klines/"
            f"{archive_symbol}/{interval}/{archive_symbol}-{interval}-{month.strftime('%Y-%m')}.zip"
        )
        try:
            with urlopen(url, timeout=30) as response:
                payload = response.read()
        except (HTTPError, URLError, TimeoutError):
            continue
        with ZipFile(BytesIO(payload)) as archive:
            csv_name = archive.namelist()[0]
            frame = pd.read_csv(archive.open(csv_name), header=None)
            if str(frame.iloc[0, 0]).lower() == "open_time":
                frame = frame.iloc[1:].reset_index(drop=True)
        frames.append(frame)

    if not frames:
        raise ValueError(f"No Binance archive data for {symbol}")

    data = pd.concat(frames, ignore_index=True)
    data = data.iloc[:, :12]
    data.columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    data["open_time"] = pd.to_numeric(data["open_time"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["open_time", "close"])
    microsecond_rows = data["open_time"] > 10_000_000_000_000
    dates = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    if microsecond_rows.any():
        dates.loc[microsecond_rows] = pd.to_datetime(
            data.loc[microsecond_rows, "open_time"],
            unit="us",
            utc=True,
        ).dt.tz_convert(None)
    if (~microsecond_rows).any():
        dates.loc[~microsecond_rows] = pd.to_datetime(
            data.loc[~microsecond_rows, "open_time"],
            unit="ms",
            utc=True,
        ).dt.tz_convert(None)
    data["date"] = dates
    series = data.set_index("date")["close"]
    series = series.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    return _normalize_series(series, name or symbol, normalize_dates=normalize_dates)


def download_nasdaq_close_series(
    ticker: str,
    start_date: str,
    end_date: str,
    name: str | None = None,
    assetclass: str = "stocks",
) -> pd.Series:
    """Download US stock/ETF daily close data from Nasdaq's public JSON API."""
    url = (
        f"https://api.nasdaq.com/api/quote/{ticker}/historical?"
        f"assetclass={assetclass}&fromdate={start_date}&todate={end_date}&limit=9999"
    )
    headers = {
        "accept": "application/json",
        "origin": "https://www.nasdaq.com",
        "referer": f"https://www.nasdaq.com/market-activity/{assetclass}/{ticker.lower()}/historical",
        "user-agent": "Mozilla/5.0",
    }
    response = requests.get(url, impersonate="chrome120", timeout=30, headers=headers)
    response.raise_for_status()
    payload = response.json()
    rows = (((payload.get("data") or {}).get("tradesTable") or {}).get("rows")) or []
    if not rows:
        raise ValueError(f"No Nasdaq data for {ticker}")
    frame = pd.DataFrame(rows)
    close = (
        frame["close"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    series = pd.Series(
        close.astype(float).to_numpy(),
        index=pd.to_datetime(frame["date"], format="%m/%d/%Y"),
        name=name or ticker,
    )
    return _normalize_series(series, name or ticker)


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
            return download_binance_archive_close_series(
                instrument.ticker,
                start_date=start_date,
                end_date=end_date,
                name=instrument.symbol,
            )
        except Exception:
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

    if instrument.kind == "crypto_spot":
        try:
            return download_binance_archive_close_series(
                instrument.ticker.replace("-USD", "/USDT"),
                start_date=start_date,
                end_date=end_date,
                name=instrument.symbol,
            )
        except Exception:
            return download_yfinance_close_series(
                instrument.ticker,
                start_date=start_date,
                end_date=end_date,
                name=instrument.symbol,
            )

    assetclass = "etf" if instrument.ticker in {"SPY", "QQQ", "SMH"} else "stocks"
    try:
        return download_nasdaq_close_series(
            instrument.ticker,
            start_date=start_date,
            end_date=end_date,
            name=instrument.symbol,
            assetclass=assetclass,
        )
    except Exception:
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
