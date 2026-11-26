"""Data Loader & Parallel Yahoo Finance Updater for TradingPro v3.

Reuses the existing `~/TradingApp_Cache` directory for full compatibility,
uses `auto_adjust=True` for clean split/dividend-adjusted downloads, and runs
parallel updates with cancellation support so the UI never freezes.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf


USER_DIR = os.path.expanduser("~")
CACHE_DIR = os.path.join(USER_DIR, "TradingApp_Cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TICKER_MAPPING = {
    "BITCOIN": "BTC-USD",
    "BTC": "BTC-USD",
    "ETHEREUM": "ETH-USD",
    "ETH": "ETH-USD",
    "SOLANA": "SOL-USD",
    "SOL": "SOL-USD",
}


def resolve_ticker(raw: str) -> str:
    clean = str(raw).upper().strip().replace("_DATEN", "").replace(".CSV", "")
    return TICKER_MAPPING.get(clean, clean)


def smart_extract_ticker(path: str) -> Optional[str]:
    try:
        df_head = pd.read_csv(path, nrows=5, header=None)
        for _, row in df_head.iterrows():
            if str(row[0]).strip() == "Ticker":
                return resolve_ticker(str(row[1]))
    except Exception:
        pass
    return None


def get_cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{resolve_ticker(ticker)}.csv")


def _clean_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            if col == "Close" and "Adj Close" in df.columns:
                df[col] = df["Adj Close"]
            else:
                return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    else:
        df["Volume"] = 0.0

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df.index >= "2020-01-01"]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_ticker_data(
    ticker: str,
    source_type: str = "yahoo",
    file_path: Optional[str] = None,
    force_download: bool = False,
) -> Tuple[Optional[pd.DataFrame], str, str]:
    final_ticker = resolve_ticker(ticker)
    cache_file = get_cache_path(final_ticker)

    try:
        if source_type == "yahoo":
            if not force_download and os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
                df = _clean_ohlcv_df(df)
                if not df.empty:
                    return df, "OK (Cache)", final_ticker

            df = yf.download(
                final_ticker,
                start="2020-01-01",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            df = _clean_ohlcv_df(df)
            if not df.empty:
                df.index.name = "Date"
                df.to_csv(cache_file)
                return df, "OK (Yahoo)", final_ticker
            return None, "Keine Daten von Yahoo", final_ticker

        # CSV import
        df = pd.read_csv(file_path)
        if "Price" in df.columns:
            df.rename(columns={"Price": "Date"}, inplace=True)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.dropna(subset=["Date"], inplace=True)
            df.set_index("Date", inplace=True)
        elif "Close" not in df.columns:
            df = pd.read_csv(
                file_path,
                skiprows=3,
                header=None,
                names=["Date", "Close", "High", "Low", "Open", "Volume"],
            )
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.dropna(subset=["Date"], inplace=True)
            df.set_index("Date", inplace=True)

        df = _clean_ohlcv_df(df)
        if not df.empty:
            df.index.name = "Date"
            df.to_csv(cache_file)
            return df, "OK (CSV)", final_ticker
        return None, "CSV leer oder ungültig", final_ticker

    except Exception as exc:
        return None, str(exc), final_ticker


def fetch_company_name(ticker: str) -> str:
    final_ticker = resolve_ticker(ticker)
    try:
        info = yf.Ticker(final_ticker).info
        return str(info.get("shortName") or info.get("longName") or final_ticker)
    except Exception:
        return final_ticker


def download_tickers_parallel(
    tickers: List[str],
    max_workers: int = 6,
    cancel_event: Optional[threading.Event] = None,
    on_item_done: Optional[Callable[[int, int, str, bool, str], None]] = None,
    chunk_size: int = 20,
) -> Dict[str, Tuple[Optional[pd.DataFrame], str]]:
    """Downloads fresh Yahoo Finance data reliably in native chunks + automatic retry pass.

    Avoids yfinance's multi-thread session race & Yahoo HTTP 429 rate-limits that occur
    when firing 100+ individual ThreadPoolExecutor downloads simultaneously.
    """
    resolved_list = [resolve_ticker(t) for t in tickers]
    total = len(resolved_list)
    results: Dict[str, Tuple[Optional[pd.DataFrame], str]] = {}
    missing: List[str] = []
    done_count = 0

    for start_idx in range(0, total, chunk_size):
        if cancel_event is not None and cancel_event.is_set():
            break
        chunk = resolved_list[start_idx : start_idx + chunk_size]
        raw = None
        try:
            raw = yf.download(
                chunk,
                start="2020-01-01",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception:
            raw = None

        for sym in chunk:
            if cancel_event is not None and cancel_event.is_set():
                break
            df_clean = pd.DataFrame()
            try:
                if raw is not None and not raw.empty:
                    if len(chunk) == 1:
                        df_sub = raw.copy()
                        if isinstance(df_sub.columns, pd.MultiIndex):
                            df_sub.columns = df_sub.columns.droplevel(0)
                        df_clean = _clean_ohlcv_df(df_sub)
                    elif isinstance(raw.columns, pd.MultiIndex) and sym in raw.columns.get_level_values(0):
                        df_clean = _clean_ohlcv_df(raw[sym].copy())
            except Exception:
                df_clean = pd.DataFrame()

            if not df_clean.empty:
                df_clean.index.name = "Date"
                df_clean.to_csv(get_cache_path(sym))
                results[sym] = (df_clean, "OK (Yahoo Batch)")
                done_count += 1
                if on_item_done is not None:
                    on_item_done(done_count, total, sym, True, "OK")
            else:
                missing.append(sym)

        if start_idx + chunk_size < total and (cancel_event is None or not cancel_event.is_set()):
            time.sleep(0.45)

    # Automatic clean single-ticker retry pass for any stragglers
    for sym in missing:
        if cancel_event is not None and cancel_event.is_set():
            break
        time.sleep(0.35)
        df_retry, msg_retry, final_t = load_ticker_data(sym, "yahoo", force_download=True)
        ok = df_retry is not None and not df_retry.empty
        results[final_t] = (df_retry, "OK (Retry)" if ok else msg_retry)
        done_count += 1
        if on_item_done is not None:
            on_item_done(done_count, total, final_t, ok, msg_retry)

    return results
