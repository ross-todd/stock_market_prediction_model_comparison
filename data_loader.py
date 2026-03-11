# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# Ross Todd
# BSc (Hons) Software Development
# Honours Project 2026 - Stock Market Prediction Comparison Analysis
#
# Data Loader File — Downloads historical stock price data from Yahoo Finance using yfinance
# Caches data locally so reruns use identical data without re-fetching from yfinance, (saved_data) file
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

import os
import time
import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CACHE_FOLDER = "saved_data"
os.makedirs(CACHE_FOLDER, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#   DOWNLOAD SINGLE TICKER
#   This function handles downloading one ticker at a time from Yahoo
#   Finance. It flattens the column headers that yfinance sometimes returns
#   as a MultiIndex, makes sure Adj Close is always present, and fills any
#   missing business days so there are no gaps in the data.
# ══════════════════════════════════════════════════════════════════════════

def download_ticker(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    raw_data = yf.download(ticker, start=start_date, end=end_date,
                           progress=False, auto_adjust=False)

    if raw_data is None or raw_data.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns if yfinance returns them that way
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    # Fall back to Close if Adj Close is missing
    if 'Adj Close' not in raw_data.columns and 'Close' in raw_data.columns:
        raw_data['Adj Close'] = raw_data['Close']

    # Fill any missing business days forward then backward
    raw_data = raw_data.asfreq('B').ffill().bfill()

    return raw_data


# ══════════════════════════════════════════════════════════════════════════
#   LOAD SINGLE TICKER WITH CACHING
#   Checks if a cached CSV exists for this ticker and date range.
#   If found, loads from CSV. If not, downloads from yfinance and saves.
#   The cache filename includes the date range so if you change the
#   data window it will automatically re-download fresh data.
# ══════════════════════════════════════════════════════════════════════════

def load_ticker_cached(ticker: str, start_date: str, end_date: str,
                       verbose: bool = True) -> pd.DataFrame:

    ticker_clean = ticker.replace(".", "_")
    start_clean  = start_date.replace("-", "")
    end_clean    = end_date.replace("-", "")
    cache_path   = os.path.join(CACHE_FOLDER,
                                f"{ticker_clean}_{start_clean}_{end_clean}.csv")

    if os.path.exists(cache_path):
        if verbose:
            print(f"    {ticker}: loaded from cache ({cache_path})")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    if verbose:
        print(f"    {ticker}: not in cache, downloading from yfinance...")

    df = download_ticker(ticker, start_date, end_date)
    df.to_csv(cache_path)

    if verbose:
        print(f"    {ticker}: {len(df)} observations saved to cache ({cache_path})")

    return df


# ══════════════════════════════════════════════════════════════════════════
#   LOAD ALL TICKERS
#   Loops through the list of tickers and loads each one in turn.
#   Uses the cache if available, otherwise downloads from yfinance.
#   An optional delay between downloads is included to avoid hitting
#   Yahoo Finance rate limits. Any ticker that fails gets skipped and
#   a message is printed, so the rest of the script can still run.
# ══════════════════════════════════════════════════════════════════════════

def load_all_tickers(
    tickers: list,
    start_date: str,
    end_date: str,
    delay: float = 0.0,
    verbose: bool = True,
) -> dict:

    downloaded_data = {}

    for ticker_idx, ticker in enumerate(tickers):
        if verbose:
            print(f"[data_loader] Loading {ticker} ({ticker_idx+1}/{len(tickers)})...")

        if delay > 0 and ticker_idx > 0:
            if verbose:
                print(f"   Sleeping {delay}s...")
            time.sleep(delay)

        try:
            df = load_ticker_cached(ticker, start_date, end_date, verbose=verbose)
            downloaded_data[ticker] = df

            if verbose:
                print(f"   Date range: {df.index[0].date()} → {df.index[-1].date()}")

        except Exception as download_error:
            print(f"   ✗ {ticker} failed: {download_error}")

    return downloaded_data