# ═════════════════════════════════════════════════════════════════════════════════════════════
# Ross Todd
# BSc (Hons) Software Development
# Honours Project 2026 - Stock Market Prediction Comparison Analysis
#
# Data Loader File — Downloads historical stock price data from Yahoo Finance using yfinance
# ═════════════════════════════════════════════════════════════════════════════════════════════

import time
import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


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
#   LOAD ALL TICKERS
#   Loops through the list of tickers and downloads each one in turn.
#   An optional delay between downloads is included to avoid hitting
#   Yahoo Finance rate limits, which is mainly needed for the GRU script
#   where the download is slower. Any ticker that fails gets skipped and
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
            print(f"[data_loader] Downloading {ticker} ({ticker_idx+1}/{len(tickers)})...")

        if delay > 0 and ticker_idx > 0:
            if verbose:
                print(f"   Sleeping {delay}s...")
            time.sleep(delay)

        try:
            df = download_ticker(ticker, start_date, end_date)
            downloaded_data[ticker] = df

            if verbose:
                print(f"   ✓ {ticker}: {len(df)} observations "
                      f"({df.index[0].date()} → {df.index[-1].date()})")

        except Exception as download_error:
            print(f"   ✗ {ticker} failed: {download_error}")

    return downloaded_data
