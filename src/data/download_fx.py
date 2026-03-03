"""
Download historical OHLCV data for XAUUSD and XAGUSD (5-minute candles).
Uses free and paid sources. Saves to data/raw/ as CSV.

Free options:
- Oanda (requires free account + API key)
- Alpha Vantage (limited free tier)
- CCXT (some exchanges offer free access)
- yfinance (limited for metals, mainly stocks)

For production, use a dedicated FX data vendor (TrueFX, Dukascopy, HistData, FXCM).
"""

import os
import pandas as pd
import requests
from typing import Optional
from pathlib import Path

# Create data directories if they don't exist
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)


def download_oanda(
    symbol: str,
    granularity: str = "M5",
    count: int = 5000,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download data from Oanda v20 API.
    
    Args:
        symbol: instrument (e.g., "XAU_USD" for Gold, "XAG_USD" for Silver)
        granularity: candle size ("M1", "M5", "H1", etc.)
        count: number of candles to download (max 5000 per request)
        api_key: Oanda API key (can be read from env var OANDA_API_KEY)
    
    Returns:
        pandas DataFrame with columns: time, open, high, low, close, volume
    """
    if api_key is None:
        api_key = os.getenv("OANDA_API_KEY")
        if not api_key:
            raise ValueError("OANDA_API_KEY not set. Set it in .env or env variables.")

    url = f"https://api-fxpractice.oanda.com/v3/instruments/{symbol}/candles"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept-Datetime-Format": "UNIX",
    }
    params = {
        "granularity": granularity,
        "count": count,
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    rows = []
    for candle in data.get("candles", []):
        rows.append({
            "time": pd.Timestamp(int(candle["time"]), unit="s"),
            "open": float(candle["bid"]["o"]),
            "high": float(candle["bid"]["h"]),
            "low": float(candle["bid"]["l"]),
            "close": float(candle["bid"]["c"]),
            "volume": int(candle.get("volume", 0)),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def download_from_csv_online(url: str, **kwargs) -> pd.DataFrame:
    """
    Download from a CSV URL (e.g., HistData, TradingView).
    
    Args:
        url: URL to CSV file
        **kwargs: passed to pd.read_csv
    
    Returns:
        pandas DataFrame
    """
    df = pd.read_csv(url, **kwargs)
    return df


def save_raw_data(df: pd.DataFrame, symbol: str) -> str:
    """
    Save raw OHLCV data to CSV.
    
    Args:
        df: DataFrame with time, open, high, low, close, volume
        symbol: symbol name (e.g., "XAUUSD")
    
    Returns:
        path to saved file
    """
    filename = f"data/raw/{symbol}_ohlcv.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} candles to {filename}")
    return filename


# Example: Download from Oanda
if __name__ == "__main__":
    import sys

    # Try to download from Oanda (requires API key in .env)
    try:
        print("Fetching XAUUSD (Gold) 5-minute candles from Oanda...")
        df_gold = download_oanda("XAU_USD", granularity="M5", count=5000)
        save_raw_data(df_gold, "XAUUSD")
        print(f"Downloaded {len(df_gold)} candles\n")

        print("Fetching XAGUSD (Silver) 5-minute candles from Oanda...")
        df_silver = download_oanda("XAG_USD", granularity="M5", count=5000)
        save_raw_data(df_silver, "XAGUSD")
        print(f"Downloaded {len(df_silver)} candles\n")

    except ValueError as e:
        print(f"Oanda download failed: {e}")
        print("\nNote: To use Oanda, sign up for a free practice account at https://www.oanda.com")
        print("and paste your API key in .env as OANDA_API_KEY=<your-key>")
        sys.exit(1)
