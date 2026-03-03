"""Simple data ingestion helpers (example using yfinance).
"""
from typing import Any
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


def fetch_ohlcv(symbol: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV data for `symbol` using yfinance (if installed).

    Returns a pandas DataFrame or raises if yfinance is not available.
    """
    if yf is None:
        raise RuntimeError("yfinance not installed. Install requirements.txt and try again.")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    return df


if __name__ == "__main__":
    df = fetch_ohlcv("AAPL", period="7d", interval="1h")
    print(df.tail())
