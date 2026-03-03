"""
Generate JSONL dataset for fine-tuning using spike-labeling heuristic.

Logic:
- Read 5-minute OHLCV data.
- For each candle, check if it's the end of a spike (25+ pips move with no reversals).
- Create prompt from preceding candles + indicators, label with action (buy/sell/hold).
- Save to data/datasets/train.jsonl and data/datasets/test.jsonl

Spike definition (customizable):
- spike_threshold: 25 pips move in one direction
- lookback_k: window size (e.g., 5 bars)
- no_reversal: all bars in window are in the same direction (all green or all red)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional

Path("data/datasets").mkdir(parents=True, exist_ok=True)


def compute_indicators(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute simple technical indicators to enrich context.
    
    Adds columns: sma, rsi, atr, volume_ma
    """
    df = df.copy()
    
    # Simple moving average
    df["sma"] = df["close"].rolling(window=window).mean()
    
    # RSI (simplified)
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # ATR (simplified)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    df["atr"] = df["tr"].rolling(window=window).mean()
    
    # Volume MA
    df["volume_ma"] = df["volume"].rolling(window=window).mean()
    
    return df


def detect_spike(
    df: pd.DataFrame,
    idx: int,
    spike_threshold: float = 25.0,
    lookback_k: int = 5,
    min_strength: int = 3,  # at least 3 consecutive bars in one direction
) -> Optional[str]:
    """
    Detect spike at index idx.
    
    Returns:
        "spike_up" if upward spike detected
        "spike_down" if downward spike detected
        None otherwise
    
    Args:
        df: DataFrame with OHLCV
        idx: current candle index
        spike_threshold: pip move threshold
        lookback_k: number of preceding candles to check
        min_strength: minimum consecutive bars in same direction
    """
    if idx < lookback_k:
        return None
    
    # Window: from (idx - lookback_k) to idx
    start_idx = idx - lookback_k
    close_start = df.loc[start_idx, "close"]
    close_end = df.loc[idx, "close"]
    
    move = close_end - close_start
    
    # Check direction
    if move >= spike_threshold:
        # Upward spike: check at least min_strength consecutive green candles
        bars = df.loc[start_idx : idx, ["open", "close"]].values
        green_count = sum(1 for o, c in bars if c > o)
        if green_count >= min_strength:
            return "spike_up"
    
    elif move <= -spike_threshold:
        # Downward spike
        bars = df.loc[start_idx : idx, ["open", "close"]].values
        red_count = sum(1 for o, c in bars if c < o)
        if red_count >= min_strength:
            return "spike_down"
    
    return None


def build_prompt(df: pd.DataFrame, idx: int, look_back: int = 5) -> str:
    """
    Build a text prompt from preceding candles and indicators.
    
    Format (novice-friendly):
    "Recent XAUUSD price action (5 bars):
    Bar 1: open 2100.0, high 2105.0, low 2098.0, close 2103.0
    Bar 2: open 2103.0, high 2110.0, low 2102.0, close 2108.0
    ...
    Current SMA: 2100.5, RSI: 65.0, ATR: 4.2
    Based on this pattern, should we buy, sell, or hold?"
    """
    if idx < look_back:
        return None
    
    rows = df.loc[max(0, idx - look_back) : idx].tail(look_back + 1)
    
    candle_str = "Recent XAUUSD/XAGUSD price action:\n"
    for i, (_, row) in enumerate(rows.iterrows()):
        candle_str += (
            f"  Bar {i+1}: O={row['open']:.2f}, H={row['high']:.2f}, "
            f"L={row['low']:.2f}, C={row['close']:.2f}\n"
        )
    
    # Add current indicators
    if idx < len(df):
        row = df.loc[idx]
        if not pd.isna(row.get("sma")):
            candle_str += f"  SMA({look_back}): {row['sma']:.2f}\n"
        if not pd.isna(row.get("rsi")):
            candle_str += f"  RSI({look_back}): {row['rsi']:.2f}\n"
        if not pd.isna(row.get("atr")):
            candle_str += f"  ATR({look_back}): {row['atr']:.4f}\n"
    
    candle_str += (
        "Based on this pattern, should we BUY, SELL, or HOLD on the next 5-minute candle? "
        "Respond with exactly one word: BUY, SELL, or HOLD."
    )
    
    return candle_str


def generate_dataset(
    csv_path: str,
    symbol: str,
    spike_threshold: float = 25.0,
    lookback_k: int = 5,
    test_split: float = 0.2,
    output_prefix: str = "data/datasets",
):
    """
    Generate training and test JSONL from raw OHLCV CSV.
    
    Args:
        csv_path: path to raw OHLCV CSV
        symbol: symbol name (e.g., "XAUUSD")
        spike_threshold: pip threshold for labeling
        lookback_k: window for spike detection
        test_split: fraction of data for test set
        output_prefix: where to save JSONL files
    """
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    
    print(f"[{symbol}] Loaded {len(df)} candles from {csv_path}")
    
    # Compute indicators
    df = compute_indicators(df)
    
    # Label spikes
    labels = []
    for idx in range(len(df)):
        spike = detect_spike(df, idx, spike_threshold=spike_threshold, lookback_k=lookback_k)
        if spike == "spike_up":
            labels.append("BUY")
        elif spike == "spike_down":
            labels.append("SELL")
        else:
            labels.append("HOLD")
    
    df["label"] = labels
    
    # Count labels
    label_counts = df["label"].value_counts()
    print(f"  Label distribution: {label_counts.to_dict()}")
    
    # Build dataset
    examples = []
    for idx in range(lookback_k, len(df)):
        prompt = build_prompt(df, idx, look_back=5)
        label = df.loc[idx, "label"]
        
        if prompt is not None:
            examples.append({
                "prompt": prompt,
                "response": label,
                "timestamp": df.loc[idx, "time"].isoformat(),
                "price": float(df.loc[idx, "close"]),
            })
    
    print(f"[{symbol}] Created {len(examples)} labeled examples")
    
    # Split into train/test (chronologically)
    split_idx = int(len(examples) * (1 - test_split))
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]
    
    # Save JSONL
    train_file = f"{output_prefix}/train_{symbol.lower()}.jsonl"
    test_file = f"{output_prefix}/test_{symbol.lower()}.jsonl"
    
    with open(train_file, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    
    with open(test_file, "w") as f:
        for ex in test_examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"  Saved {len(train_examples)} train examples to {train_file}")
    print(f"  Saved {len(test_examples)} test examples to {test_file}")
    
    return train_file, test_file


if __name__ == "__main__":
    # Generate datasets for both symbols
    for symbol_abbr, csv_name in [("XAUUSD", "xauusd"), ("XAGUSD", "xagusd")]:
        csv_path = f"data/raw/{csv_name}_ohlcv.csv"
        if Path(csv_path).exists():
            print(f"\nGenerating dataset for {symbol_abbr}...")
            generate_dataset(csv_path, symbol_abbr, spike_threshold=25.0)
        else:
            print(f"Warning: {csv_path} not found. Run src/data/download_fx.py first.")
