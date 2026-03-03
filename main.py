"""
End-to-end trading pipeline: data -> label -> infer -> execute -> backtest.

This is a demonstration pipeline for XAUUSD trading using a fine-tuned LLM.

Usage (after setting up data):
    python main.py --mode backtest --model_path models/checkpoints/xauusd_lora
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Local imports
from src.models.llm_wrapper import LLMWrapper
from src.trading.executor import PaperExecutor


def load_market_data(csv_path: str, symbol: str = "XAUUSD") -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"Loaded {len(df)} candles for {symbol}")
    return df


def add_indicators(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add simple indicators to data."""
    import numpy as np
    
    df = df.copy()
    df["sma"] = df["close"].rolling(window=window).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    return df


def build_prompt_for_inference(df: pd.DataFrame, idx: int, look_back: int = 5) -> str:
    """Build text prompt from market data at index."""
    if idx < look_back:
        return None
    
    rows = df.loc[max(0, idx - look_back) : idx].tail(look_back + 1)
    
    candle_str = "Recent XAUUSD price action:\n"
    for i, (_, row) in enumerate(rows.iterrows()):
        candle_str += (
            f"  Bar {i+1}: O={row['open']:.2f}, H={row['high']:.2f}, "
            f"L={row['low']:.2f}, C={row['close']:.2f}\n"
        )
    
    if idx < len(df):
        row = df.loc[idx]
        if not pd.isna(row.get("sma")):
            candle_str += f"  SMA({look_back}): {row['sma']:.2f}\n"
        if not pd.isna(row.get("rsi")):
            candle_str += f"  RSI({look_back}): {row['rsi']:.2f}\n"
    
    candle_str += (
        "Based on this pattern, should we BUY, SELL, or HOLD on the next 5-minute candle? "
        "Respond with exactly one word: BUY, SELL, or HOLD."
    )
    
    return candle_str


def backtest(
    csv_path: str,
    model_path: str,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.1",
    symbol: str = "XAUUSD",
    initial_capital: float = 10000.0,
):
    """
    Run backtest: replay market data, generate signals, execute trades.
    """
    print(f"\n=== Starting Backtest for {symbol} ===\n")
    
    # Load data
    df = load_market_data(csv_path, symbol)
    df = add_indicators(df)
    
    # Initialize model
    print(f"Loading model from {base_model}...")
    try:
        model = LLMWrapper(
            model_name_or_path=base_model,
            adapter_path=model_path,
            use_4bit=False,
            device="cpu",
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to demo mode (random signals)")
        model = None
    
    # Initialize executor
    executor = PaperExecutor(
        initial_capital=initial_capital,
        max_position_size=50000.0,
        max_daily_loss=500.0,
        slippage_pips=1.0,
        fee_per_trade=2.0,
    )
    
    # Simulation parameters
    open_trade = None
    position_size = 1000.0  # notional position size
    trade_count = 0
    
    # Backtest loop
    print(f"\nRunning simulation over {len(df)} candles...\n")
    
    for idx in range(len(df)):
        if idx < 5:
            continue  # Skip first few candles (need history)
        
        row = df.iloc[idx]
        timestamp = row["time"].isoformat()
        close_price = row["close"]
        
        # Build prompt and get signal
        if model is not None:
            prompt = build_prompt_for_inference(df, idx, look_back=5)
            if prompt:
                action = model.predict_action(prompt)
            else:
                action = "HOLD"
        else:
            # Demo: random decision (remove for real)
            import random
            action = random.choice(["BUY", "SELL", "HOLD"])
        
        # Trading logic
        if action == "BUY" and open_trade is None:
            open_trade = executor.execute(
                symbol=symbol,
                side="BUY",
                size=position_size,
                current_price=close_price,
                timestamp=timestamp,
            )
            if open_trade:
                trade_count += 1
        
        elif action == "SELL" and open_trade is not None:
            executor.close_position(symbol, open_price, timestamp)
            open_trade = None
        
        # Debug: print every 50 candles
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(df)}] Price: {close_price:.2f}, Action: {action}, Open trade: {open_trade is not None}")
    
    # Close any remaining open position
    if open_trade:
        executor.close_position(symbol, df.iloc[-1]["close"], df.iloc[-1]["time"].isoformat())
    
    # Print final stats
    stats = executor.get_statistics()
    print(f"\n=== Final Statistics ===")
    print(f"  Total trades executed: {stats['total_trades']}")
    print(f"  Closed trades: {stats['closed_trades']}")
    print(f"  Total P&L: ${stats['total_pnl']:.2f}")
    print(f"  Win rate: {stats['win_rate']*100:.1f}%")
    print(f"  Current balance: ${stats['current_balance']:.2f}")
    print(f"  Return: {stats['return_pct']:.2f}%")
    
    # Log trades
    executor.log_trades("logs/backtest_trades.jsonl")
    
    return executor


def demo_inference(model_name: str, adapter_path: str):
    """Test inference on a single sample."""
    print(f"\n=== Testing Inference ===\n")
    
    model = LLMWrapper(
        model_name_or_path=model_name,
        adapter_path=adapter_path,
        use_4bit=False,
        device="cpu",
    )
    
    sample_prompt = """Recent XAUUSD price action:
  Bar 1: O=2100.0, H=2105.0, L=2098.0, C=2103.0
  Bar 2: O=2103.0, H=2110.0, L=2102.0, C=2108.0
  Bar 3: O=2108.0, H=2112.0, L=2107.0, C=2111.0
  
Based on this pattern, should we BUY, SELL, or HOLD on the next 5-minute candle? Respond with exactly one word: BUY, SELL, or HOLD."""
    
    print("Sample prompt:")
    print(sample_prompt)
    
    action = model.predict_action(sample_prompt)
    print(f"\nModel output: {action}")


def main():
    parser = argparse.ArgumentParser(description="Trading LLM pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "backtest"],
        help="Run mode",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/raw/xauusd_ohlcv.csv",
        help="Path to OHLCV CSV",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/checkpoints/xauusd_lora",
        help="Path to fine-tuned adapter",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Base model name",
    )
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demo_inference(args.base_model, args.model_path)
    elif args.mode == "backtest":
        if not Path(args.csv_path).exists():
            print(f"Error: {args.csv_path} not found. Run src/data/download_fx.py first.")
            return
        backtest(args.csv_path, args.model_path, args.base_model)


if __name__ == "__main__":
    main()
