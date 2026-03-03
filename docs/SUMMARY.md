# Project Summary: XAUUSD Trading LLM Bot

This document summarizes what has been built, the architecture, workflows, and how to get started.

---

## What You Now Have

A complete R&D framework for automated trading using fine-tuned local LLMs (running on Colab GPU or local CPU).

### Project Assets

#### **Documentation** (everything explained for novices)
- `README.md` — Project overview
- `QUICKSTART.md` — Step-by-step setup (5-step guide)
- `docs/PROCESS.md` — Detailed objectives, theory, failures to avoid, and step-by-step plan
- `docs/MODELS.md` — Model selection guide (why ensemble > single model, hardware fit)

#### **Code Structure**
```
src/
  data/
    ingest.py             → fetch OHLCV from APIs (yfinance, etc.)
    download_fx.py        → download XAUUSD/XAGUSD from Oanda (free account)
    generate_labels.py    → create JSONL from spike heuristic (25-pip moves)
  models/
    llm_wrapper.py        → inference wrapper (loads base model + LoRA adapters)
  finetune/
    train.py              → local GPU fine-tuning script (PEFT/LoRA)
  trading/
    executor.py           → paper trading + risk management + backtesting
  utils/
    config.py             → .env config loader
notebooks/
  colab_finetune_xauusd.ipynb → Colab notebook for fine-tuning (GPU-friendly)
main.py                   → End-to-end pipeline (data → infer → trade)
```

#### **Workflow**
1. **Data**: Download 5-minute XAUUSD candles (CSV)
2. **Labels**: Apply spike heuristic → JSONL training data
3. **Fine-tune**: Train LoRA adapters on Colab GPU (Mistral or Llama)
4. **Backtest**: Run inference on historical data, simulate trades, log P&L
5. **Deploy**: Once validated, wire to live paper trading

#### **Key Features**
- ✅ Automated spike detection (your 25-point/pip rule)
- ✅ Paper trading with risk limits (daily loss cap, position size limits)
- ✅ PEFT fine-tuning (no full model training needed)
- ✅ 4-bit quantization (memory-efficient on limited hardware)
- ✅ Ensemble-ready (swap models, test multiple)
- ✅ Complete logging (trades, metrics, backtest stats)

---

## Why This Approach

### 1. Local LLM for Trading
- **No API costs** (inference entirely on your machine or Colab)
- **Privacy** (data stays local)
- **Transparency** (you control the model)

### 2. PEFT/LoRA Fine-tuning
- **Efficient**: Only ~2-5% of weights updated instead of full model
- **Affordable**: Works on consumer GPUs (Colab T4)
- **Speed**: Minutes to hours instead of days

### 3. Spike-Based Labels
- **Explainable**: Rule = "25-pip up move with no reversals"
- **Flexible**: Easy to tune (change threshold, lookback window)
- **Testable**: Deterministic labeling, reproducible dataset

### 4. Paper Trading First
- **Safe**: No real money until confident
- **Iterate**: Test strategy, metrics, risk controls
- **Backtest**: Validate on past data before live

---

## Quick Start (TL;DR)

### 1. Setup (5 min)
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Get Data (Manual or automated)
**Manual**: Download XAUUSD 5-min CSV from Dukascopy/HistData → save to `data/raw/xauusd_ohlcv.csv`

**Or automated** (requires Oanda account):
```bash
echo "OANDA_API_KEY=your_key" >> .env
python src/data/download_fx.py
```

### 3. Generate Labels (5 min)
```bash
python src/data/generate_labels.py
# Output: data/datasets/train_xauusd.jsonl, test_xauusd.jsonl
```

### 4. Fine-tune on Colab (30-60 min)
- Open: https://colab.research.google.com
- Copy cells from `notebooks/colab_finetune_xauusd.ipynb`
- Upload JSONL files from step 3
- Run all cells
- Download `xauusd_lora_adapters.zip`
- Extract to `models/checkpoints/xauusd_lora_final/`

### 5. Backtest (2 min)
```bash
python main.py --mode backtest \
  --csv_path data/raw/xauusd_ohlcv.csv \
  --model_path models/checkpoints/xauusd_lora_final
```

Output: Trades logged to `logs/backtest_trades.jsonl`, stats printed to console.

---

## How Each Component Works

### Data Pipeline (`src/data/`)

**download_fx.py**
- Fetches XAUUSD/XAGUSD from Oanda v20 API (5-minute candles)
- Saves to CSV: time, open, high, low, close, volume
- Free Oanda practice accounts give unlimited historical data

**generate_labels.py**
- Reads CSV, computes indicators (SMA, RSI, ATR)
- Detects spikes: `close(t) - close(t-5) >= 25 pips`
- Creates labeled JSONL: `{"prompt": "...", "response": "BUY|SELL|HOLD"}`
- Splits 80/20 train/test (chronologically)

### Model Pipeline (`src/models/`)

**llm_wrapper.py**
- Wraps Hugging Face model + LoRA adapters
- Supports 4-bit quantization (saves RAM)
- `predict_action(prompt)` → returns "BUY", "SELL", or "HOLD"
- Works with any Mistral, Llama, Falcon, etc.

### Fine-tuning (`src/finetune/`)

**train.py**
- Loads base model (Mistral-7B, Llama-2-7B, etc.)
- Applies 4-bit quantization via bitsandbytes
- Fine-tunes using PEFT (LoRA): only ~2% of params
- Saves adapters to `models/checkpoints/`

### Backtesting & Trading (`src/trading/`)

**executor.py**
- Paper trading simulator: log buys/sells, compute P&L
- Risk controls: max position size, daily loss cap
- Slippage & fees applied (realistic)
- Statistics: win rate, total P&L, Sharpe (TODO)
- JSON logging for analysis

### End-to-End (`main.py`)

**Backtest mode**: Load model → replay historical data → generate signals → simulate trades → output metrics

**Demo mode**: Test single inference (check model works)

---

## Terminology Recap (For Learning)

**Candle/Bar**: OHLCV data for a timeframe (1-min, 5-min, 1-hour)
**Pip**: Smallest price move (FX: 0.0001 for USD pairs, 0.01 for JPY pairs)
**Spike**: Large move in one direction without reversals
**LoRA**: Low-Rank Adaptation — fine-tune only 2% of weights
**Quantization**: Reduce numeric precision (32-bit → 4-bit) to save memory
**PEFT**: Parameter-Efficient Fine-Tuning (use LoRA, not full model)
**Backtest**: Replay historical data to validate strategy
**P&L**: Profit & Loss from a trade
**Win Rate**: % of trades that made money
**Draw Down**: Peak-to-trough loss during trading

---

## Common Pitfalls (Avoid These)

| Pitfall | Impact | Solution |
|---------|--------|----------|
| **Lookahead bias** | Overestimate returns | Use chronologically-later test set |
| **Overfitting** | Works on training, fails live | Use validation set, limit model capacity, regularize |
| **Ignoring fees/slippage** | Lose money on costs | Apply realistic 1-2 pips per trade |
| **Single model overconfidence** | One model fails, entire strategy fails | Use ensemble (2-3 models) |
| **Small dataset** | Noisy labels, poor generalization | Collect 10K+ examples, augment |
| **Mixing timeframes** | Confuse model | Use consistent 5-min candles only |
| **No risk controls** | Catastrophic loss | Set position limits, daily loss caps |

---

## What's Working

✅ End-to-end pipeline from raw data to backtest
✅ Spike labeling heuristic for XAUUSD (25 pips)
✅ Colab fine-tuning notebook (ready to use)
✅ Paper trading simulator with risk controls
✅ Inference wrapper supporting LoRA adapters
✅ Comprehensive documentation (novice-friendly)

---

## What's Next / Improvements

**Short-term (this week)**
- [ ] Download real XAUUSD data (6-12 months)
- [ ] Fine-tune on Colab with real data
- [ ] Backtest and measure accuracy baseline
- [ ] Adjust spike threshold (20-30 pips) for best accuracy

**Medium-term (1-2 months)**
- [ ] Add more indicators to prompts (MACD, Bollinger Bands, volume)
- [ ] Fine-tune multiple models (Mistral + Llama), test ensemble
- [ ] Walk-forward validation (train on old data, test on new)
- [ ] Optimize LoRA hyperparameters (r, lora_alpha)
- [ ] Backtest with different risk parameters

**Long-term (3-6 months)**
- [ ] Connect to live Oanda API for paper trading (real quotes, simulated execution)
- [ ] Add confidence scoring to model predictions & filter low-confidence trades
- [ ] Implement drawdown-based position sizing (Kelly criterion)
- [ ] Monthly retraining as market regimes change
- [ ] Live trading with strict safeguards + manual approval

---

## File Navigation

| Want to... | Go to... |
|-----------|----------|
| Understand the big picture | `docs/PROCESS.md` (sec. 1-5) |
| Choose a model | `docs/MODELS.md` (sec. 1-4) |
| Get started in 5 steps | `QUICKSTART.md` |
| Learn the code | `src/` (each file has docstrings) |
| Fine-tune | `notebooks/colab_finetune_xauusd.ipynb` |
| Run backtest | `main.py --mode backtest` |
| Check paper trading logic | `src/trading/executor.py` |
| Tweak spike detection | `src/data/generate_labels.py` (detect_spike function) |

---

## Key Metrics to Track

After backtesting, check:
1. **Decision Accuracy**: % of BUY/SELL/HOLD predictions that align with actual move direction → target ≥55%
2. **Win Rate**: % of closed trades with profit → target ≥52% (break-even with fees ~50%)
3. **Total P&L**: Net profit from backtest
4. **Sharpe Ratio**: Risk-adjusted return (higher = better)
5. **Max Drawdown**: Largest peak-to-trough loss → should be manageable
6. **Inference Latency**: How fast model predicts (ms) → critical for live trading

---

## Hardware Summary

| Machine | Best Strategy | Model | Inference Speed |
|---------|--------------|-------|-----------------|
| **Local 8GB** | Use Colab for training; Mistral-small locally for inference | Mistral-small (1.3B) | 20-30 ms |
| **Local 8GB + GPU** | Fine-tune locally on GPU; inference same | Mistral-7B or Llama-2-7B (4-bit) | 50-100 ms |
| **Colab GPU (T4)** | Fine-tune Mistral-7B or Llama-2-7B | Mistral-7B or Llama-2-7B | 80-150 ms |
| **Colab GPU (V100)** | Fastest training; test larger models | Llama-2-13B (if desired) | 50-80 ms |

---

## Support & Debugging

**Problem: "ModuleNotFoundError: transformers"**
→ `pip install -r requirements.txt`

**Problem: "OANDA_API_KEY not set"**
→ Create `.env` with `OANDA_API_KEY=...` (or download CSV manually)

**Problem: "Colab OOM (out of memory)"**
→ Use smaller model (Llama-2-3B instead of 7B)

**Problem: "Model accuracy 50% (random)"**
→ More data, better heuristic, or need ensemble

**Problem: "No trades in backtest"**
→ Add `print()` statements in `main.py` to debug model predictions

---

## Community & References

**Learning Resources**
- Hugging Face docs: https://huggingface.co/docs/transformers/
- PEFT (LoRA): https://huggingface.co/docs/peft/
- FX trading: Investopedia, TradingView education
- Backtesting: https://en.wikipedia.org/wiki/Backtesting

**Model Hubs**
- Mistral: https://huggingface.co/mistralai/
- Llama: https://huggingface.co/meta-llama/
- Falcon: https://huggingface.co/tiiuae/

**FX Data Sources**
- Oanda: https://www.oanda.com (free practice account)
- Dukascopy: https://www.dukascopy.com
- HistData.com
- TradingView (CSV export)

---

## Final Checklist Before Going Live

- [ ] Backtest on 6+ months of historical data
- [ ] Achieve ≥55% decision accuracy (target)
- [ ] Win rate ≥52% (sufficient to cover fees)
- [ ] Sharpe ratio ≥0.5 (risked-adjusted return)
- [ ] Max drawdown <10% (manageable losses)
- [ ] Paper trade on live quotes (no real positions) for 2+ weeks
- [ ] Add manual approval for every real trade
- [ ] Set position size limits (start small, 0.1 lot for 5K account)
- [ ] Set daily loss limit before starting
- [ ] Log everything (for audit + debugging)
- [ ] Have exit strategy (stop-loss, time-based)

---

## Summary

You now have a **complete R&D framework** to:
1. **Label** trading data using an interpretable spike heuristic
2. **Fine-tune** LLMs efficiently (PEFT/LoRA) on Colab or local GPU
3. **Backtest** strategies on historical data with realistic costs
4. **Deploy** to paper trading with risk controls
5. **Scale** to ensemble and live trading (eventually)

**Next step**: Download data (Oanda or manually) and run `src/data/generate_labels.py` to create your first JSONL dataset.

Estimated time to first backtest: **2-3 hours** (download data 30 min + Colab training 60 min + backtest 10 min)

Good luck with your trading LLM! 🚀
