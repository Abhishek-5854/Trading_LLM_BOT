Overview

This document explains goals, constraints, data strategy, model recommendations, spike-labeling heuristics (your 25‑point example), implementation steps, common past failures to avoid, required theory & terminology, and libraries/functions to use. It's written for a novice developer building a Forex paper-trading pipeline that trains a local/Colab LLM to predict trade actions from short-run price behavior.

1) What you're trying to achieve (concise)
- Target: Forex (FX) paper trading on live market data.
- Research goal: Build an LLM-assisted decision system that identifies short-term spikes (e.g., 25-point moves over a short window) and suggests actions (buy/sell/hold).
- Minimum baseline: ~55% decision accuracy in early iterations.
- Execution mode: Paper trading only initially; live trading later with strict safety controls.

2) Definitions & assumptions (clarify before implementation)
- "25 points": interpret as 25 pips/points in price move. Note: pip definition depends on pair (most pairs 4 decimal places, JPY pairs 2 decimals). Confirm which currency pairs (e.g., EUR/USD, USD/JPY).
- "Shot up from 3025 to 3050 without red candle": means consecutive same-direction candles over N bars producing >=25-point net move with no intervening bearish candle.
- Timeframe: you said "last 5 minutes" — assume 1-minute or 5-minute candles for labelling. Specify exact candle timeframe when generating dataset.

3) Success criteria & metrics
- Classification accuracy (buy/sell/hold) >= 55% on holdout test.
- Backtest metrics: cumulative return, Sharpe ratio, max drawdown, win-rate, average P&L per trade.
- Execution safety: zero real trades until manual approval; per-trade risk cap set in paper trader.

4) Constraints
- Local dev machine: 8 GB RAM (limited for large models). Plan for lighter local inference or remote GPU (Colab) for training/inference.
- Data: no labeled dataset yet — will generate labels using spike heuristics from historical FX data.

5) Data sources & ingestion
- Recommended sources: Oanda (historical), TrueFX, Dukascopy, HistData, FXCM, and `ccxt` for exchange tick data for crypto; Yahoo Finance / `yfinance` for majors (limited). Use whichever provides required minute-level history.
- File storage: `data/raw/` (downloaded CSV/zip), `data/processed/` (cleaned, resampled), `data/datasets/` (JSONL for fine-tuning).

6) Spike-labeling heuristic (for the "25-point" rule)
- Parameters to choose: timeframe (e.g., 1m candles), lookback_k (e.g., 5 bars), spike_threshold (25 points), allowed_gap (no bearish candles in the window), min_volume (optional).
- Algorithm (example):
  1. Resample price to chosen candle timeframe.
  2. For each new candle t, compute delta = close(t) - close(t - lookback_k).
  3. If delta >= spike_threshold and all intermediate candles are bullish (close > open) — label event "spike_up".
  4. Mirror for negative spikes: delta <= -spike_threshold and all intermediate candles bearish — label "spike_down".
  5. Surrounding context: collect preceding N candles and optional news/text features as the prompt context.
  6. Add label action: buy for spike_up, sell for spike_down, hold otherwise (but sample hold examples to balance dataset).
- Convert to JSONL: {"prompt": "[context candles + indicators] -> what is action?", "response": "Buy"}

7) Dataset balancing & augmentation
- Spikes are rare: oversample spike examples, undersample hold, and augment with synthetic noise, small shifts in price, or jittered timestamps to reduce overfitting.
- Avoid label leakage: ensure test fold is chronologically after training fold (no lookahead).

8) Model inventory & suitability (multiple options)
- Local CPU (8GB RAM): run only very small models or use remote inference. Options:
  - Tiny quantized models (ggml / GGUF): small community models ~<1.5B parameters (if available) via `llama.cpp` or `ggml` backends.
  - Use HF Inference API or remote Colab for heavier models.
- Colab GPU (recommended for fine-tuning/inference): choose by VRAM:
  - 12–16 GB GPU: Llama-2-7B or Mistral-7B with 4-bit quantization (bitsandbytes + accelerate + PEFT).
  - 8 GB GPU: Llama-2-3B or Mistral-small (1–3B) with QLoRA for training.
- Suggested open-source models (starting point):
  - Mistral-small (≈1.3B) — lightweight, good tradeoff for Colab CPU/low VRAM.
  - Llama-2-3B — smaller Llama family, good for Colab with limited VRAM.
  - Llama-2-7B / Mistral-7B — stronger but requires >=12GB VRAM or 4-bit quantization.
  - Falcon-7B-Instruct — similar to 7B models, works with bitsandbytes/PEFT.
  - Quantized community GGUF models (gpt4all-style) — usable locally via `llama.cpp`.
- Strategy: use smaller model(s) for quick iteration and an ensemble of models for robustness; do not lock to a single model.

9) Fine-tuning approach
- Use PEFT (LoRA) or QLoRA to fine-tune only adapter weights instead of full model.
- Workflow:
  1. Prepare JSONL (prompt/response) dataset.
  2. Tokenize with the model tokenizer.
  3. Use `transformers.Trainer` or `trl` with `peft` integration to train LoRA adapters.
  4. Save LoRA weights to `models/checkpoints/`.
- Colab notebook: use accelerated training (`accelerate`) and bitsandbytes for 4-bit.

10) Inference & paper execution
- Wrap model inference behind `LLMWrapper` and an adapter that converts model text output -> structured action (buy/sell/hold + confidence).
- `PaperExecutor` applies risk rules and records trades. Add simulated slippage and fees in backtest.

11) Backtesting & evaluation
- Use time-series-aware train/test split.
- Simulate orders with slippage, spreads, and execution delay.
- Compute classification metrics and financial metrics.
- Perform walk-forward validation.

12) Common failures & pitfalls (learned from others)
- Lookahead/data leakage: mixing future info into training.
- Overfitting to small, noisy FX moves; models memorize patterns causing poor generalization.
- Ignoring transaction costs & slippage — strategies that look profitable on raw returns often fail after fees.
- Survivorship bias & dataset selection bias.
- Labeling bias: naive heuristics produce noisy labels that harm training.
- Wrong granularity: mixing timeframes (1m vs 5m) can confuse models.
- Excessive reliance on a single model — ensemble and model-agnostic features help.

13) Theory & learning topics (recommended order)
- Time-series basics: stationarity, autocorrelation, resampling.
- Candlestick basics & forex pip conventions.
- Backtesting fundamentals: walk-forward testing, slippage, transaction costs.
- Machine learning foundations: classification metrics, cross-validation, overfitting.
- Transformer fundamentals: tokens, attention, prompt/response format.
- Fine-tuning methods: LoRA, QLoRA, quantization (bitsandbytes), PEFT.

14) Key terminologies
- Candle/bar: OHLC price for a timeframe.
- Pip/point: smallest price increment (varies by pair).
- Spike: abrupt price move above a threshold in a short period.
- PEFT / LoRA / QLoRA: parameter-efficient fine-tuning methods.
- Quantization: reducing model numeric precision for smaller memory footprint.

15) Libraries & useful functions
- Data & numeric: `pandas` (resample, rolling), `numpy`.
- Market data: `yfinance`, data vendor SDKs (Oanda, Dukascopy), `ccxt` for exchange ticks.
- Modeling: `transformers` (AutoTokenizer, AutoModelForCausalLM, pipeline), `peft`, `bitsandbytes`, `accelerate`, `datasets`.
- Local quantized inference: `llama.cpp` / `ggml` / GGUF converters.
- Backtesting & plotting: `vectorbt`, custom `pandas` backtester, `matplotlib` / `seaborn`.
- Env: `python-dotenv` for keys.

16) File structure & where to implement
- `data/` (scripts to download & preprocess)
  - `src/data/ingest.py` (starter exists)
  - add `src/data/generate_labels.py` (create JSONL)
- `models/`
  - `src/models/llm_wrapper.py` (starter exists)
  - `src/finetune/train.py` (starter exists) — add PEFT pipeline
- `trading/`
  - `src/trading/executor.py` (starter exists)
- `notebooks/` — Colab-ready fine-tune notebook
- `docs/PROCESS.md` — (this file)

17) Step-by-step plan to implement (practical)
1. Confirm FX pairs and confirm "25 points" meaning (pips) and candle timeframe.
2. Choose data source and download 6–12 months of minute data for chosen pairs.
3. Implement `generate_labels.py` with the spike heuristic and export JSONL.
4. Pick a small model for Colab (e.g., Mistral-small or Llama-2-3B) and build a Colab notebook to fine-tune using PEFT.
5. Train adapters on the generated dataset; evaluate on holdout chronologically later in time.
6. Implement inference wrapper that outputs structured actions and confidence.
7. Connect inference to `PaperExecutor` and run paper trading on live stream (or replay historical stream) with logging.
8. Backtest results and iterate on labeling, features, and model choice.

18) Running commands (environment)
- Create venv and install:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Example: run ingestion (after installing deps):

```bash
python src/data/ingest.py
```

- Example: run paper executor demo:

```bash
python src/trading/executor.py
```

19) Next actions I can take now (pick one)
- A: Implement `src/data/generate_labels.py` that creates JSONL from minute-level FX CSVs using the 25-point spike rule.
- B: Create a Colab-ready notebook (fine-tune via PEFT / QLoRA) using a small model template.
- C: Build an end-to-end demo that downloads short FX history for a pair, labels spikes, trains a tiny model locally (toy) and runs a paper-trade replay.

20) Questions for you (required to proceed)
- Confirm which currency pairs to target (e.g., EUR/USD, USD/JPY, GBP/USD). If you want multiple, list them.
- Confirm "25 points" is pips (yes/no) and confirm candle timeframe (1m / 5m).
- Do you prefer I implement option A, B or C next?

Appendix: Quick heuristics and tips
- Always keep an out-of-sample chronological test set.
- Start with paper trading via replay before connecting to live data streams.
- Log everything (inputs to model, model output, trade executions) for debugging and audit.

End of PROCESS.md
