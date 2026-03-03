Quick Start Guide: XAUUSD Trading LLM (Step-by-Step)

This guide walks you through:
1. Environment setup
2. Getting data
3. Generating labeled examples
4. Fine-tuning on Colab
5. Running backtest

Prerequisites
- Python 3.10+
- Git
- GPU access (optional but recommended for training)

Step 1: Clone & Setup Environment

```bash
git clone <your-repo>
cd Trading_LLM_BOT

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Or on macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Step 2: Setup API Keys (Optional for live data download)

Create a `.env` file in the project root:

```
OANDA_API_KEY=your_oanda_api_key_here
```

Alternatively, download historical XAUUSD/XAGUSD data manually from free sources like:
- Dukascopy (www.dukascopy.com/swiss/english/marketwatch)
- HistData.com
- TradingView CSV export

Place downloaded CSVs in `data/raw/` as:
- `data/raw/xauusd_ohlcv.csv`
- `data/raw/xagusd_ohlcv.csv`

Expected CSV format:
```
time,open,high,low,close,volume
2024-01-01 00:00:00,2050.00,2051.00,2049.50,2050.50,1000
...
```

Step 3: Generate Training Data

If you have downloaded raw OHLCV data:

```bash
python src/data/generate_labels.py
```

This will:
- Read `data/raw/xauusd_ohlcv.csv`
- Apply the 25-pip spike heuristic
- Create JSONL files:
  - `data/datasets/train_xauusd.jsonl`
  - `data/datasets/test_xauusd.jsonl`

Check output:
```bash
cat data/datasets/train_xauusd.jsonl | head -1
```

Example line:
```json
{"prompt": "Recent XAUUSD price action:\n  Bar 1: O=2100.0, H=2105.0, L=2098.0, C=2103.0\n...", "response": "BUY", "timestamp": "2024-01-01T00:05:00", "price": 2050.5}
```

Step 4: Fine-Tune Model on Colab (Recommended)

1. Open Google Colab: https://colab.research.google.com
2. Create a new Python 3 notebook
3. Copy content from `notebooks/colab_finetune_xauusd.ipynb` into the cells
4. Follow the notebook cells in order:
   - Install dependencies
   - Upload `data/datasets/train_xauusd.jsonl` and test JSONL
   - Load base model (Mistral-7B or Llama-2-3B)
   - Run training loop
   - Download trained adapter weights

Expected training time: 30 min - 2 hours (depending on dataset size and GPU).

Expected output:
- `xauusd_lora_adapters.zip` → download and extract to `models/checkpoints/xauusd_lora_final/`

Step 5: Run Local Backtest (Demo)

After downloading adapters from Colab:

```bash
# Extract adapters
unzip xauusd_lora_adapters.zip -d models/checkpoints/

# Run backtest (requires data in data/raw/)
python main.py --mode backtest \
  --csv_path data/raw/xauusd_ohlcv.csv \
  --model_path models/checkpoints/xauusd_lora_final \
  --base_model "mistralai/Mistral-7B-Instruct-v0.1"
```

Expected output:
```
=== Starting Backtest for XAUUSD ===
Loaded 5000 candles for XAUUSD
Loading model from mistralai/Mistral-7B-Instruct-v0.1...
Model loaded successfully
Running simulation over 5000 candles...

[50/5000] Price: 2050.23, Action: HOLD, Open trade: False
[100/5000] Price: 2051.45, Action: BUY, Open trade: True
...

=== Final Statistics ===
  Total trades executed: 24
  Closed trades: 20
  Total P&L: $345.67
  Win rate: 58.3%
  Current balance: $10345.67
  Return: 3.46%
```

All trades logged to `logs/backtest_trades.jsonl`.

Step 6 (Optional): Run Inference Demo

Test a single prediction without backtest:

```bash
python main.py --mode demo \
  --model_path models/checkpoints/xauusd_lora_final \
  --base_model "mistralai/Mistral-7B-Instruct-v0.1"
```

Next Steps

1. Improve the spike heuristic (adjust `spike_threshold`, `lookback_k` in `src/data/generate_labels.py`)
2. Collect more data (longer time windows, different currency pairs)
3. Experiment with different models (Llama-2-7B, Mistral-7B)
4. Tune LoRA hyperparameters (r, lora_alpha in Colab notebook)
5. Add more technical indicators (MACD, Bollinger Bands) to prompts
6. Backtest on validation set and evaluate metrics (Sharpe ratio, max drawdown)
7. Once confident, set up live paper trading with broker API (Oanda, Interactive Brokers)

Troubleshooting

Q: "ModuleNotFoundError: No module named 'transformers'"
A: Run: pip install -r requirements.txt

Q: "OANDA_API_KEY not set"
A: Either add to .env or download data manually and save to data/raw/*.csv

Q: "CUDA out of memory" on local machine
A: Use Colab GPU for training. For inference on 8GB RAM machine, use smaller models (Llama-2-3B, Mistral-small) or quantize.

Q: "Test accuracy is 50% (random)"
A: More data or better heuristics needed. Try:
  - Longer lookback window
  - Lower spike threshold (20 instead of 25)
  - Add more indicators to the prompt

Q: "No trades executed in backtest"
A: Check if model predictions are being made. Add print statements in main.py loops.

Support & Learning

- Transformers docs: https://huggingface.co/docs/transformers
- PEFT (LoRA) docs: https://huggingface.co/docs/peft
- Backtesting best practices: https://en.wikipedia.org/wiki/Backtesting
- FX trading basics: Investopedia Forex guide

---
For more details, see `docs/PROCESS.md`
