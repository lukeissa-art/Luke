# Alpaca Stock Trader API

This project gives you a trading bot API that connects directly to Alpaca.

Important: no bot can guarantee profit. This one is built to be disciplined and risk-limited, and it is configured for paper trading first.

## What it does

- Pulls recent bars from Alpaca market data
- Uses EMA + RSI momentum signals (`BUY`, `SELL`, `HOLD`)
- Applies risk controls:
  - max position size
  - max open positions
  - cooldown between trades
  - optional bracket exits (take-profit + stop-loss)
- Exposes API routes so you can trigger trades from Alpaca-connected workflows

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy env template and add your Alpaca paper keys:

```bash
cp .env.example .env
```

4. Keep `DRY_RUN=true` until you verify behavior.

## Run API server

```bash
uvicorn main:app --reload --port 8000
```

## API endpoints

- `GET /health`: service check
- `GET /status`: account + position status
- `GET /signal/{symbol}`: signal for one symbol
- `POST /run-once`: run one trading cycle

Example:

```bash
curl -X POST http://127.0.0.1:8000/run-once \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL","MSFT"],"force":false}'
```

## Run as a loop

```bash
python run_bot.py
```

It will execute every `POLL_INTERVAL_SECONDS`.

## Suggested first run

1. Use Alpaca paper account only.
2. Start with `DRY_RUN=true`.
3. Call `/signal/{symbol}` for your symbols.
4. Call `/run-once` and inspect generated orders in response.
5. Switch to `DRY_RUN=false` only after validating behavior.

## Backtest and parameter search

```bash
python backtest.py                  # run with current settings
python backtest.py --optimize       # grid-search params; needs Alpaca data keys
```
Targets in the optimizer: average return ≥0.5% and win rate ≥55%. Update `.env` with the suggested parameters after it finds a set that meets the targets.
Current default tuned set (from recent run): 5Min bars, EMA 5/13, RSI 7, stop 0.75%, take profit 1.2%, cooldown 15 minutes, 2% position size.

## Deploy to Railway (always-on)

1. Set these Railway variables: `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_BASE_URL`, `ALPACA_DATA_URL`, plus any tunables (`SYMBOLS`, `POLL_INTERVAL_SECONDS`, `FAST_EMA`, `SLOW_EMA`, `RSI_PERIOD`, `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, `COOLDOWN_MINUTES`, `DRY_RUN`).
2. Deploy with the provided `Dockerfile`.
3. Create a Worker service with command: `python run_bot.py`.
4. (Optional) Create a Web service with command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. Tail Railway logs to confirm cycles are running and orders are submitting.
