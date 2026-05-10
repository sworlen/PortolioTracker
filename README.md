# PortolioTracker

FastAPI-based investor platform foundation with advanced analytics, scoring, risk, and valuation modules.

## Implemented capabilities

- User + portfolio data model
- Persistent positions + transaction logging
- Quote refresh job endpoint
- Multi-factor stock scoring
- Quarterly analyzer (good/mid/bad)
- News impact estimator
- Screener + saved screens
- Portfolio concentration risk endpoint
- Simple backtest endpoint
- **DCF valuation endpoint**
- **Risk metrics endpoint** (Sharpe, Sortino, vol, drawdown)
- **Monte Carlo simulation endpoint**

## Key API examples

- `GET /api/stocks/AAPL/valuation/dcf`
- `GET /api/stocks/AAPL/risk-metrics`
- `GET /api/stocks/AAPL/monte-carlo?days=252&sims=500`
- `GET /api/backtest/simple?tickers=AAPL,MSFT&start=2024-01-01&end=2025-01-01`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for API docs.
