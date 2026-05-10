# PortolioTracker

A growing Python web app for a personal portfolio tracker using FastAPI + yfinance.

## Current capabilities

- Add/update stock positions (ticker, shares, average cost)
- Persistent position storage with SQLite + SQLAlchemy
- Live quote pull with yfinance
- Portfolio dashboard with total value and P&L
- Portfolio summary API
- Multi-factor stock scoring (quality, growth, valuation, momentum, risk)
- Quarterly analyzer with `good/mid/bad` verdict and explainable component scores
- Ticker news endpoint for downstream impact-analysis work
- Basic screener endpoint with score threshold filtering

## API Endpoints

- `GET /api/positions`
- `GET /api/portfolio/summary`
- `GET /api/stocks/{ticker}/quarterly`
- `GET /api/stocks/{ticker}/score`
- `GET /api/stocks/{ticker}/quarterly-analyzer`
- `GET /api/stocks/{ticker}/news`
- `GET /api/screener?tickers=AAPL,MSFT,NVDA&min_score=65`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000`

## Next high-complexity milestones

- Background schedulers for quote/news refresh
- Portfolio-level risk decomposition and stress tests
- Natural-language portfolio copilot
- Custom factor backtesting and strategy lab
