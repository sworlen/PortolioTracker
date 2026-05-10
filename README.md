# PortolioTracker

Advanced FastAPI investor platform with portfolio management, scoring, risk, valuation, alerts, thesis tracking, and research workflow tools.

## Added advanced modules

- Thesis tracking (`/api/thesis`)
- Quarterly translator (`/api/stocks/{ticker}/quarterly-translator`)
- Watchlist funnel stages (`/api/watchlist`)
- Score-based alert rules (`/api/alerts`, `/api/alerts/check`)
- Portfolio daily brief (`/api/portfolio/daily-brief`)
- DCF valuation, risk metrics, Monte Carlo, backtest, screener, risk concentration, and news impact endpoints

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for API docs.
