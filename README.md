# PortolioTracker

A starter Python web app for a personal portfolio tracker using FastAPI + yfinance.

## Features (MVP)

- Add/update stock positions (ticker, shares, average cost)
- Live quote pull with yfinance
- Portfolio dashboard with total value and P&L
- API endpoint for positions
- API endpoint for last four quarters (revenue + EPS when available)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000`

## Next step roadmap

- Persistent database (PostgreSQL)
- News ingestion and sentiment/impact engine
- Quarterly report good/mid/bad analyzer
- Stock scoring and screener page
- Portfolio-level risk and behavior insights
