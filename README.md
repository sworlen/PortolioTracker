# PortolioTracker

FastAPI-based investor platform foundation with seven major capability areas implemented as starter modules.

## Implemented (7 parts)

1. **Auth + Multi-portfolio model** (`/api/users`, `/api/portfolios`)
2. **Refresh job bootstrap** (`/api/jobs/refresh`)
3. **Quarterly analyzer v2** (`/api/stocks/{ticker}/quarterly-analyzer`)
4. **News impact endpoint** (`/api/stocks/{ticker}/news-impact`)
5. **Screener v2 + saved screens** (`/api/screener`, `/api/screens/save`)
6. **Portfolio risk module starter** (`/api/portfolio/risk`)
7. **Simple backtest module starter** (`/api/backtest/simple`)

## Also included

- SQLite persistence with SQLAlchemy
- Position add/update with transaction logging
- Portfolio summary endpoint
- Stock scoring endpoint

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for API docs.
