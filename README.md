# PortolioTracker

Advanced FastAPI investor platform with architecture upgrades toward production.

## Added in this iteration

- Password hashing + JWT auth utilities (`app/security.py`)
- User registration/login token flow with bearer-token dependency checks
- Environment-based DB configuration (`DATABASE_URL`)
- Lightweight in-process caching + rate limiting for quote pulls
- Operational metrics endpoint (`/api/ops/metrics`)
- Snapshot job endpoint for historical price storage (`/api/jobs/snapshot-prices`)
- Additional normalized persistence tables for:
  - price snapshots
  - news snapshots
  - earnings/catalyst events
  - portfolio NAV ledger
- Investor workflow features:
  - thesis tracking
  - watchlist funnel
  - score alerts
  - daily brief
  - quarterly translator
- Quant algorithms module: returns, drawdown, Sharpe/Sortino, DCF, Monte Carlo
- Basic automated tests + GitHub Actions CI scaffold

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs at `http://127.0.0.1:8000/docs`.
