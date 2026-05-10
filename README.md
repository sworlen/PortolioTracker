# PortfolioTracker - Wave 1

Wave 1 delivers production-foundation upgrades:

- Environment-driven config (`DATABASE_URL`, token lifetimes, secret)
- JWT access + refresh token flow
- Refresh token storage + revoke endpoint
- RBAC guard helper and admin audit endpoint
- Audit log model for sensitive actions
- Quote caching + rate limiting
- Additional normalized data models (price/news snapshots, earnings events, NAV ledger)
- Scheduler worker stub for periodic price snapshots
- CI quality gates scaffold (ruff, mypy, bandit, pytest)
- Security unit tests + analytics unit tests
- Migration directory scaffold (`migrations/0001_initial.sql`)

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Notes
Install dependencies before running tests.
