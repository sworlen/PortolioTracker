"""Wave 1 scheduler stub for periodic ingestion."""
from datetime import datetime

from app.main import SessionLocal, PriceSnapshotRow, _quote


def run_price_snapshot_job(tickers: list[str]):
    db = SessionLocal()
    try:
        for t in tickers:
            db.add(PriceSnapshotRow(ticker=t.upper(), close=_quote(t.upper())))
        db.commit()
        return {"ok": True, "ts": datetime.utcnow().isoformat(), "tickers": tickers}
    finally:
        db.close()
