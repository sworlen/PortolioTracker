# PortfolioTracker - Wave 2

Wave 2 focuses on quarterly and news intelligence.

## Added in Wave 2

- Quarterly deep dive endpoint: `/api/stocks/{ticker}/quarterly/deep-dive`
  - QoQ + YoY decomposition
  - Margin bridge (gross/op/net)
  - Cash earnings quality check
  - Accounting red-flag rules
  - Sector-specific KPI packs
  - Consensus integration stub for estimate surprise engine

- News intelligence endpoint: `/api/stocks/{ticker}/news/intelligence`
  - Headline deduplication
  - Event taxonomy classification (guidance/legal/M&A/analyst/earnings/macro)
  - Sentiment scoring + confidence
  - Source ranking
  - Portfolio impact hint by position weight
  - News analysis persistence in DB

- New intelligence module: `app/intelligence.py`

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
