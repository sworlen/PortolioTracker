from datetime import datetime
from typing import Dict, List

import yfinance as yf
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

app = FastAPI(title="Portfolio Tracker")
templates = Jinja2Templates(directory="app/templates")


class PositionCreate(BaseModel):
    ticker: str = Field(min_length=1)
    shares: float = Field(gt=0)
    avg_cost: float = Field(gt=0)


class Position(BaseModel):
    ticker: str
    shares: float
    avg_cost: float
    last_price: float
    market_value: float
    pnl: float


class QuarterInsight(BaseModel):
    quarter: str
    revenue: float | None
    eps: float | None


positions_store: Dict[str, PositionCreate] = {}


def _quote(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="1d")
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No price found for {ticker}")
    return float(hist["Close"].iloc[-1])


def _position_view(p: PositionCreate) -> Position:
    last = _quote(p.ticker)
    mv = last * p.shares
    pnl = (last - p.avg_cost) * p.shares
    return Position(
        ticker=p.ticker,
        shares=p.shares,
        avg_cost=p.avg_cost,
        last_price=last,
        market_value=mv,
        pnl=pnl,
    )


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    rows = []
    total_value = 0.0
    total_pnl = 0.0
    for p in positions_store.values():
        try:
            row = _position_view(p)
        except HTTPException:
            continue
        rows.append(row)
        total_value += row.market_value
        total_pnl += row.pnl

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "positions": rows,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        },
    )


@app.post("/positions")
def add_position(ticker: str = Form(...), shares: float = Form(...), avg_cost: float = Form(...)):
    payload = PositionCreate(ticker=ticker.upper(), shares=shares, avg_cost=avg_cost)
    positions_store[payload.ticker] = payload
    return {"ok": True, "ticker": payload.ticker}


@app.get("/api/positions", response_model=List[Position])
def get_positions():
    return [_position_view(p) for p in positions_store.values()]


@app.get("/api/stocks/{ticker}/quarterly", response_model=List[QuarterInsight])
def quarterly(ticker: str):
    tk = yf.Ticker(ticker.upper())
    df = tk.quarterly_income_stmt
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No quarterly data for {ticker}")

    quarters: List[QuarterInsight] = []
    for col in list(df.columns)[:4]:
        revenue = None
        eps = None
        if "Total Revenue" in df.index:
            revenue = float(df.loc["Total Revenue", col])
        if "Basic EPS" in df.index:
            eps = float(df.loc["Basic EPS", col])

        quarters.append(
            QuarterInsight(
                quarter=col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col),
                revenue=revenue,
                eps=eps,
            )
        )

    return quarters


@app.get("/health")
def health():
    return {"ok": True}
