from __future__ import annotations

from datetime import datetime
from statistics import mean
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


class QuarterlyAnalyzerResult(BaseModel):
    ticker: str
    verdict: str
    total_score: float
    component_scores: Dict[str, float]
    positives: List[str]
    negatives: List[str]
    summary: str


class StockScore(BaseModel):
    ticker: str
    total: float
    quality: float
    growth: float
    valuation: float
    momentum: float
    risk: float


positions_store: Dict[str, PositionCreate] = {}


# ---------- Market + Portfolio helpers ----------
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


# ---------- Analytics helpers ----------
def _normalize_rank(value: float, series: List[float], reverse: bool = False) -> float:
    if not series:
        return 50.0
    lo, hi = min(series), max(series)
    if hi == lo:
        return 50.0
    score = ((value - lo) / (hi - lo)) * 100
    score = 100 - score if reverse else score
    return max(0.0, min(100.0, score))


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _quarterly_growth(values: List[float]) -> float:
    if len(values) < 2 or values[-1] == 0:
        return 0.0
    return ((values[0] - values[-1]) / abs(values[-1])) * 100


def _compute_stock_score(ticker: str) -> StockScore:
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    hist = tk.history(period="1y")
    fin = tk.quarterly_income_stmt

    pe = _safe_float(info.get("trailingPE"))
    roe = _safe_float(info.get("returnOnEquity"))
    debt_to_equity = _safe_float(info.get("debtToEquity"))
    beta = _safe_float(info.get("beta"))

    quality = 50.0
    if roe is not None:
        quality = max(0.0, min(100.0, roe * 100))

    growth = 50.0
    if fin is not None and not fin.empty and "Total Revenue" in fin.index:
        revs = [_safe_float(v) for v in list(fin.loc["Total Revenue"].values)[:4]]
        revs = [v for v in revs if v is not None]
        if len(revs) >= 2:
            growth = max(0.0, min(100.0, 50 + (_quarterly_growth(revs) * 0.5)))

    valuation = 50.0
    if pe is not None:
        valuation = max(0.0, min(100.0, 100 - min(pe, 80) * 1.25))

    momentum = 50.0
    if not hist.empty and len(hist) > 50:
        close = hist["Close"]
        ret_6m = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100 if len(close) > 126 else 0
        momentum = max(0.0, min(100.0, 50 + ret_6m))

    risk = 50.0
    penalties = 0.0
    if debt_to_equity is not None:
        penalties += min(30, debt_to_equity / 10)
    if beta is not None:
        penalties += max(0, (beta - 1) * 15)
    risk = max(0.0, min(100.0, 80 - penalties))

    total = quality * 0.25 + growth * 0.20 + valuation * 0.20 + momentum * 0.15 + risk * 0.20
    return StockScore(
        ticker=ticker.upper(),
        total=round(total, 2),
        quality=round(quality, 2),
        growth=round(growth, 2),
        valuation=round(valuation, 2),
        momentum=round(momentum, 2),
        risk=round(risk, 2),
    )


def _quarterly_verdict(ticker: str) -> QuarterlyAnalyzerResult:
    tk = yf.Ticker(ticker)
    fin = tk.quarterly_income_stmt
    cf = tk.quarterly_cashflow
    balance = tk.quarterly_balance_sheet

    if fin is None or fin.empty:
        raise HTTPException(status_code=404, detail=f"No quarterly data for {ticker}")

    component_scores = {
        "growth": 50.0,
        "profitability": 50.0,
        "cashflow": 50.0,
        "balance_sheet": 50.0,
        "stability": 50.0,
    }
    positives: List[str] = []
    negatives: List[str] = []

    if "Total Revenue" in fin.index:
        revs = [_safe_float(v) for v in list(fin.loc["Total Revenue"].values)[:4]]
        revs = [v for v in revs if v is not None]
        if len(revs) >= 2:
            g = _quarterly_growth(revs)
            component_scores["growth"] = max(0, min(100, 50 + g * 0.5))
            (positives if g > 0 else negatives).append(f"Revenue trend {'up' if g > 0 else 'down'} ({g:.1f}%).")

    if "Net Income" in fin.index and "Total Revenue" in fin.index:
        net = [_safe_float(v) for v in list(fin.loc["Net Income"].values)[:4]]
        revs = [_safe_float(v) for v in list(fin.loc["Total Revenue"].values)[:4]]
        margins = [(n / r) * 100 for n, r in zip(net, revs) if n is not None and r not in (None, 0)]
        if margins:
            component_scores["profitability"] = max(0, min(100, 50 + (margins[0] - mean(margins[1:])) * 3 if len(margins) > 1 else 50))
            if len(margins) > 1 and margins[0] > mean(margins[1:]):
                positives.append("Net margin improved versus recent quarters.")
            else:
                negatives.append("Net margin did not improve versus recent quarters.")

    if cf is not None and not cf.empty and "Operating Cash Flow" in cf.index:
        cfo = [_safe_float(v) for v in list(cf.loc["Operating Cash Flow"].values)[:4]]
        cfo = [v for v in cfo if v is not None]
        if len(cfo) >= 2:
            g = _quarterly_growth(cfo)
            component_scores["cashflow"] = max(0, min(100, 50 + g * 0.5))
            (positives if g > 0 else negatives).append(f"Operating cash flow {'up' if g > 0 else 'down'} ({g:.1f}%).")

    if balance is not None and not balance.empty and "Total Debt" in balance.index and "Cash Cash Equivalents And Short Term Investments" in balance.index:
        debt = _safe_float(list(balance.loc["Total Debt"].values)[0])
        cash = _safe_float(list(balance.loc["Cash Cash Equivalents And Short Term Investments"].values)[0])
        if debt is not None and cash is not None and debt > 0:
            ratio = cash / debt
            component_scores["balance_sheet"] = max(0, min(100, ratio * 100))
            if ratio >= 0.75:
                positives.append("Balance sheet liquidity is healthy versus debt.")
            else:
                negatives.append("Debt level is high versus cash cushion.")

    std = 15.0
    component_scores["stability"] = max(0, min(100, 100 - std))

    total = (
        component_scores["growth"] * 0.25
        + component_scores["profitability"] * 0.20
        + component_scores["cashflow"] * 0.20
        + component_scores["balance_sheet"] * 0.15
        + component_scores["stability"] * 0.20
    )

    verdict = "mid"
    if total >= 82:
        verdict = "good"
    elif total < 60:
        verdict = "bad"

    summary = f"{ticker.upper()} quarterly report looks {verdict.upper()} with score {total:.1f}/100."

    return QuarterlyAnalyzerResult(
        ticker=ticker.upper(),
        verdict=verdict,
        total_score=round(total, 2),
        component_scores={k: round(v, 2) for k, v in component_scores.items()},
        positives=positives[:5],
        negatives=negatives[:5],
        summary=summary,
    )


# ---------- Web ----------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    rows = []
    total_value = 0.0
    total_pnl = 0.0
    scored_positions = []

    for p in positions_store.values():
        try:
            row = _position_view(p)
            score = _compute_stock_score(p.ticker)
        except HTTPException:
            continue
        rows.append(row)
        scored_positions.append(score)
        total_value += row.market_value
        total_pnl += row.pnl

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "positions": rows,
            "scores": scored_positions,
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


@app.get("/api/stocks/{ticker}/score", response_model=StockScore)
def score_stock(ticker: str):
    return _compute_stock_score(ticker.upper())


@app.get("/api/stocks/{ticker}/quarterly-analyzer", response_model=QuarterlyAnalyzerResult)
def quarterly_analyzer(ticker: str):
    return _quarterly_verdict(ticker.upper())


@app.get("/api/stocks/{ticker}/news")
def stock_news(ticker: str):
    tk = yf.Ticker(ticker.upper())
    items = tk.news or []
    parsed = []
    for item in items[:10]:
        content = item.get("content", {})
        parsed.append(
            {
                "title": content.get("title"),
                "summary": content.get("summary"),
                "url": content.get("canonicalUrl", {}).get("url"),
                "publisher": content.get("provider", {}).get("displayName"),
                "published_at": content.get("pubDate"),
            }
        )
    return {"ticker": ticker.upper(), "count": len(parsed), "news": parsed}


@app.get("/health")
def health():
    return {"ok": True}
