from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Dict, List

import yfinance as yf
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

app = FastAPI(title="Portfolio Tracker")
templates = Jinja2Templates(directory="app/templates")

engine = create_engine("sqlite:///portfolio.db", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


class PositionRow(Base):
    __tablename__ = "positions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    shares: Mapped[float] = mapped_column(Float)
    avg_cost: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


Base.metadata.create_all(engine)


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


class PortfolioSummary(BaseModel):
    total_value: float
    total_pnl: float
    top_position: str | None
    positions_count: int


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _quote(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="1d")
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No price found for {ticker}")
    return float(hist["Close"].iloc[-1])


def _position_view(row: PositionRow) -> Position:
    last = _quote(row.ticker)
    mv = last * row.shares
    pnl = (last - row.avg_cost) * row.shares
    return Position(
        ticker=row.ticker,
        shares=row.shares,
        avg_cost=row.avg_cost,
        last_price=last,
        market_value=mv,
        pnl=pnl,
    )


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

    quality = max(0.0, min(100.0, (roe or 0.5) * 100))

    growth = 50.0
    if fin is not None and not fin.empty and "Total Revenue" in fin.index:
        revs = [_safe_float(v) for v in list(fin.loc["Total Revenue"].values)[:4]]
        revs = [v for v in revs if v is not None]
        if len(revs) >= 2:
            growth = max(0.0, min(100.0, 50 + (_quarterly_growth(revs) * 0.5)))

    valuation = max(0.0, min(100.0, 100 - min(pe or 35, 80) * 1.25))

    momentum = 50.0
    if not hist.empty and len(hist) > 126:
        close = hist["Close"]
        ret_6m = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100
        momentum = max(0.0, min(100.0, 50 + ret_6m))

    penalties = min(30, (debt_to_equity or 100) / 10) + max(0, ((beta or 1) - 1) * 15)
    risk = max(0.0, min(100.0, 80 - penalties))

    total = quality * 0.25 + growth * 0.20 + valuation * 0.20 + momentum * 0.15 + risk * 0.20
    return StockScore(ticker=ticker.upper(), total=round(total, 2), quality=round(quality, 2), growth=round(growth, 2), valuation=round(valuation, 2), momentum=round(momentum, 2), risk=round(risk, 2))


def _quarterly_verdict(ticker: str) -> QuarterlyAnalyzerResult:
    tk = yf.Ticker(ticker)
    fin = tk.quarterly_income_stmt
    cf = tk.quarterly_cashflow
    if fin is None or fin.empty:
        raise HTTPException(status_code=404, detail=f"No quarterly data for {ticker}")

    scores = {"growth": 50.0, "profitability": 50.0, "cashflow": 50.0, "balance_sheet": 50.0, "stability": 85.0}
    positives, negatives = [], []

    if "Total Revenue" in fin.index:
        revs = [_safe_float(v) for v in list(fin.loc["Total Revenue"].values)[:4]]
        revs = [v for v in revs if v is not None]
        if len(revs) >= 2:
            g = _quarterly_growth(revs)
            scores["growth"] = max(0, min(100, 50 + g * 0.5))
            (positives if g > 0 else negatives).append(f"Revenue trend {'up' if g > 0 else 'down'} ({g:.1f}%).")

    if "Net Income" in fin.index and "Total Revenue" in fin.index:
        net = [_safe_float(v) for v in list(fin.loc["Net Income"].values)[:4]]
        revs = [_safe_float(v) for v in list(fin.loc["Total Revenue"].values)[:4]]
        margins = [(n / r) * 100 for n, r in zip(net, revs) if n is not None and r not in (None, 0)]
        if len(margins) > 1:
            delta = margins[0] - mean(margins[1:])
            scores["profitability"] = max(0, min(100, 50 + delta * 3))
            (positives if delta > 0 else negatives).append("Margin trend improved." if delta > 0 else "Margin trend weakened.")

    if cf is not None and not cf.empty and "Operating Cash Flow" in cf.index:
        cfo = [_safe_float(v) for v in list(cf.loc["Operating Cash Flow"].values)[:4]]
        cfo = [v for v in cfo if v is not None]
        if len(cfo) >= 2:
            g = _quarterly_growth(cfo)
            scores["cashflow"] = max(0, min(100, 50 + g * 0.5))
            (positives if g > 0 else negatives).append("Operating cash flow improved." if g > 0 else "Operating cash flow declined.")

    total = scores["growth"] * 0.25 + scores["profitability"] * 0.20 + scores["cashflow"] * 0.20 + scores["balance_sheet"] * 0.15 + scores["stability"] * 0.20
    verdict = "good" if total >= 82 else "bad" if total < 60 else "mid"

    return QuarterlyAnalyzerResult(
        ticker=ticker.upper(), verdict=verdict, total_score=round(total, 2), component_scores={k: round(v, 2) for k, v in scores.items()},
        positives=positives[:5], negatives=negatives[:5], summary=f"{ticker.upper()} report looks {verdict.upper()} ({total:.1f}/100)."
    )


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    rows, scores = [], []
    total_value = total_pnl = 0.0
    for row in db.scalars(select(PositionRow)).all():
        try:
            p = _position_view(row)
            rows.append(p)
            scores.append(_compute_stock_score(row.ticker))
            total_value += p.market_value
            total_pnl += p.pnl
        except HTTPException:
            continue

    return templates.TemplateResponse(request, "index.html", {"positions": rows, "scores": scores, "total_value": total_value, "total_pnl": total_pnl, "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")})


@app.post("/positions")
def add_position(ticker: str = Form(...), shares: float = Form(...), avg_cost: float = Form(...), db: Session = Depends(get_db)):
    payload = PositionCreate(ticker=ticker.upper(), shares=shares, avg_cost=avg_cost)
    row = db.scalar(select(PositionRow).where(PositionRow.ticker == payload.ticker))
    if row:
        row.shares = payload.shares
        row.avg_cost = payload.avg_cost
    else:
        db.add(PositionRow(ticker=payload.ticker, shares=payload.shares, avg_cost=payload.avg_cost))
    db.commit()
    return {"ok": True, "ticker": payload.ticker}


@app.get("/api/positions", response_model=List[Position])
def get_positions(db: Session = Depends(get_db)):
    return [_position_view(row) for row in db.scalars(select(PositionRow)).all()]


@app.get("/api/portfolio/summary", response_model=PortfolioSummary)
def portfolio_summary(db: Session = Depends(get_db)):
    positions = [_position_view(row) for row in db.scalars(select(PositionRow)).all()]
    total_value = sum(p.market_value for p in positions)
    total_pnl = sum(p.pnl for p in positions)
    top = max(positions, key=lambda p: p.market_value).ticker if positions else None
    return PortfolioSummary(total_value=round(total_value, 2), total_pnl=round(total_pnl, 2), top_position=top, positions_count=len(positions))


@app.get("/api/stocks/{ticker}/quarterly", response_model=List[QuarterInsight])
def quarterly(ticker: str):
    tk = yf.Ticker(ticker.upper())
    df = tk.quarterly_income_stmt
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No quarterly data for {ticker}")
    out = []
    for col in list(df.columns)[:4]:
        out.append(QuarterInsight(quarter=col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col), revenue=float(df.loc["Total Revenue", col]) if "Total Revenue" in df.index else None, eps=float(df.loc["Basic EPS", col]) if "Basic EPS" in df.index else None))
    return out


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
    return {"ticker": ticker.upper(), "news": [{"title": i.get("content", {}).get("title"), "url": i.get("content", {}).get("canonicalUrl", {}).get("url")} for i in items[:10]]}


@app.get("/api/screener")
def simple_screener(tickers: str, min_score: float = 60):
    names = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    scored = [ _compute_stock_score(t) for t in names ]
    filtered = [s for s in scored if s.total >= min_score]
    filtered.sort(key=lambda x: x.total, reverse=True)
    return {"count": len(filtered), "results": [s.model_dump() for s in filtered]}


@app.get("/health")
def health():
    return {"ok": True}
