from __future__ import annotations

from datetime import datetime
from statistics import mean

from app.analytics import compute_return_stats, composite_score, dcf_intrinsic_value, monte_carlo_paths
from typing import Dict, List

import numpy as np
import yfinance as yf
from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

app = FastAPI(title="Portfolio Tracker")
templates = Jinja2Templates(directory="app/templates")
engine = create_engine("sqlite:///portfolio.db", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


class UserRow(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    portfolios: Mapped[List["PortfolioRow"]] = relationship(back_populates="user")


class PortfolioRow(Base):
    __tablename__ = "portfolios"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(120), default="Main")
    user: Mapped[UserRow] = relationship(back_populates="portfolios")
    positions: Mapped[List["PositionRow"]] = relationship(back_populates="portfolio")


class PositionRow(Base):
    __tablename__ = "positions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    shares: Mapped[float] = mapped_column(Float)
    avg_cost: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    portfolio: Mapped[PortfolioRow] = relationship(back_populates="positions")


class TransactionRow(Base):
    __tablename__ = "transactions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    side: Mapped[str] = mapped_column(String(8))
    shares: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    executed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SavedScreenRow(Base):
    __tablename__ = "saved_screens"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    tickers_csv: Mapped[str] = mapped_column(String(2000), default="")
    min_score: Mapped[float] = mapped_column(Float, default=60.0)




class ThesisRow(Base):
    __tablename__ = "theses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    thesis: Mapped[str] = mapped_column(String(3000))
    must_happen: Mapped[str] = mapped_column(String(3000), default="")
    invalidation: Mapped[str] = mapped_column(String(3000), default="")
    status: Mapped[str] = mapped_column(String(16), default="active")


class AlertRuleRow(Base):
    __tablename__ = "alert_rules"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    min_score: Mapped[float] = mapped_column(Float, default=0)
    max_score: Mapped[float] = mapped_column(Float, default=100)
    note: Mapped[str] = mapped_column(String(500), default="")


class WatchlistRow(Base):
    __tablename__ = "watchlist"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    stage: Mapped[str] = mapped_column(String(20), default="idea")

Base.metadata.create_all(engine)


class UserCreate(BaseModel):
    email: str


class PortfolioCreate(BaseModel):
    user_id: int
    name: str = "Main"


class PositionCreate(BaseModel):
    portfolio_id: int
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


class PortfolioSummary(BaseModel):
    total_value: float
    total_pnl: float
    top_position: str | None
    positions_count: int


class StockScore(BaseModel):
    ticker: str
    total: float
    quality: float
    growth: float
    valuation: float
    momentum: float
    risk: float


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
    return Position(ticker=row.ticker, shares=row.shares, avg_cost=row.avg_cost, last_price=last, market_value=mv, pnl=pnl)


def _safe_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
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
        ret_6m = ((hist["Close"].iloc[-1] / hist["Close"].iloc[-126]) - 1) * 100
        momentum = max(0.0, min(100.0, 50 + ret_6m))
    penalties = min(30, (debt_to_equity or 100) / 10) + max(0, ((beta or 1) - 1) * 15)
    risk = max(0.0, min(100.0, 80 - penalties))
    total = composite_score(quality, growth, valuation, momentum, risk)
    return StockScore(ticker=ticker.upper(), total=round(total, 2), quality=round(quality, 2), growth=round(growth, 2), valuation=round(valuation, 2), momentum=round(momentum, 2), risk=round(risk, 2))


# 1) Auth + Multi-portfolio data model
@app.post("/api/users")
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    user = UserRow(email=payload.email)
    db.add(user)
    db.commit()
    db.refresh(user)
    default_portfolio = PortfolioRow(user_id=user.id, name="Main")
    db.add(default_portfolio)
    db.commit()
    return {"user_id": user.id, "email": user.email, "default_portfolio_id": default_portfolio.id}


@app.post("/api/portfolios")
def create_portfolio(payload: PortfolioCreate, db: Session = Depends(get_db)):
    db.add(PortfolioRow(user_id=payload.user_id, name=payload.name))
    db.commit()
    return {"ok": True}


# 2) background refresh pipeline bootstrap
@app.post("/api/jobs/refresh")
def refresh_quotes(tickers: str = Query(..., description="Comma-separated tickers")):
    names = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    data = {t: _quote(t) for t in names}
    return {"run_at": datetime.utcnow().isoformat(), "quotes": data}


@app.post("/positions")
def add_position(portfolio_id: int = Form(...), ticker: str = Form(...), shares: float = Form(...), avg_cost: float = Form(...), db: Session = Depends(get_db)):
    payload = PositionCreate(portfolio_id=portfolio_id, ticker=ticker.upper(), shares=shares, avg_cost=avg_cost)
    row = db.scalar(select(PositionRow).where(PositionRow.portfolio_id == payload.portfolio_id, PositionRow.ticker == payload.ticker))
    if row:
        row.shares = payload.shares
        row.avg_cost = payload.avg_cost
    else:
        db.add(PositionRow(portfolio_id=payload.portfolio_id, ticker=payload.ticker, shares=payload.shares, avg_cost=payload.avg_cost))
    db.add(TransactionRow(portfolio_id=payload.portfolio_id, ticker=payload.ticker, side="BUY", shares=payload.shares, price=payload.avg_cost))
    db.commit()
    return {"ok": True, "ticker": payload.ticker}


@app.get("/api/positions", response_model=List[Position])
def get_positions(portfolio_id: int, db: Session = Depends(get_db)):
    return [_position_view(row) for row in db.scalars(select(PositionRow).where(PositionRow.portfolio_id == portfolio_id)).all()]


@app.get("/api/portfolio/summary", response_model=PortfolioSummary)
def portfolio_summary(portfolio_id: int, db: Session = Depends(get_db)):
    positions = [_position_view(row) for row in db.scalars(select(PositionRow).where(PositionRow.portfolio_id == portfolio_id)).all()]
    total_value = sum(p.market_value for p in positions)
    total_pnl = sum(p.pnl for p in positions)
    top = max(positions, key=lambda p: p.market_value).ticker if positions else None
    return PortfolioSummary(total_value=round(total_value, 2), total_pnl=round(total_pnl, 2), top_position=top, positions_count=len(positions))


# 3) quarterly analyzer v2
@app.get("/api/stocks/{ticker}/quarterly-analyzer")
def quarterly_analyzer(ticker: str):
    tk = yf.Ticker(ticker.upper())
    fin = tk.quarterly_income_stmt
    if fin is None or fin.empty:
        raise HTTPException(status_code=404, detail="No quarterly data")
    revs = [_safe_float(v) for v in list(fin.loc["Total Revenue"].values)[:4]] if "Total Revenue" in fin.index else []
    eps = [_safe_float(v) for v in list(fin.loc["Basic EPS"].values)[:4]] if "Basic EPS" in fin.index else []
    growth = _quarterly_growth([v for v in revs if v is not None]) if revs else 0.0
    eps_growth = _quarterly_growth([v for v in eps if v is not None]) if eps else 0.0
    score = 50 + growth * 0.3 + eps_growth * 0.3
    verdict = "good" if score >= 82 else "bad" if score < 60 else "mid"
    return {"ticker": ticker.upper(), "verdict": verdict, "score": round(max(0, min(100, score)), 2), "growth_pct": round(growth, 2), "eps_growth_pct": round(eps_growth, 2)}


# 4) news impact engine (simple version)
@app.get("/api/stocks/{ticker}/news-impact")
def news_impact(ticker: str, portfolio_id: int, db: Session = Depends(get_db)):
    items = (yf.Ticker(ticker.upper()).news or [])[:5]
    position = db.scalar(select(PositionRow).where(PositionRow.portfolio_id == portfolio_id, PositionRow.ticker == ticker.upper()))
    weight = 0.0
    if position:
        positions = [_position_view(row) for row in db.scalars(select(PositionRow).where(PositionRow.portfolio_id == portfolio_id)).all()]
        total = sum(p.market_value for p in positions) or 1
        weight = (_position_view(position).market_value / total) * 100
    return {"ticker": ticker.upper(), "portfolio_weight_pct": round(weight, 2), "news_count": len(items), "impact_hint": "Higher weight means higher portfolio sensitivity."}


# 5) screener v2 + saved screens
@app.post("/api/screens/save")
def save_screen(user_id: int, name: str, tickers: str, min_score: float = 60, db: Session = Depends(get_db)):
    db.add(SavedScreenRow(user_id=user_id, name=name, tickers_csv=tickers, min_score=min_score))
    db.commit()
    return {"ok": True}


@app.get("/api/screener")
def screener(tickers: str, min_score: float = 60):
    names = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    scored = [_compute_stock_score(t) for t in names]
    filtered = sorted([s for s in scored if s.total >= min_score], key=lambda x: x.total, reverse=True)
    return {"count": len(filtered), "results": [s.model_dump() for s in filtered]}


# 6) risk module (starter)
@app.get("/api/portfolio/risk")
def portfolio_risk(portfolio_id: int, db: Session = Depends(get_db)):
    positions = [_position_view(row) for row in db.scalars(select(PositionRow).where(PositionRow.portfolio_id == portfolio_id)).all()]
    total = sum(p.market_value for p in positions) or 1
    concentration = sorted([{"ticker": p.ticker, "weight_pct": round((p.market_value / total) * 100, 2)} for p in positions], key=lambda x: x["weight_pct"], reverse=True)
    top3 = sum(x["weight_pct"] for x in concentration[:3])
    return {"top_positions": concentration[:10], "top3_concentration_pct": round(top3, 2), "risk_flag": "high" if top3 > 55 else "medium" if top3 > 35 else "low"}


# 7) backtesting starter
@app.get("/api/backtest/simple")
def simple_backtest(tickers: str, start: str, end: str):
    names = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    rets = []
    for t in names:
        hist = yf.Ticker(t).history(start=start, end=end)
        if hist.empty:
            continue
        ret = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
        rets.append({"ticker": t, "return_pct": round(float(ret), 2)})
    avg = round(mean([r["return_pct"] for r in rets]), 2) if rets else 0.0
    return {"period": {"start": start, "end": end}, "count": len(rets), "average_return_pct": avg, "results": rets}


@app.get("/api/stocks/{ticker}/valuation/dcf")
def dcf_valuation(ticker: str):
    tk = yf.Ticker(ticker.upper())
    info = tk.info or {}
    fcf = float(info.get("freeCashflow") or 0)
    net_debt = float((info.get("totalDebt") or 0) - (info.get("totalCash") or 0))
    shares = float(info.get("sharesOutstanding") or 1)
    value = dcf_intrinsic_value(fcf0=fcf, net_debt=net_debt, shares_outstanding=shares)
    price = _quote(ticker.upper())
    upside = ((value / price) - 1) * 100 if price else 0
    return {"ticker": ticker.upper(), "dcf_value_per_share": round(value, 2), "last_price": round(price, 2), "upside_pct": round(upside, 2)}


@app.get("/api/stocks/{ticker}/risk-metrics")
def risk_metrics(ticker: str):
    hist = yf.Ticker(ticker.upper()).history(period="2y")
    if hist.empty:
        raise HTTPException(status_code=404, detail="No history")
    stats = compute_return_stats(hist["Close"].tolist())
    return {"ticker": ticker.upper(), **stats.__dict__}


@app.get("/api/stocks/{ticker}/monte-carlo")
def monte_carlo(ticker: str, days: int = 252, sims: int = 500):
    hist = yf.Ticker(ticker.upper()).history(period="2y")
    if hist.empty:
        raise HTTPException(status_code=404, detail="No history")
    closes = hist["Close"].tolist()
    stats = compute_return_stats(closes)
    mu = stats.annual_return
    sigma = stats.annual_volatility
    start = closes[-1]
    paths = monte_carlo_paths(start_price=start, mu=mu, sigma=sigma, days=days, n_sims=sims)
    final = paths[:, -1]
    return {
        "ticker": ticker.upper(),
        "start_price": round(start, 2),
        "days": days,
        "simulations": sims,
        "p5": round(float(np.percentile(final, 5)), 2),
        "p50": round(float(np.percentile(final, 50)), 2),
        "p95": round(float(np.percentile(final, 95)), 2),
    }




@app.post("/api/thesis")
def save_thesis(portfolio_id: int, ticker: str, thesis: str, must_happen: str = "", invalidation: str = "", db: Session = Depends(get_db)):
    row = db.scalar(select(ThesisRow).where(ThesisRow.portfolio_id == portfolio_id, ThesisRow.ticker == ticker.upper()))
    if row:
        row.thesis = thesis
        row.must_happen = must_happen
        row.invalidation = invalidation
    else:
        db.add(ThesisRow(portfolio_id=portfolio_id, ticker=ticker.upper(), thesis=thesis, must_happen=must_happen, invalidation=invalidation))
    db.commit()
    return {"ok": True}


@app.get("/api/stocks/{ticker}/quarterly-translator")
def quarterly_translator(ticker: str):
    qa = quarterly_analyzer(ticker)
    why = []
    why.append("Revenue trend improved versus prior periods." if qa["growth_pct"] > 0 else "Revenue trend weakened versus prior periods.")
    why.append("EPS trend improved, supporting profitability outlook." if qa["eps_growth_pct"] > 0 else "EPS trend deteriorated and needs attention.")
    return {
        "ticker": ticker.upper(),
        "label": qa["verdict"],
        "score": qa["score"],
        "summary": f"Quarter verdict is {qa['verdict'].upper()} based on trend signals.",
        "what_went_good": [w for w in why if "improved" in w],
        "what_went_bad": [w for w in why if "weakened" in w or "deteriorated" in w],
        "what_to_watch_next": ["Next quarter guidance", "Margin trend", "Cashflow conversion"],
    }


@app.post("/api/watchlist")
def upsert_watchlist(portfolio_id: int, ticker: str, stage: str = "idea", db: Session = Depends(get_db)):
    row = db.scalar(select(WatchlistRow).where(WatchlistRow.portfolio_id == portfolio_id, WatchlistRow.ticker == ticker.upper()))
    if row:
        row.stage = stage
    else:
        db.add(WatchlistRow(portfolio_id=portfolio_id, ticker=ticker.upper(), stage=stage))
    db.commit()
    return {"ok": True}


@app.get("/api/watchlist")
def get_watchlist(portfolio_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(select(WatchlistRow).where(WatchlistRow.portfolio_id == portfolio_id)).all()
    return [{"ticker": r.ticker, "stage": r.stage} for r in rows]


@app.post("/api/alerts")
def create_alert(portfolio_id: int, ticker: str, min_score: float = 0, max_score: float = 100, note: str = "", db: Session = Depends(get_db)):
    db.add(AlertRuleRow(portfolio_id=portfolio_id, ticker=ticker.upper(), min_score=min_score, max_score=max_score, note=note))
    db.commit()
    return {"ok": True}


@app.get("/api/alerts/check")
def check_alerts(portfolio_id: int, db: Session = Depends(get_db)):
    rules = db.scalars(select(AlertRuleRow).where(AlertRuleRow.portfolio_id == portfolio_id)).all()
    triggered = []
    for r in rules:
        score = _compute_stock_score(r.ticker).total
        if score < r.min_score or score > r.max_score:
            triggered.append({"ticker": r.ticker, "score": score, "note": r.note})
    return {"portfolio_id": portfolio_id, "triggered": triggered, "count": len(triggered)}


@app.get("/api/portfolio/daily-brief")
def daily_brief(portfolio_id: int, db: Session = Depends(get_db)):
    positions = db.scalars(select(PositionRow).where(PositionRow.portfolio_id == portfolio_id)).all()
    cards = []
    for row in positions[:10]:
        score = _compute_stock_score(row.ticker)
        news = (yf.Ticker(row.ticker).news or [])[:1]
        cards.append({
            "ticker": row.ticker,
            "score": score.total,
            "signal": "strong" if score.total >= 75 else "watch" if score.total >= 60 else "risk",
            "headline": (news[0].get("content", {}).get("title") if news else None),
        })
    return {"date": datetime.utcnow().strftime("%Y-%m-%d"), "portfolio_id": portfolio_id, "cards": cards}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html", {"updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "positions": [], "scores": [], "total_value": 0, "total_pnl": 0})


@app.get("/health")
def health():
    return {"ok": True}
