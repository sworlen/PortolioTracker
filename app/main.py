from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Dict, List

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
    total = quality * 0.25 + growth * 0.20 + valuation * 0.20 + momentum * 0.15 + risk * 0.20
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


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html", {"updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "positions": [], "scores": [], "total_value": 0, "total_pnl": 0})


@app.get("/health")
def health():
    return {"ok": True}
