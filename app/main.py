from __future__ import annotations

from datetime import datetime
import os
import time
import logging
from statistics import mean

from app.analytics import compute_return_stats, composite_score, dcf_intrinsic_value, monte_carlo_paths
from typing import Dict, List

import numpy as np
import yfinance as yf
from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request, Header
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine, func, select, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from app.security import create_access_token, create_refresh_token, decode_token, hash_password, verify_password
from app.config import DATABASE_URL
from app.intelligence import classify_news, source_rank

app = FastAPI(title="Portfolio Tracker")
templates = Jinja2Templates(directory="app/templates")
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
logger = logging.getLogger("portfolio_tracker")
logging.basicConfig(level=logging.INFO)
RATE_LIMIT: dict[str, list[float]] = {}
CACHE: dict[str, tuple[float, object]] = {}


class Base(DeclarativeBase):
    pass


class UserRow(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), default="")
    password_salt: Mapped[str] = mapped_column(String(64), default="")
    role: Mapped[str] = mapped_column(String(20), default="user")
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



class RefreshTokenRow(Base):
    __tablename__ = "refresh_tokens"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token_jti: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    revoked: Mapped[int] = mapped_column(Integer, default=0)


class AuditLogRow(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    action: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())




class NewsAnalysisRow(Base):
    __tablename__ = "news_analysis"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    title: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(200), default="")
    sentiment: Mapped[str] = mapped_column(String(20), default="neutral")
    event_type: Mapped[str] = mapped_column(String(30), default="general")
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

class PriceSnapshotRow(Base):
    __tablename__ = "price_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    close: Mapped[float] = mapped_column(Float)


class NewsSnapshotRow(Base):
    __tablename__ = "news_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    published_at: Mapped[str] = mapped_column(String(64), default="")
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(200), default="")


class EarningsEventRow(Base):
    __tablename__ = "earnings_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    event_date: Mapped[str] = mapped_column(String(32), index=True)
    event_type: Mapped[str] = mapped_column(String(50), default="earnings")


class PortfolioNAVRow(Base):
    __tablename__ = "portfolio_nav"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    nav: Mapped[float] = mapped_column(Float)

Base.metadata.create_all(engine)


class UserCreate(BaseModel):
    email: str
    password: str


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


def _rate_limit(key: str, max_req: int = 60, window_sec: int = 60):
    now = time.time()
    arr = [t for t in RATE_LIMIT.get(key, []) if now - t < window_sec]
    if len(arr) >= max_req:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    arr.append(now)
    RATE_LIMIT[key] = arr


def _cached(key: str, ttl: int, fn):
    now = time.time()
    if key in CACHE and now - CACHE[key][0] < ttl:
        return CACHE[key][1]
    value = fn()
    CACHE[key] = (now, value)
    return value


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _quote(ticker: str) -> float:
    _rate_limit(f"quote:{ticker}", max_req=120, window_sec=60)
    hist = _cached(f"quote:{ticker}", ttl=30, fn=lambda: yf.Ticker(ticker).history(period="1d"))
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
    ph, ps = hash_password(payload.password)
    user = UserRow(email=payload.email, password_hash=ph, password_salt=ps)
    db.add(user)
    db.commit()
    db.refresh(user)
    default_portfolio = PortfolioRow(user_id=user.id, name="Main")
    db.add(default_portfolio)
    db.commit()
    access = create_access_token(str(user.id), user.role)
    refresh = create_refresh_token(str(user.id))
    payload = decode_token(refresh)
    db.add(RefreshTokenRow(user_id=user.id, token_jti=payload["jti"]))
    db.commit()
    return {"user_id": user.id, "email": user.email, "default_portfolio_id": default_portfolio.id, "access_token": access, "refresh_token": refresh}


@app.post("/api/login")
def login(email: str, password: str, db: Session = Depends(get_db)):
    user = db.scalar(select(UserRow).where(UserRow.email == email))
    if not user or not verify_password(password, user.password_hash, user.password_salt):
        raise HTTPException(status_code=401, detail="invalid credentials")
    access = create_access_token(str(user.id), user.role)
    refresh = create_refresh_token(str(user.id))
    payload = decode_token(refresh)
    db.add(RefreshTokenRow(user_id=user.id, token_jti=payload["jti"]))
    db.commit()
    return {"access_token": access, "refresh_token": refresh}


def _require_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> UserRow:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    user = db.get(UserRow, int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="invalid token")
    return user




def _require_role(required: str):
    def _inner(user: UserRow = Depends(_require_user)):
        if user.role != required:
            raise HTTPException(status_code=403, detail="insufficient role")
        return user
    return _inner


@app.post("/api/token/refresh")
def token_refresh(refresh_token: str, db: Session = Depends(get_db)):
    payload = decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="invalid refresh token")
    row = db.scalar(select(RefreshTokenRow).where(RefreshTokenRow.token_jti == payload.get("jti")))
    if not row or row.revoked:
        raise HTTPException(status_code=401, detail="revoked refresh token")
    user = db.get(UserRow, int(payload["sub"]))
    return {"access_token": create_access_token(str(user.id), user.role)}


@app.post("/api/token/revoke")
def revoke_token(refresh_token: str, db: Session = Depends(get_db), user: UserRow = Depends(_require_user)):
    payload = decode_token(refresh_token)
    row = db.scalar(select(RefreshTokenRow).where(RefreshTokenRow.token_jti == payload.get("jti"), RefreshTokenRow.user_id == user.id))
    if row:
        row.revoked = 1
        db.add(AuditLogRow(user_id=user.id, action="revoke_refresh_token"))
        db.commit()
    return {"ok": True}


@app.get("/api/admin/audit")
def audit_logs(db: Session = Depends(get_db), admin: UserRow = Depends(_require_role("admin"))):
    logs = db.scalars(select(AuditLogRow).order_by(AuditLogRow.id.desc())).all()
    return [{"user_id": l.user_id, "action": l.action, "created_at": str(l.created_at)} for l in logs[:200]]


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




@app.get("/api/ops/metrics")
def ops_metrics(db: Session = Depends(get_db)):
    return {
        "users": len(db.scalars(select(UserRow)).all()),
        "portfolios": len(db.scalars(select(PortfolioRow)).all()),
        "positions": len(db.scalars(select(PositionRow)).all()),
        "cache_keys": len(CACHE),
        "rate_limit_keys": len(RATE_LIMIT),
    }


@app.post("/api/jobs/snapshot-prices")
def snapshot_prices(tickers: str, db: Session = Depends(get_db), user: UserRow = Depends(_require_user)):
    for t in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
        px = _quote(t)
        db.add(PriceSnapshotRow(ticker=t, close=px))
    db.commit()
    return {"ok": True}



@app.get("/api/stocks/{ticker}/quarterly/deep-dive")
def quarterly_deep_dive(ticker: str):
    tk = yf.Ticker(ticker.upper())
    fin = tk.quarterly_income_stmt
    cf = tk.quarterly_cashflow
    if fin is None or fin.empty:
        raise HTTPException(status_code=404, detail="No quarterly data")

    def pick(metric: str):
        if metric not in fin.index:
            return []
        vals = [_safe_float(v) for v in list(fin.loc[metric].values)[:5]]
        return [v for v in vals if v is not None]

    rev = pick("Total Revenue")
    ni = pick("Net Income")
    gp = pick("Gross Profit")
    op = pick("Operating Income")

    def yoy(vals):
        return ((vals[0] - vals[4]) / abs(vals[4]) * 100) if len(vals) >= 5 and vals[4] else None

    def qoq(vals):
        return ((vals[0] - vals[1]) / abs(vals[1]) * 100) if len(vals) >= 2 and vals[1] else None

    qoq_rev = qoq(rev)
    yoy_rev = yoy(rev)
    qoq_ni = qoq(ni)
    yoy_ni = yoy(ni)

    gm = (gp[0] / rev[0] * 100) if gp and rev else None
    om = (op[0] / rev[0] * 100) if op and rev else None
    nm = (ni[0] / rev[0] * 100) if ni and rev else None

    cfo = []
    if cf is not None and not cf.empty and "Operating Cash Flow" in cf.index:
        cfo = [_safe_float(v) for v in list(cf.loc["Operating Cash Flow"].values)[:5]]
        cfo = [v for v in cfo if v is not None]
    cash_quality = ((cfo[0] / ni[0]) if cfo and ni and ni[0] else None)

    red_flags = []
    if nm is not None and nm < 0:
        red_flags.append("Negative net margin")
    if cash_quality is not None and cash_quality < 0.8:
        red_flags.append("Weak cash conversion vs net income")

    sector = (tk.info or {}).get("sector", "Unknown")
    kpi_pack = {
        "Technology": ["Revenue growth", "Gross margin", "R&D intensity", "FCF margin"],
        "Financial Services": ["NIM", "Loan growth", "Credit quality", "CET1"],
        "Consumer Cyclical": ["Same-store sales", "Inventory turns", "Gross margin"],
    }.get(sector, ["Revenue growth", "Operating margin", "Cash conversion"])

    return {
        "ticker": ticker.upper(),
        "sector": sector,
        "qoq": {"revenue_pct": qoq_rev, "net_income_pct": qoq_ni},
        "yoy": {"revenue_pct": yoy_rev, "net_income_pct": yoy_ni},
        "margin_bridge": {"gross_margin": gm, "operating_margin": om, "net_margin": nm},
        "cash_quality": {"cfo_to_net_income": cash_quality},
        "red_flags": red_flags,
        "kpi_pack": kpi_pack,
        "consensus_stub": "Add provider integration for estimate surprise engine.",
    }


@app.get("/api/stocks/{ticker}/news/intelligence")
def news_intelligence(ticker: str, portfolio_id: int, db: Session = Depends(get_db)):
    items = (yf.Ticker(ticker.upper()).news or [])[:30]
    dedup = {}
    for item in items:
        c = item.get("content", {})
        title = (c.get("title") or "").strip()
        if not title:
            continue
        key = title.lower()
        if key in dedup:
            continue
        dedup[key] = {
            "title": title,
            "summary": c.get("summary") or "",
            "source": (c.get("provider", {}) or {}).get("displayName") or "",
            "url": (c.get("canonicalUrl", {}) or {}).get("url"),
        }

    analyzed = []
    for x in dedup.values():
        insight = classify_news(x["title"], x["summary"])
        analyzed.append({**x, **insight.__dict__, "source_rank": source_rank(x["source"])})

    analyzed.sort(key=lambda z: (z["source_rank"], -z["confidence"]))

    position = db.scalar(select(PositionRow).where(PositionRow.portfolio_id == portfolio_id, PositionRow.ticker == ticker.upper()))
    weight = 0.0
    if position:
        positions = [_position_view(row) for row in db.scalars(select(PositionRow).where(PositionRow.portfolio_id == portfolio_id)).all()]
        total = sum(p.market_value for p in positions) or 1
        weight = (_position_view(position).market_value / total) * 100

    for a in analyzed[:20]:
        db.add(NewsAnalysisRow(ticker=ticker.upper(), title=a["title"], source=a["source"], sentiment=a["sentiment"], event_type=a["event_type"], confidence=a["confidence"]))
    db.commit()

    portfolio_hint = "low impact" if weight < 3 else "medium impact" if weight < 10 else "high impact"

    return {
        "ticker": ticker.upper(),
        "portfolio_weight_pct": round(weight, 2),
        "portfolio_impact": portfolio_hint,
        "news_count": len(analyzed),
        "items": analyzed[:20],
    }


@app.get("/health")
def health():
    return {"ok": True}
