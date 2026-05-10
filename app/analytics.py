from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Iterable

import numpy as np


@dataclass
class ReturnStats:
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    sortino: float


def pct_returns(prices: Iterable[float]) -> list[float]:
    p = list(prices)
    if len(p) < 2:
        return []
    return [(p[i] / p[i - 1] - 1) for i in range(1, len(p)) if p[i - 1] != 0]


def max_drawdown(prices: Iterable[float]) -> float:
    p = np.array(list(prices), dtype=float)
    if p.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(p)
    dd = (p - peaks) / peaks
    return float(dd.min())


def compute_return_stats(prices: Iterable[float], risk_free_rate: float = 0.02) -> ReturnStats:
    r = pct_returns(prices)
    if not r:
        return ReturnStats(0, 0, 0, 0, 0)
    mu = mean(r)
    sigma = pstdev(r) if len(r) > 1 else 0.0
    ann_ret = ((1 + mu) ** 252) - 1
    ann_vol = sigma * sqrt(252)
    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = [x - rf_daily for x in r]
    sharpe = (mean(excess) / sigma * sqrt(252)) if sigma > 0 else 0.0
    downside = [min(0, x - rf_daily) for x in r]
    downside_dev = sqrt(sum(x * x for x in downside) / len(downside)) * sqrt(252) if downside else 0.0
    sortino = ((ann_ret - risk_free_rate) / downside_dev) if downside_dev > 0 else 0.0
    mdd = max_drawdown(prices)
    return ReturnStats(annual_return=float(ann_ret), annual_volatility=float(ann_vol), sharpe=float(sharpe), max_drawdown=float(mdd), sortino=float(sortino))


def monte_carlo_paths(start_price: float, mu: float, sigma: float, days: int = 252, n_sims: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=(n_sims, days))
    paths = np.exp(np.cumsum(shocks, axis=1)) * start_price
    return paths


def dcf_intrinsic_value(
    fcf0: float,
    growth_years: int = 5,
    growth_rate: float = 0.08,
    terminal_growth: float = 0.025,
    discount_rate: float = 0.10,
    net_debt: float = 0.0,
    shares_outstanding: float = 1.0,
) -> float:
    if shares_outstanding <= 0:
        return 0.0
    cashflows = []
    fcf = fcf0
    for _ in range(growth_years):
        fcf *= 1 + growth_rate
        cashflows.append(fcf)
    pv = sum(cf / ((1 + discount_rate) ** (i + 1)) for i, cf in enumerate(cashflows))
    terminal = (cashflows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal = terminal / ((1 + discount_rate) ** growth_years)
    equity_value = pv + pv_terminal - net_debt
    return float(equity_value / shares_outstanding)


def composite_score(quality: float, growth: float, valuation: float, momentum: float, risk: float) -> float:
    return 0.25 * quality + 0.2 * growth + 0.2 * valuation + 0.15 * momentum + 0.2 * risk
