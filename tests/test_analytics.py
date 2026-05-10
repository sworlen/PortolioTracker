import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.analytics import compute_return_stats, dcf_intrinsic_value, pct_returns


def test_pct_returns_non_empty():
    r = pct_returns([100, 110, 121])
    assert len(r) == 2


def test_return_stats_shape():
    stats = compute_return_stats([100, 101, 102, 99, 105])
    assert hasattr(stats, "sharpe")


def test_dcf_positive():
    v = dcf_intrinsic_value(fcf0=1000, shares_outstanding=100)
    assert v > 0
