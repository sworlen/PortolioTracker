from __future__ import annotations

from dataclasses import dataclass

POSITIVE = ["beat", "raise", "upgrade", "growth", "expands", "outperform", "record"]
NEGATIVE = ["miss", "cut", "downgrade", "lawsuit", "probe", "decline", "warning"]
EVENT_RULES = {
    "guidance": ["guidance", "outlook", "forecast"],
    "legal": ["lawsuit", "investigation", "probe", "sec"],
    "m&a": ["acquire", "merger", "deal", "buyout"],
    "analyst": ["upgrade", "downgrade", "price target"],
    "earnings": ["eps", "revenue", "earnings", "quarter"],
    "macro": ["rates", "inflation", "fed", "tariff"],
}


@dataclass
class NewsInsight:
    sentiment: str
    sentiment_score: float
    event_type: str
    confidence: float
    action_hint: str


def classify_news(title: str, summary: str = "") -> NewsInsight:
    text = f"{title} {summary}".lower()
    pos = sum(1 for w in POSITIVE if w in text)
    neg = sum(1 for w in NEGATIVE if w in text)
    score = (pos - neg) / max(1, pos + neg)
    sentiment = "neutral"
    if score > 0.1:
        sentiment = "positive"
    elif score < -0.1:
        sentiment = "negative"

    event = "general"
    for k, keywords in EVENT_RULES.items():
        if any(w in text for w in keywords):
            event = k
            break

    confidence = min(0.95, 0.45 + (abs(score) * 0.35) + (0.15 if event != "general" else 0))

    action_hint = "monitor"
    if sentiment == "negative" and event in {"guidance", "legal", "earnings"}:
        action_hint = "review position risk"
    elif sentiment == "positive" and event in {"earnings", "analyst"}:
        action_hint = "validate thesis and consider add"

    return NewsInsight(sentiment=sentiment, sentiment_score=round(score, 3), event_type=event, confidence=round(confidence, 3), action_hint=action_hint)


def source_rank(source: str) -> int:
    s = (source or "").lower()
    if any(x in s for x in ["reuters", "bloomberg", "wsj", "ft"]):
        return 1
    if any(x in s for x in ["yahoo", "marketwatch", "cnbc"]):
        return 2
    return 3
