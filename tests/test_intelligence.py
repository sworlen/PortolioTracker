import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.intelligence import classify_news, source_rank


def test_classify_news_negative_guidance():
    i = classify_news("Company cuts guidance after earnings miss")
    assert i.sentiment in {"negative", "neutral"}
    assert i.event_type in {"guidance", "earnings"}


def test_source_rank():
    assert source_rank("Reuters") == 1
