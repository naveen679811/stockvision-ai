"""
utils/sentiment.py
──────────────────
Finance news fetching + NLP sentiment scoring.
Uses NewsAPI (if key available) with TextBlob fallback.
"""

import os
import logging
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
import pandas as pd

logger = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")


def fetch_news(query: str, days_back: int = 7, max_articles: int = 20) -> list[dict]:
    """
    Fetch recent finance news articles.
    Falls back to a curated mock feed if no API key is set.
    """
    if NEWS_API_KEY:
        return _fetch_newsapi(query, days_back, max_articles)
    else:
        return _mock_news(query)


def _fetch_newsapi(query: str, days_back: int, max_articles: int) -> list[dict]:
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}+stock&from={from_date}&sortBy=relevancy"
        f"&language=en&pageSize={max_articles}&apiKey={NEWS_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=8)
        data = resp.json()
        articles = data.get("articles", [])
        return [
            {
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "source":      a.get("source", {}).get("name", ""),
                "url":         a.get("url", ""),
                "published":   a.get("publishedAt", ""),
            }
            for a in articles if a.get("title")
        ]
    except Exception as e:
        logger.warning(f"NewsAPI fetch failed: {e}")
        return _mock_news(query)


def _mock_news(query: str) -> list[dict]:
    """Return placeholder articles for demo mode."""
    templates = [
        f"{query} reports strong quarterly earnings, beats analyst expectations",
        f"Analysts upgrade {query} to 'Buy' amid robust demand outlook",
        f"{query} announces share buyback programme worth $5B",
        f"Market volatility impacts {query} amid macroeconomic uncertainty",
        f"{query} faces regulatory scrutiny over data practices",
        f"Institutional investors increase stakes in {query}",
        f"{query} partners with leading AI firm to accelerate growth",
        f"Supply chain concerns weigh on {query} short-term outlook",
    ]
    return [
        {"title": t, "description": t + " — full article.",
         "source": "Demo Feed", "url": "#", "published": datetime.now().isoformat()}
        for t in templates
    ]


def analyze_sentiment(articles: list[dict]) -> pd.DataFrame:
    """
    Score each article with TextBlob polarity.

    Returns DataFrame with columns:
      title, source, published, polarity, subjectivity, label, score_pct
    """
    rows = []
    for a in articles:
        text = (a.get("title", "") + " " + a.get("description", "")).strip()
        blob = TextBlob(text)
        pol  = blob.sentiment.polarity      # −1 … +1
        sub  = blob.sentiment.subjectivity  # 0 … 1

        if pol >  0.1:
            label = "Positive"
        elif pol < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

        rows.append({
            "title":          a.get("title", ""),
            "source":         a.get("source", ""),
            "published":      a.get("published", ""),
            "url":            a.get("url", "#"),
            "polarity":       round(pol, 4),
            "subjectivity":   round(sub, 4),
            "label":          label,
            "score_pct":      round((pol + 1) / 2 * 100, 1),  # 0–100 scale
        })

    return pd.DataFrame(rows)


def aggregate_sentiment(df: pd.DataFrame) -> dict:
    """
    Compute aggregate sentiment metrics from the article DataFrame.
    """
    if df.empty:
        return {"overall": "Neutral", "score": 50.0,
                "positive": 0, "negative": 0, "neutral": 0, "summary": "No data."}

    counts = df["label"].value_counts().to_dict()
    avg_score = df["score_pct"].mean()

    if avg_score >= 60:
        overall = "Bullish 🟢"
    elif avg_score <= 40:
        overall = "Bearish 🔴"
    else:
        overall = "Neutral 🟡"

    return {
        "overall":  overall,
        "score":    round(avg_score, 1),
        "positive": counts.get("Positive", 0),
        "negative": counts.get("Negative", 0),
        "neutral":  counts.get("Neutral", 0),
        "summary":  (
            f"Analysed {len(df)} articles. "
            f"Sentiment score: {avg_score:.1f}/100. "
            f"Market outlook: {overall}."
        ),
    }
