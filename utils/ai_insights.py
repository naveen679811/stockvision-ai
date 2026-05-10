"""
utils/ai_insights.py
─────────────────────
Rule-based plain-English insights engine.
Synthesises price, technical indicators, model metrics, and sentiment
into human-readable market commentary.
"""

import pandas as pd
import numpy as np


def _rsi_comment(rsi: float) -> str:
    if rsi > 70:
        return f"RSI at {rsi:.1f} is in **overbought** territory — momentum may be fading."
    elif rsi < 30:
        return f"RSI at {rsi:.1f} signals **oversold** conditions — a bounce could be near."
    elif rsi > 55:
        return f"RSI at {rsi:.1f} reflects **bullish momentum** with room to run."
    else:
        return f"RSI at {rsi:.1f} is **neutral** — no clear directional bias."


def _macd_comment(macd: float, signal: float) -> str:
    if macd > signal:
        return "MACD is **above its signal line** — bullish crossover in play."
    else:
        return "MACD is **below its signal line** — bearish pressure building."


def _bb_comment(close: float, upper: float, lower: float, mid: float) -> str:
    if close >= upper * 0.98:
        return "Price is near the **upper Bollinger Band** — potential resistance ahead."
    elif close <= lower * 1.02:
        return "Price is near the **lower Bollinger Band** — potential support zone."
    elif close > mid:
        return "Price is trading **above the Bollinger midline** — mild positive bias."
    else:
        return "Price is trading **below the Bollinger midline** — mild negative bias."


def _trend_comment(sma20: float, sma50: float) -> str:
    if sma20 > sma50:
        return "The **20-day SMA is above the 50-day SMA** (Golden Cross alignment) — medium-term uptrend."
    else:
        return "The **20-day SMA is below the 50-day SMA** (Death Cross alignment) — medium-term downtrend."


def _model_comment(accuracy: float, rmse: float) -> str:
    if accuracy >= 90:
        return f"LSTM model accuracy is **{accuracy:.1f}%** with RMSE {rmse:.2f} — high-confidence predictions."
    elif accuracy >= 80:
        return f"LSTM model accuracy is **{accuracy:.1f}%** — moderate confidence in the forecast."
    else:
        return f"LSTM model accuracy is **{accuracy:.1f}%** — treat forecast with caution; consider retraining."


def _sentiment_comment(score: float, overall: str) -> str:
    return f"News sentiment score is **{score:.0f}/100** ({overall}) — market narrative is {'supportive' if score >= 55 else 'cautionary'}."


def _overall_verdict(rsi, macd, signal, close, upper, lower, sentiment_score) -> str:
    bull_pts = 0
    if rsi < 55:   bull_pts += 1
    if rsi > 45:   bull_pts += 1
    if macd > signal: bull_pts += 1
    if close < upper * 0.97: bull_pts += 1
    if sentiment_score >= 55: bull_pts += 1

    if bull_pts >= 4:
        return "🟢 **Overall Outlook: BULLISH** — Multiple indicators align positively."
    elif bull_pts >= 3:
        return "🟡 **Overall Outlook: NEUTRAL-BULLISH** — Cautiously optimistic."
    elif bull_pts >= 2:
        return "🟡 **Overall Outlook: NEUTRAL** — Mixed signals; wait for confirmation."
    else:
        return "🔴 **Overall Outlook: BEARISH** — Risk-off signals dominate."


def generate_insights(df: pd.DataFrame, metrics: dict, sentiment: dict) -> list[str]:
    """
    Generate a list of insight strings from latest indicator values,
    model performance, and sentiment aggregates.
    """
    latest = df.iloc[-1]
    insights = []

    # Trend
    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        insights.append(_trend_comment(latest["SMA_20"], latest["SMA_50"]))

    # RSI
    if "RSI_14" in df.columns:
        insights.append(_rsi_comment(latest["RSI_14"]))

    # MACD
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        insights.append(_macd_comment(latest["MACD"], latest["MACD_Signal"]))

    # Bollinger
    if all(k in df.columns for k in ["BB_Upper", "BB_Lower", "BB_Middle"]):
        insights.append(_bb_comment(
            latest["Close"], latest["BB_Upper"],
            latest["BB_Lower"], latest["BB_Middle"]
        ))

    # Model
    if metrics:
        insights.append(_model_comment(metrics.get("Accuracy", 0), metrics.get("RMSE", 0)))

    # Sentiment
    if sentiment:
        insights.append(_sentiment_comment(sentiment.get("score", 50), sentiment.get("overall", "Neutral")))

    # Verdict
    rsi = latest.get("RSI_14", 50)
    macd = latest.get("MACD", 0)
    sig  = latest.get("MACD_Signal", 0)
    up   = latest.get("BB_Upper", latest["Close"] * 1.05)
    lo   = latest.get("BB_Lower", latest["Close"] * 0.95)
    insights.append(_overall_verdict(rsi, macd, sig, latest["Close"], up, lo,
                                     sentiment.get("score", 50) if sentiment else 50))

    return insights
