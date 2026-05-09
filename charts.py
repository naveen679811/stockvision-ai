"""
utils/charts.py
───────────────
All Plotly figure builders for StockVision AI.
Dark finance theme throughout. Every chart is interactive.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Design Tokens ──────────────────────────────────────────────────────────────
THEME = dict(
    bg        = "#0D0F14",
    bg2       = "#151821",
    grid      = "#1E2330",
    text      = "#E2E8F0",
    text_muted= "#64748B",
    accent    = "#00D4FF",
    green     = "#00E676",
    red       = "#FF4444",
    yellow    = "#FFD600",
    purple    = "#A855F7",
    orange    = "#FF6B35",
    font      = "IBM Plex Mono, monospace",
)

BASE_LAYOUT = dict(
    template        = "plotly_dark",
    paper_bgcolor   = THEME["bg"],
    plot_bgcolor    = THEME["bg"],
    font            = dict(family=THEME["font"], color=THEME["text"], size=11),
    xaxis           = dict(gridcolor=THEME["grid"], showgrid=True, zeroline=False),
    yaxis           = dict(gridcolor=THEME["grid"], showgrid=True, zeroline=False),
    legend          = dict(bgcolor="rgba(0,0,0,0)", bordercolor=THEME["grid"]),
    margin          = dict(l=10, r=10, t=40, b=10),
    hovermode       = "x unified",
)


def _base_fig(**extra_layout) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**{**BASE_LAYOUT, **extra_layout})
    return fig


# ── Candlestick Chart ──────────────────────────────────────────────────────────
def candlestick_chart(df: pd.DataFrame, ticker: str, show_volume: bool = True) -> go.Figure:
    rows = 2 if show_volume else 1
    specs = [[{"secondary_y": False}]] * rows
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25] if show_volume else [1.0],
        specs=specs,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=THEME["green"],
        decreasing_line_color=THEME["red"],
        name="OHLC",
    ), row=1, col=1)

    # SMA overlays
    for col, color, dash in [
        ("SMA_20", THEME["accent"],  "solid"),
        ("SMA_50", THEME["yellow"],  "dash"),
        ("EMA_12", THEME["purple"],  "dot"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col,
                line=dict(color=color, width=1.2, dash=dash), opacity=0.8,
            ), row=1, col=1)

    # Volume
    if show_volume and "Volume" in df.columns:
        colors = [THEME["green"] if c >= o else THEME["red"]
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=colors, opacity=0.6, name="Volume",
        ), row=2, col=1)

    fig.update_layout(
        **BASE_LAYOUT,
        title=f"<b>{ticker}</b> — Price & Volume",
        xaxis_rangeslider_visible=False,
        height=600,
    )
    return fig


# ── Prediction Chart ───────────────────────────────────────────────────────────
def prediction_chart(
    dates_train, actual_train,
    dates_test,  actual_test, predicted_test,
    dates_future, future_preds,
    ticker: str,
) -> go.Figure:
    fig = _base_fig(title=f"<b>{ticker}</b> — LSTM Price Prediction", height=520)

    fig.add_trace(go.Scatter(
        x=dates_train, y=actual_train,
        name="Historical (Train)", mode="lines",
        line=dict(color=THEME["text_muted"], width=1),
    ))
    fig.add_trace(go.Scatter(
        x=dates_test, y=actual_test,
        name="Actual (Test)", mode="lines",
        line=dict(color=THEME["accent"], width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=dates_test, y=predicted_test,
        name="LSTM Prediction", mode="lines",
        line=dict(color=THEME["orange"], width=1.8, dash="dash"),
    ))

    if future_preds is not None and len(future_preds):
        # Confidence band (simple ±2% envelope)
        upper = future_preds * 1.02
        lower = future_preds * 0.98

        fig.add_trace(go.Scatter(
            x=list(dates_future) + list(reversed(dates_future)),
            y=list(upper) + list(reversed(lower)),
            fill="toself",
            fillcolor="rgba(0,212,255,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Forecast Band",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=dates_future, y=future_preds,
            name="Forecast", mode="lines+markers",
            line=dict(color=THEME["green"], width=2),
            marker=dict(size=4),
        ))

    fig.add_vline(
        x=dates_test[0] if len(dates_test) else 0,
        line_dash="dot", line_color=THEME["yellow"],
        annotation_text="Train/Test Split",
        annotation_font_color=THEME["yellow"],
    )
    return fig


# ── RSI Chart ─────────────────────────────────────────────────────────────────
def rsi_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = _base_fig(title=f"<b>{ticker}</b> — RSI (14)", height=280)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI_14"],
        name="RSI", mode="lines",
        line=dict(color=THEME["purple"], width=1.5),
        fill="tozeroy", fillcolor="rgba(168,85,247,0.08)",
    ))
    fig.add_hline(y=70, line_dash="dot", line_color=THEME["red"],
                  annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dot", line_color=THEME["green"],
                  annotation_text="Oversold (30)")
    fig.update_yaxes(range=[0, 100])
    return fig


# ── MACD Chart ────────────────────────────────────────────────────────────────
def macd_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)

    colors = [THEME["green"] if v >= 0 else THEME["red"] for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"],
                         marker_color=colors, name="Histogram", opacity=0.7))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
                             name="MACD", line=dict(color=THEME["accent"], width=1.4)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"],
                             name="Signal", line=dict(color=THEME["orange"], width=1.4)))

    fig.update_layout(**BASE_LAYOUT, title=f"<b>{ticker}</b> — MACD", height=280)
    return fig


# ── Bollinger Bands ───────────────────────────────────────────────────────────
def bollinger_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = _base_fig(title=f"<b>{ticker}</b> — Bollinger Bands", height=380)

    fig.add_trace(go.Scatter(
        x=list(df.index) + list(reversed(df.index)),
        y=list(df["BB_Upper"]) + list(reversed(df["BB_Lower"])),
        fill="toself", fillcolor="rgba(0,212,255,0.05)",
        line=dict(color="rgba(0,0,0,0)"), name="BB Band", hoverinfo="skip",
    ))
    for col, color, dash, name in [
        ("BB_Upper",  THEME["red"],    "dot",   "Upper"),
        ("BB_Middle", THEME["yellow"], "dash",  "Middle (SMA20)"),
        ("BB_Lower",  THEME["green"],  "dot",   "Lower"),
        ("Close",     THEME["accent"], "solid", "Price"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=name,
                line=dict(color=color, width=1.3, dash=dash),
            ))
    return fig


# ── Loss Curves ───────────────────────────────────────────────────────────────
def loss_curves(history) -> go.Figure:
    fig = _base_fig(title="Training vs Validation Loss", height=340)
    epochs = list(range(1, len(history.history["loss"]) + 1))

    fig.add_trace(go.Scatter(x=epochs, y=history.history["loss"],
                             name="Train Loss", line=dict(color=THEME["accent"], width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=history.history["val_loss"],
                             name="Val Loss", line=dict(color=THEME["orange"], width=2)))
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="Loss (Huber)")
    return fig


# ── Portfolio Donut ───────────────────────────────────────────────────────────
def portfolio_donut(holdings_df: pd.DataFrame) -> go.Figure:
    colors = px.colors.qualitative.Dark24
    fig = go.Figure(go.Pie(
        labels=holdings_df["Ticker"],
        values=holdings_df["Current Value"],
        hole=0.55,
        marker=dict(colors=colors[:len(holdings_df)]),
        textinfo="label+percent",
        textfont=dict(family=THEME["font"], size=11),
    ))
    fig.update_layout(**BASE_LAYOUT, title="Portfolio Allocation", height=360)
    return fig


# ── Signal Scatter ────────────────────────────────────────────────────────────
def signal_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = _base_fig(title=f"<b>{ticker}</b> — AI Trading Signals", height=440)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Price", mode="lines",
        line=dict(color=THEME["text_muted"], width=1),
    ))

    for label, color, symbol in [("BUY", THEME["green"], "triangle-up"),
                                  ("SELL", THEME["red"],   "triangle-down")]:
        mask = df["Signal_Label"] == label
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df[mask].index, y=df[mask]["Close"],
                name=label, mode="markers",
                marker=dict(symbol=symbol, size=10,
                            color=color, line=dict(width=1, color="white")),
            ))
    return fig


# ── Correlation Heatmap ───────────────────────────────────────────────────────
def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = ["Close", "RSI_14", "MACD", "BB_Width", "ATR_14", "OBV", "Daily_Return"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(**BASE_LAYOUT, title="Indicator Correlation Matrix", height=420)
    return fig
