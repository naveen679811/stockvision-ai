"""
app/api.py
──────────
FastAPI REST API — serves LSTM predictions programmatically.
Enables integration with external dashboards or mobile apps.

Run: uvicorn app.api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import os
import sys
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.data_handler import (
    fetch_stock_data, add_technical_indicators,
    prepare_lstm_data, generate_trading_signals,
    get_stock_info, POPULAR_STOCKS,
)
from models.lstm_model import (
    build_lstm_model, train_model, evaluate_model,
    forecast_future, save_model, list_saved_models,
)

app = FastAPI(
    title="StockVision AI API",
    description="LSTM-powered stock price prediction REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)


# ── Request / Response Models ──────────────────────────────────────────────────
class TrainRequest(BaseModel):
    ticker:      str              = Field(..., example="AAPL")
    period:      str              = Field("1y", example="1y")
    window:      int              = Field(60, ge=20, le=120)
    epochs:      int              = Field(80, ge=10, le=300)
    batch_size:  int              = Field(32)
    dropout:     float            = Field(0.2, ge=0.0, le=0.5)
    lstm_units:  list[int]        = Field([128, 64, 32])
    future_days: int              = Field(30, ge=1, le=180)


class PredictRequest(BaseModel):
    ticker:      str  = Field(..., example="TSLA")
    period:      str  = Field("1y")
    future_days: int  = Field(30)


# ── In-memory model registry ───────────────────────────────────────────────────
MODEL_REGISTRY: dict = {}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "StockVision AI API", "version": "1.0.0"}


@app.get("/stocks/popular", tags=["Data"])
def popular_stocks():
    return POPULAR_STOCKS


@app.get("/stocks/{ticker}/info", tags=["Data"])
def stock_info(ticker: str):
    try:
        return get_stock_info(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/stocks/{ticker}/price", tags=["Data"])
def stock_price(ticker: str, period: str = "3mo", interval: str = "1d"):
    try:
        df = fetch_stock_data(ticker.upper(), period=period, interval=interval)
        records = df[["Open", "High", "Low", "Close", "Volume"]].tail(100).reset_index()
        records["Date"] = records["Date"].astype(str)
        return records.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stocks/{ticker}/signals", tags=["Analysis"])
def trading_signals(ticker: str, period: str = "6mo"):
    try:
        df = fetch_stock_data(ticker.upper(), period=period)
        df = add_technical_indicators(df)
        df = generate_trading_signals(df)
        result = df[["Close", "RSI_14", "MACD", "Signal_Label"]].tail(30).reset_index()
        result["Date"] = result["Date"].astype(str)
        latest = df["Signal_Label"].iloc[-1]
        return {"ticker": ticker, "latest_signal": latest, "history": result.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/model/train", tags=["Model"])
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Train an LSTM model for the given ticker.
    Returns immediately; training happens in background.
    Poll /model/status/{ticker} for completion.
    """
    MODEL_REGISTRY[req.ticker] = {"status": "training", "metrics": None}
    background_tasks.add_task(_train_background, req)
    return {"message": f"Training started for {req.ticker}", "status": "training"}


def _train_background(req: TrainRequest):
    try:
        df = fetch_stock_data(req.ticker, period=req.period)
        df = add_technical_indicators(df)

        feature_cols = [c for c in [
            "Open", "High", "Low", "Volume",
            "SMA_20", "SMA_50", "EMA_12", "EMA_26",
            "RSI_14", "MACD", "MACD_Signal",
            "BB_Upper", "BB_Lower", "BB_Width", "ATR_14",
        ] if c in df.columns]

        X_tr, y_tr, X_te, y_te, sc_f, sc_t, ts = prepare_lstm_data(
            df, feature_cols=feature_cols, window=req.window
        )
        val_split = int(len(X_tr) * 0.85)

        model = build_lstm_model(
            input_shape=(X_tr.shape[1], X_tr.shape[2]),
            units=req.lstm_units, dropout=req.dropout,
        )
        history, model = train_model(
            model,
            X_tr[:val_split], y_tr[:val_split],
            X_tr[val_split:], y_tr[val_split:],
            epochs=req.epochs, batch_size=req.batch_size,
            ticker=req.ticker,
        )
        metrics  = evaluate_model(model, X_te, y_te, sc_t)
        future   = forecast_future(model, X_te[-1], sc_t, n_steps=req.future_days)

        MODEL_REGISTRY[req.ticker] = {
            "status":   "ready",
            "metrics":  {k: v for k, v in metrics.items()
                         if k not in ("predictions", "actual")},
            "forecast": future.tolist(),
        }
    except Exception as e:
        MODEL_REGISTRY[req.ticker] = {"status": "error", "error": str(e)}
        logger.error(f"Training failed for {req.ticker}: {e}")


@app.get("/model/status/{ticker}", tags=["Model"])
def model_status(ticker: str):
    result = MODEL_REGISTRY.get(ticker.upper())
    if not result:
        raise HTTPException(status_code=404, detail="No model found for this ticker.")
    return result


@app.get("/model/list", tags=["Model"])
def list_models():
    saved = list_saved_models()
    return [{"name": n, "path": p} for n, p in saved]
